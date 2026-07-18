#!/usr/bin/env python3
"""Compare frozen Mantis checkpoints on one shared, purged pre-holdout probe sample."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from futures_foundation.finetune import ssl, ssl_probe
from futures_foundation.finetune.native_contracts import verify_admission_report


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=float) + '\n')
    tmp.replace(path)


def _attest_run(run_path, report_path, report):
    """Bind a strict benchmark result to a completed stage run for lineage authorization."""
    run_path = Path(run_path).resolve()
    run = json.loads(run_path.read_text())
    if run.get('status') != 'complete':
        raise ValueError("only a completed run can receive strict probe attestation")
    checkpoint = Path(run['output']).resolve()
    result = report['results'].get(checkpoint.name)
    if result is None or Path(result['path']).resolve() != checkpoint:
        raise ValueError("benchmark does not contain the run checkpoint")
    if result['sha256'] != _sha256(checkpoint):
        raise ValueError("benchmark checkpoint hash mismatch")
    cfg = report['config']
    if run.get('val_start') != cfg['val_start'] or run.get('holdout_start') != cfg['holdout_start']:
        raise ValueError("benchmark date boundaries do not match the run")

    p, stage = result['probe'], run.get('stage')
    if stage == 'mask':
        passed = p['mean_core_delta'] > 0.0
    elif stage == 'contrastive':
        passed = p['descriptive_delta'] >= -1e-9
    elif stage == 'forecast':
        passed = (p['descriptive_delta'] >= -1e-9 and
                  p['fwd_absmove_delta'] > 0.0 and p['fwd_dir_delta'] >= 0.0)
    else:
        raise ValueError(f"unsupported stage for attestation: {stage}")
    run['strict_probe'] = {
        'protocol': 'expanding_walk_forward', 'passed': bool(passed),
        'report': str(Path(report_path).resolve()),
        'report_sha256': _sha256(Path(report_path)),
        'checkpoint_sha256': result['sha256'],
        'metrics': {
            'mean_core_delta': p['mean_core_delta'],
            'descriptive_delta': p['descriptive_delta'],
            'fwd_absmove_delta': p['fwd_absmove_delta'],
            'fwd_dir_delta': p['fwd_dir_delta'],
        },
        'attested_utc': datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(run_path, run)
    print(f"strict probe attestation -> {run_path} passed={passed}")
    return passed


def benchmark(data_dir, checkpoints, *, output, tickers, tfs, holdout_start='2026-01-01',
              val_start=None, val_frac=0.1, seq=64, fwd_k=16, max_windows=20000,
              batch=512, device='cuda', seed=0, folds=5,
              model_id='paris-noah/Mantis-8M', model_version=None, attest_run=None,
              admission_reports=()):
    if len(admission_reports) != len(checkpoints):
        raise ValueError('one --admission-report is required for each --checkpoint in order')
    arm_key = 'mantis_v2' if model_version == 'v2' else 'mantis_v1'
    admissions = []
    for checkpoint, report in zip(checkpoints, admission_reports):
        admissions.append(verify_admission_report(
            report, arm_key=arm_key, track='C',
            route='historical_custom_representation_extraction', require_training=False,
            required_artifacts={'checkpoint': checkpoint},
        ))
    streams, big, _, va, groups = ssl._load_assemble(
        data_dir, tickers, tfs, seq, fwd_k, val_frac, holdout_start, True,
        val_start=val_start, return_groups=True)
    sampled, sampled_groups = ssl._balanced_group_sample(
        va, groups['val_bounds'], max_windows, seed, return_group_ids=True)
    keep = ssl_probe.non_overlapping_rows(
        sampled, seq + fwd_k, group_ids=sampled_groups)
    used = sampled[keep]
    used_groups = sampled_groups[keep]
    sampled_timestamps = ssl._probe_start_timestamps(
        sampled, sampled_groups, streams, groups['row_bounds'])
    used_timestamps = sampled_timestamps[keep]
    probe_span_ns = int(max(pd.Timedelta(s['tf']).value for s in streams)
                        * (seq + fwd_k))
    if len(used) < 100:
        raise ValueError(f"only {len(used)} non-overlapping probe windows; need at least 100")

    from futures_foundation.finetune import _ssl_torch
    targets = ssl_probe.targets_from_windows(big, used, seq, fwd_k=fwd_k)
    splits = ssl_probe.walk_forward_splits(
        used, used_groups, folds=folds, span=seq + fwd_k,
        timestamps=used_timestamps, span_ns=probe_span_ns)
    vanilla, vanilla_used = _ssl_torch.embed_encoder(
        big, used, seq, ckpt=None, model_id=model_id, model_version=model_version,
        device=device, batch=batch,
        max_windows=len(used), seed=seed)
    if not np.array_equal(vanilla_used, used):
        raise RuntimeError("vanilla embedding sample drifted from the sealed probe rows")

    results = {}
    for spec, admission in zip(checkpoints, admissions):
        path = Path(spec).resolve()
        if not path.is_file() or path.stat().st_size < 1024:
            raise FileNotFoundError(f"checkpoint is missing or a Git LFS pointer: {path}")
        emb, ckpt_used = _ssl_torch.embed_encoder(
            big, used, seq, ckpt=str(path), model_id=model_id, model_version=model_version,
            device=device, batch=batch,
            max_windows=len(used), seed=seed)
        if not np.array_equal(ckpt_used, used):
            raise RuntimeError(f"checkpoint sample drifted: {path}")
        results[path.name] = {
            'path': str(path), 'sha256': _sha256(path),
            'admission': {
                'integrity': admission['integrity'],
                'registry_sha256': admission['registry_sha256'],
                'dossier_sha256': admission['dossier_sha256'],
            },
            'probe': ssl_probe.compare(
                emb, vanilla, targets, seed=seed, folds=folds, splits=splits),
        }

    corpus_manifest = Path(data_dir) / 'MANIFEST.json'
    report = {
        'schema_version': 'ffm_ssl_checkpoint_benchmark_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'holdout_policy': (f'train < {val_start}; validation [{val_start}, {holdout_start}); '
                           f'rows >= {holdout_start} excluded'),
        'data_dir': str(Path(data_dir).resolve()),
        'corpus_manifest_sha256': (_sha256(corpus_manifest) if corpus_manifest.exists() else None),
        'tickers': list(tickers), 'timeframes': list(tfs), 'stream_count': len(streams),
        'bars_loaded': int(len(big)), 'validation_candidate_windows': int(len(va)),
        'probe_windows': int(len(used)), 'probe_span_bars': int(seq + fwd_k),
        'probe_protocol': 'expanding_walk_forward',
        'probe_embargo_bars': int(seq + fwd_k),
        'probe_max_span_ns': probe_span_ns,
        'config': {'val_start': val_start, 'holdout_start': holdout_start,
                   'val_frac': val_frac, 'seq': seq, 'model_id': model_id,
                   'model_version': model_version,
                   'fwd_k': fwd_k, 'max_windows': max_windows, 'batch': batch,
                   'device': device, 'seed': seed, 'folds': folds},
        'results': results,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, report)
    print(f"benchmark report -> {output}")
    for name, result in results.items():
        p = result['probe']
        print(f"{name}: core={p['mean_core_delta']:+.4f} "
              f"desc={p['descriptive_delta']:+.4f} "
              f"fwd_size={p['fwd_absmove_delta']:+.4f} "
              f"fwd_dir={p['fwd_dir_delta']:+.4f}")
    if attest_run is not None:
        _attest_run(attest_run, output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--checkpoint', action='append', required=True)
    p.add_argument('--admission-report', action='append', required=True,
                   help='repeat once per checkpoint in the same order')
    p.add_argument('--output', required=True)
    p.add_argument('--tickers', default='ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN')
    p.add_argument('--tfs', default='1min,3min,5min,15min,30min,60min')
    p.add_argument('--model-id', default='paris-noah/MantisV2')
    p.add_argument('--model-version', choices=('v1', 'v2'), default='v2')
    p.add_argument('--val-start', default='2024-01-01')
    p.add_argument('--holdout-start', default='2025-07-01')
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--seq', type=int, default=64)
    p.add_argument('--fwd-k', type=int, default=16)
    p.add_argument('--max-windows', type=int, default=20000)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--attest-run', help='completed .run.json to bind to this strict report')
    a = p.parse_args()
    benchmark(a.data_dir, a.checkpoint, output=a.output,
              tickers=tuple(a.tickers.split(',')), tfs=tuple(a.tfs.split(',')),
              val_start=a.val_start, holdout_start=a.holdout_start,
              val_frac=a.val_frac, seq=a.seq,
              fwd_k=a.fwd_k, max_windows=a.max_windows, batch=a.batch,
              device=a.device, seed=a.seed, folds=a.folds,
              model_id=a.model_id, model_version=a.model_version,
              attest_run=a.attest_run, admission_reports=a.admission_report)


if __name__ == '__main__':
    main()
