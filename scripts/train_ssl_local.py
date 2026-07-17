#!/usr/bin/env python3
"""Crash-safe local runner for the leak-audited Mantis SSL stage sequence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOTS = ('ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN')
TFS = ('1min', '3min', '5min', '15min', '30min', '60min')
PREVIOUS_STAGE = {'contrastive': 'mask', 'forecast': 'contrastive'}
CANONICAL_VAL_START = '2024-01-01'
CANONICAL_OOS_START = '2025-07-01'
TOURNAMENT_TRAIN_START = '2019-07-01'
TOURNAMENT_VAL_START = '2024-07-01'
TOURNAMENT_OOS_START = '2025-07-01'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_CHOICES = ('per_window_per_channel_zscore_v1',
                         'per_window_shared_ohlc_zscore_v1',
                         'per_window_log_price_rel_volume_zscore_v1')

STAGES = {
    'mask': dict(pretext='mask', batch=768, epochs=60, steps_per_epoch=200,
                 lr=1e-4, weight_decay=0.05, patience=8, freeze_encoder_layers=0,
                 seq=64, max_jitter=16, new_channels=5, mask_ratio=0.4,
                 span_mean=0.0, span_max=10),
    'contrastive': dict(pretext='contrastive', batch=128, epochs=60, steps_per_epoch=200,
                        lr=1e-4, weight_decay=0.05, patience=8,
                        freeze_encoder_layers=0, seq=64, max_jitter=16,
                        new_channels=5, pos_deltas=(2, 16, 64), far_min=512,
                        temperature=0.1, metrics_n=768,
                        contrastive_objective='elapsed_time_v2',
                        positive_gap_fractions=(0.6, 1.0, 2.0),
                        max_positive_overlap=0.5, positive_tolerance_fraction=0.20,
                        negative_min_contexts=4.0, sync_exclusion_minutes=60.0,
                        min_valid_negatives=1, vol_weight=0.0),
    'forecast': dict(pretext='forecast', batch=512, epochs=60, steps_per_epoch=200,
                     lr=1e-4, weight_decay=0.05, patience=8,
                     freeze_encoder_layers=3, seq=64, max_jitter=16,
                     new_channels=5, horizons=(5, 10, 20, 25),
                     context_lengths=(64, 100, 150, 200), objective='candle_mse',
                     dir_weight=0.0),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _git(*args):
    r = subprocess.run(['git', *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def _experiment_source_files():
    """Tracked + untracked experiment source, excluding data/output/secrets by construction."""
    root = Path(_git('rev-parse', '--show-toplevel') or '.').resolve()
    listed = (_git('ls-files', '--cached', '--others', '--exclude-standard') or '').splitlines()
    roots = ('futures_foundation/', 'scripts/', 'tests/', 'colab/')
    root_files = {'requirements.txt', 'setup.py', 'pyproject.toml', 'pytest.ini', 'AGENTS.md'}
    files = []
    for rel in sorted(set(listed)):
        path = root / rel
        if path.is_file() and (rel.startswith(roots) or rel in root_files):
            files.append((rel, path))
    return root, files


def _source_tree_sha256(files):
    h = hashlib.sha256()
    for rel, path in files:
        h.update(rel.encode() + b'\0')
        h.update(_sha256(path).encode() + b'\n')
    return h.hexdigest()


def _write_source_archive(output: Path, root: Path, files):
    path = Path(str(output) + '.source.tar.gz')
    tmp = Path(str(path) + '.tmp')
    with tarfile.open(tmp, 'w:gz') as tf:
        for rel, source in files:
            tf.add(source, arcname=rel, recursive=False)
    os.replace(tmp, path)
    return path


def _write_dependency_lock(output: Path):
    path = Path(str(output) + '.environment.txt')
    result = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'],
                            capture_output=True, text=True)
    if result.returncode == 0:
        content = '# capture=pip-freeze-all\n' + result.stdout
    else:
        # uv/externally managed environments may intentionally omit the pip module. Installed
        # distribution metadata still provides an exact name/version lock for reproducibility.
        from importlib.metadata import distributions
        locked = sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()
                         if d.metadata.get('Name')}, key=str.lower)
        if not locked:
            raise RuntimeError(f'failed to capture dependency lock: {result.stderr.strip()}')
        content = '# capture=importlib-metadata (pip module unavailable)\n' + '\n'.join(locked) + '\n'
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(content)
    os.replace(tmp, path)
    return path


def _resolve_lineage(stage, requested, warm):
    lineage = ('vanilla' if stage == 'mask' else 'canonical') if requested == 'auto' else requested
    if stage == 'mask' and lineage != 'vanilla':
        raise ValueError('mask has no predecessor and must use --lineage vanilla')
    if lineage == 'canonical' and stage in PREVIOUS_STAGE and warm is None:
        raise ValueError(f'canonical {stage} requires --warm-checkpoint')
    if lineage == 'diagnostic' and (stage not in PREVIOUS_STAGE or warm is None):
        raise ValueError('diagnostic lineage requires a staged predecessor checkpoint')
    if lineage == 'vanilla' and warm is not None:
        raise ValueError('--lineage vanilla cannot also specify --warm-checkpoint')
    return lineage


def _versions():
    import numpy, pandas, sklearn, torch
    import mantis
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'torch_cuda': torch.version.cuda, 'numpy': numpy.__version__,
            'pandas': pandas.__version__, 'sklearn': sklearn.__version__,
            'mantis': getattr(mantis, '__version__', '1.0.0'),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def _atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=float) + '\n')
    os.replace(tmp, path)


def run(args):
    from futures_foundation.finetune import ssl
    from futures_foundation.finetune.pretext._torch.backbone import resolve_mantis_version

    resolve_mantis_version(args.model_id, args.model_version)  # fail before reading/training data
    if args.protocol == 'canonical_v2':
        if args.model_version == 'v2' and (args.train_start is not None or
                                           args.val_start != CANONICAL_VAL_START or
                                           args.holdout_start != CANONICAL_OOS_START):
            raise ValueError(
                f"MantisV2 lineage dates are immutable: train < {CANONICAL_VAL_START}, "
                f"validation [{CANONICAL_VAL_START}, {CANONICAL_OOS_START}), "
                f"OOS >= {CANONICAL_OOS_START}")
    elif (args.train_start, args.val_start, args.holdout_start) != (
            TOURNAMENT_TRAIN_START, TOURNAMENT_VAL_START, TOURNAMENT_OOS_START):
        raise ValueError(
            'foundation_5y1y1y_v1 requires train=2019-07-01, '
            'validation=2024-07-01, OOS=2025-07-01')

    data_dir = Path(args.data_dir).resolve()
    corpus_manifest = data_dir / 'MANIFEST.json'
    tournament_cache = data_dir / 'TOURNAMENT_CACHE.json'
    if not corpus_manifest.is_file() and tournament_cache.is_file():
        if args.protocol != 'foundation_5y1y1y_v1':
            raise ValueError('the bounded tournament cache cannot serve a canonical lineage')
        corpus_manifest = tournament_cache
    if not corpus_manifest.is_file():
        raise FileNotFoundError(f"sealed corpus manifest missing: {corpus_manifest}")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite checkpoint without --resume: {output}")
    if args.resume and not Path(str(output) + '.train.pt').is_file():
        raise FileNotFoundError(
            f"exact resume requires full training state: {output}.train.pt; "
            "an encoder-only checkpoint may be supplied as --warm-checkpoint instead")
    if output.name in {'mantis_ssl_ohlcv.pt', 'mantis_ssl_seq2seq.pt',
                       'mantis_ssl_ctr_seq2seq.pt', 'mantis_ssl_regime.pt'}:
        raise ValueError("output name is protected; use a distinct experiment checkpoint")

    cfg = dict(STAGES[args.stage])
    if args.no_probe and not args.smoke:
        raise ValueError("--no-probe is permitted only with --smoke")
    for key in ('batch', 'epochs', 'steps_per_epoch', 'lr', 'weight_decay', 'patience',
                'mask_ratio', 'span_mean', 'span_max', 'feature_anchor_weight',
                'temperature', 'crop_max',
                'aug_noise', 'aug_scale', 'aug_tmask', 'dir_weight',
                'freeze_encoder_layers', 'objective'):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.seq is not None:
        cfg['seq'] = int(args.seq)
    if args.context_lengths:
        cfg['context_lengths'] = tuple(int(x) for x in args.context_lengths.split(','))
    if args.contrastive_objective:
        if args.stage != 'contrastive':
            raise ValueError('--contrastive-objective is valid only for stage contrastive')
        cfg['contrastive_objective'] = args.contrastive_objective
        cfg['vol_weight'] = 1.0 if args.contrastive_objective == 'bar_offset_v1' else 0.0
    if args.contrastive_reserve_contexts is not None:
        if args.stage != 'contrastive':
            raise ValueError('--contrastive-reserve-contexts is valid only for stage contrastive')
        cfg['contrastive_reserve_contexts'] = float(args.contrastive_reserve_contexts)
    cfg['preprocessing'] = args.preprocessing
    cfg['allow_aligned_market_gaps'] = args.protocol == 'foundation_5y1y1y_v1'
    if args.protocol == 'foundation_5y1y1y_v1' and args.stage == 'contrastive':
        # A 2x gap needs a 3-context parent. At seq=256 that exceeds some CL front-contract
        # segments at 60m. Never cross a roll merely to satisfy an augmentation; retain one
        # overlapping (0.5x) and one non-overlapping (1.0x) elapsed-time positive instead.
        cfg['positive_gap_fractions'] = (0.5, 1.0)
    if args.smoke:
        cfg.update(epochs=1, steps_per_epoch=2, batch=min(cfg['batch'], 16), metrics_n=16,
                   val_batches=1)
    warm = Path(args.warm_checkpoint).resolve() if args.warm_checkpoint else None
    lineage = _resolve_lineage(args.stage, args.lineage, warm)
    if warm is not None:
        if not warm.is_file() or warm.stat().st_size < 1024:
            raise FileNotFoundError(f"warm checkpoint missing or Git LFS pointer: {warm}")
        if warm == output:
            raise ValueError("warm checkpoint and output must be distinct")
        parent_run = warm.with_suffix(warm.suffix + '.run.json')
        if not parent_run.is_file():
            raise FileNotFoundError(f"warm checkpoint has no auditable run metadata: {parent_run}")
        parent = json.loads(parent_run.read_text())
        expected = PREVIOUS_STAGE[args.stage]
        if parent.get('status') != 'complete' or parent.get('stage') != expected:
            raise ValueError(f"{args.stage} must warm from a completed {expected} run; got "
                             f"status={parent.get('status')} stage={parent.get('stage')}")
        parent_cfg = parent.get('config', {})
        if parent_cfg.get('model_version') != args.model_version:
            raise ValueError("warm checkpoint architecture does not match this run")
        if parent_cfg.get('preprocessing', PREPROCESSING_CHOICES[0]) != cfg['preprocessing']:
            raise ValueError('warm checkpoint preprocessing contract does not match this run')
        if (parent.get('train_start') != args.train_start or
                parent.get('val_start') != args.val_start or
                parent.get('holdout_start') != args.holdout_start):
            raise ValueError("warm checkpoint uses different train/validation/OOS boundaries")
        strict = parent.get('strict_probe', {})
        if strict.get('protocol') != 'expanding_walk_forward':
            raise ValueError("warm checkpoint lacks strict expanding-walk-forward probe attestation")
        if strict.get('checkpoint_sha256') != _sha256(warm):
            raise ValueError("strict probe attestation does not match the warm checkpoint bytes")
        if lineage == 'canonical' and strict.get('passed') is not True:
            raise ValueError("canonical lineage requires a predecessor that passed its strict probe")
        if lineage == 'diagnostic':
            cfg['diagnostic_continuation'] = {
                'non_promotable': True,
                'reason': 'continued_for_stagewise_representation_comparison',
                'parent_strict_probe_passed': bool(strict.get('passed')),
            }
        cfg['backbone_ckpt'] = str(warm)
    cfg.update(device=args.device, seed=args.seed, verbose=True, resume=args.resume,
               model_id=args.model_id, model_version=args.model_version,
               probe_folds=args.probe_folds, probe_seed=args.probe_seed)

    run_path = output.with_suffix(output.suffix + '.run.json')
    source_root, source_files = _experiment_source_files()
    source_tree_sha = _source_tree_sha256(source_files)
    previous = json.loads(run_path.read_text()) if args.resume and run_path.is_file() else None
    if previous is not None and previous.get('source_tree_sha256') != source_tree_sha:
        raise ValueError('exact resume refused: experiment source differs from the original run')
    if previous is not None:
        if previous.get('lineage') != lineage or previous.get('protocol') != args.protocol:
            raise ValueError('exact resume refused: lineage changed')
        old_cfg, new_cfg = dict(previous.get('config') or {}), dict(cfg)
        old_cfg.pop('resume', None); new_cfg.pop('resume', None)
        # JSON canonicalization removes tuple/list representation differences without weakening
        # the comparison. Data/corpus boundaries are checked separately below.
        if json.dumps(old_cfg, sort_keys=True, default=float) != json.dumps(new_cfg, sort_keys=True,
                                                                            default=float):
            raise ValueError('exact resume refused: run configuration changed')
        if (previous.get('data_dir') != str(data_dir) or
                previous.get('corpus_manifest_sha256') != _sha256(corpus_manifest)):
            raise ValueError('exact resume refused: corpus identity changed')
    if previous is None:
        source_archive = _write_source_archive(output, source_root, source_files)
        dependency_lock = _write_dependency_lock(output)
    else:
        source_archive = Path(previous['source_archive']['path'])
        dependency_lock = Path(previous['dependency_lock']['path'])
        for artifact in (source_archive, dependency_lock):
            if not artifact.is_file():
                raise FileNotFoundError(f'exact resume provenance artifact missing: {artifact}')
    metadata = {
        'schema_version': 'ffm_ssl_run_v2', 'status': 'running',
        'started_utc': (previous or {}).get('started_utc', datetime.now(timezone.utc).isoformat()),
        'resumed_utc': (datetime.now(timezone.utc).isoformat() if previous else None),
        'stage': args.stage, 'output': str(output),
        'protocol': args.protocol,
        'lineage': lineage,
        'train_start': args.train_start,
        'val_start': args.val_start, 'holdout_start': args.holdout_start,
        'holdout_policy': (f'V2 train [{args.train_start or "beginning"}, {args.val_start}); validation '
                           f'[{args.val_start}, {args.holdout_start}); '
                           f'>= {args.holdout_start} excluded from every V2 training/probe decision'),
        'data_dir': str(data_dir), 'corpus_manifest_sha256': _sha256(corpus_manifest),
        'source_snapshot_sha256': (
            json.loads(corpus_manifest.read_text()).get('source_snapshot_sha256') or
            json.loads(corpus_manifest.read_text()).get('source_manifest_sha256')),
        'tickers': list(args.tickers), 'timeframes': list(args.tfs),
        'warm_checkpoint': (None if warm is None else
                            {'path': str(warm), 'sha256': _sha256(warm)}),
        'git_commit': _git('rev-parse', 'HEAD'),
        'source_tree_sha256': source_tree_sha,
        'source_archive': {'path': str(source_archive), 'sha256': _sha256(source_archive)},
        'dependency_lock': {'path': str(dependency_lock), 'sha256': _sha256(dependency_lock)},
        'command': [sys.executable, *sys.argv],
        'versions': _versions(), 'config': cfg,
    }
    _atomic_json(run_path, metadata)
    try:
        verdict = ssl.loop_ssl(
            data_dir=str(data_dir), out_path=str(output), tickers=args.tickers, tfs=args.tfs,
            controls=args.controls, probe=not args.no_probe, train_start=args.train_start,
            val_start=args.val_start,
            holdout_start=args.holdout_start,
            val_frac=args.val_frac, **cfg)
    except BaseException as exc:
        metadata.update(status='failed', finished_utc=datetime.now(timezone.utc).isoformat(),
                        error=f'{type(exc).__name__}: {exc}')
        _atomic_json(run_path, metadata)
        raise
    report_path = Path(str(output) + '.report.json')
    strict_probe = None
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        probe_result = report.get('probe') or {}
        if probe_result.get('probe_protocol') == 'expanding_walk_forward':
            strict_probe = {
                'protocol': 'expanding_walk_forward',
                'report': str(report_path), 'report_sha256': _sha256(report_path),
                'checkpoint_sha256': _sha256(output),
                'passed': bool((report.get('history') or [{}])[0].get('gate_ok')),
            }
    bundle_path = Path(str(output) + '.bundle.pt')
    train_state_path = Path(str(output) + '.train.pt')
    metadata.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                    checkpoint_sha256=_sha256(output),
                    deployment_bundle={'path': str(bundle_path), 'sha256': _sha256(bundle_path)},
                    training_state={'path': str(train_state_path), 'sha256': _sha256(train_state_path)},
                    verdict=verdict,
                    strict_probe=strict_probe)
    _atomic_json(run_path, metadata)
    return verdict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=tuple(STAGES), required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--warm-checkpoint')
    p.add_argument('--lineage', choices=('auto', 'canonical', 'vanilla', 'diagnostic'),
                   default='auto',
                   help=('canonical requires the promoted predecessor; vanilla is an explicit '
                         'ablation; diagnostic continues a failed staged predecessor and is never '
                         'promotion-eligible'))
    p.add_argument('--tickers', default=','.join(ROOTS))
    p.add_argument('--tfs', default=','.join(TFS))
    p.add_argument('--model-id', default='paris-noah/MantisV2')
    p.add_argument('--model-version', choices=('v1', 'v2'), default='v2')
    p.add_argument('--protocol', choices=('canonical_v2', 'foundation_5y1y1y_v1'),
                   default='canonical_v2')
    p.add_argument('--train-start', help='inclusive lower bound for adaptation data')
    p.add_argument('--val-start', default=CANONICAL_VAL_START)
    p.add_argument('--holdout-start', default=CANONICAL_OOS_START)
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--controls', default='', help='comma-separated shuffle,random')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--probe-folds', type=int, default=5)
    p.add_argument('--probe-seed', type=int,
                   help='fixed validation sampling seed, independent of training seed')
    p.add_argument('--preprocessing', choices=PREPROCESSING_CHOICES,
                   default=PREPROCESSING_CHOICES[0])
    p.add_argument('--seq', type=int, choices=(64, 128, 256))
    p.add_argument('--context-lengths', help='comma-separated forecast contexts, e.g. 64,128,256')
    p.add_argument('--contrastive-objective', choices=('elapsed_time_v2', 'bar_offset_v1'),
                   help='versioned Stage-2 objective; legacy v1 is comparison-only')
    p.add_argument('--contrastive-reserve-contexts', type=float,
                   help='lock the eligible anchor universe across Stage-2 objective comparisons')
    p.add_argument('--no-probe', action='store_true', help='smoke/debug only; full runs must probe')
    p.add_argument('--batch', type=int)
    p.add_argument('--epochs', type=int)
    p.add_argument('--steps-per-epoch', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--weight-decay', type=float)
    p.add_argument('--mask-ratio', type=float)
    p.add_argument('--span-mean', type=float,
                   help='Stage-1 contiguous masked-span mean; zero restores scattered-bar masking')
    p.add_argument('--span-max', type=int,
                   help='Stage-1 maximum contiguous masked-span length')
    p.add_argument('--feature-anchor-weight', type=float,
                   help='Stage-1 frozen-teacher embedding anchor; zero disables it')
    p.add_argument('--temperature', type=float)
    p.add_argument('--crop-max', type=float)
    p.add_argument('--aug-noise', type=float)
    p.add_argument('--aug-scale', type=float)
    p.add_argument('--aug-tmask', type=float)
    p.add_argument('--objective', choices=('candle_mse', 'candle_direction'),
                   help='Stage-3 forecast supervision objective')
    p.add_argument('--dir-weight', type=float,
                   help='Stage-3 auxiliary direction-loss weight')
    p.add_argument('--freeze-encoder-layers', type=int,
                   help='Freeze tokenizer plus the first N encoder layers')
    p.add_argument('--patience', type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.tickers = tuple(x for x in a.tickers.split(',') if x)
    a.tfs = tuple(x for x in a.tfs.split(',') if x)
    a.controls = tuple(x for x in a.controls.split(',') if x)
    if a.objective is not None and a.stage != 'forecast':
        p.error('--objective is valid only for stage forecast')
    run(a)


if __name__ == '__main__':
    main()
