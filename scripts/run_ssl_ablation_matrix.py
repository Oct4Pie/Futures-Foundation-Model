#!/usr/bin/env python3
"""Generate or execute the bounded MantisV2 Stage-2 plumbing ablation matrix.

This runner is intentionally smoke-only. It establishes that direct vanilla lineage,
preprocessing contracts, context lengths, artifact capture, and exact checkpointing all work.
It does not produce promotable checkpoints and never probes or reads the declared holdout.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'scripts' / 'train_ssl_local.py'
PREPROCESSING = {
    'channel': 'per_window_per_channel_zscore_v1',
    'shared': 'per_window_shared_ohlc_zscore_v1',
}


def _atomic_json(path, value):
    path = Path(path)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def build_matrix(*, data_dir, output_dir, admission_report, seqs=(64, 128, 256),
                 preprocessing=('channel', 'shared'), ticker='ES', tf='3min',
                 device='cuda', seed=0):
    rows = []
    for prep_name in preprocessing:
        prep = PREPROCESSING[prep_name]
        for seq in seqs:
            name = f'v2_stage2_vanilla_{prep_name}_seq{int(seq)}_smoke'
            output = Path(output_dir).resolve() / f'{name}.pt'
            command = [
                sys.executable, str(TRAIN), '--stage', 'contrastive',
                '--lineage', 'vanilla', '--data-dir', str(Path(data_dir).resolve()),
                '--output', str(output), '--preprocessing', prep, '--seq', str(int(seq)),
                '--contrastive-objective', 'elapsed_time_v2',
                '--tickers', ticker, '--tfs', tf, '--device', device, '--seed', str(int(seed)),
                '--controls', '', '--smoke', '--no-probe', '--batch', '8',
                '--admission-report', str(Path(admission_report).resolve()),
            ]
            rows.append({'name': name, 'stage': 'contrastive', 'lineage': 'vanilla',
                         'contrastive_objective': 'elapsed_time_v2',
                         'preprocessing': prep, 'seq': int(seq), 'ticker': ticker, 'tf': tf,
                         'output': str(output), 'command': command, 'status': 'pending'})
    return rows


def run_matrix(rows, manifest_path):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 'ffm_ssl_ablation_matrix_v1',
                'purpose': 'plumbing_smoke_only_not_promotable',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'runs': rows}
    _atomic_json(manifest_path, manifest)
    for row in rows:
        output = Path(row['output'])
        output.parent.mkdir(parents=True, exist_ok=True)
        log = Path(str(output) + '.log')
        row.update(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                   log=str(log))
        _atomic_json(manifest_path, manifest)
        with log.open('w') as stream:
            result = subprocess.run(row['command'], cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT)
        row.update(status=('complete' if result.returncode == 0 else 'failed'),
                   returncode=int(result.returncode),
                   finished_utc=datetime.now(timezone.utc).isoformat())
        _atomic_json(manifest_path, manifest)
        if result.returncode:
            raise RuntimeError(f"smoke run failed: {row['name']} (see {log})")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', default='output/ssl_ablation_smoke')
    p.add_argument('--manifest')
    p.add_argument('--seqs', default='64,128,256')
    p.add_argument('--preprocessing', default='channel,shared')
    p.add_argument('--ticker', default='ES')
    p.add_argument('--tf', default='3min')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--admission-report', required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    preps = tuple(x for x in a.preprocessing.split(',') if x)
    unknown = sorted(set(preps) - set(PREPROCESSING))
    if unknown:
        raise ValueError(f'unknown preprocessing aliases: {unknown}')
    seqs = tuple(int(x) for x in a.seqs.split(',') if x)
    if not seqs or any(x not in (64, 128, 256) for x in seqs):
        raise ValueError('smoke seqs must be drawn from 64,128,256')
    rows = build_matrix(data_dir=a.data_dir, output_dir=a.output_dir,
                        admission_report=a.admission_report, seqs=seqs,
                        preprocessing=preps, ticker=a.ticker, tf=a.tf,
                        device=a.device, seed=a.seed)
    manifest_path = Path(a.manifest or (Path(a.output_dir) / 'matrix.json')).resolve()
    if a.execute:
        run_matrix(rows, manifest_path)
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest_path, {
            'schema_version': 'ffm_ssl_ablation_matrix_v1',
            'purpose': 'plumbing_smoke_only_not_promotable', 'runs': rows})
        print(f'wrote {len(rows)}-run smoke matrix -> {manifest_path}')


if __name__ == '__main__':
    main()
