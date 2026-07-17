#!/usr/bin/env python3
"""Run the bounded 256-bar MantisV2 direct-Stage-2 SSL comparison.

The two arms differ only in the Stage-2 objective: elapsed-time InfoNCE versus negative-free
VICReg. Both start from vanilla MantisV2, use the same sealed 2019-2025 corpus, 2019-07 through
2024-06 training interval, 2024-07 through 2025-06 development interval, anchor universe,
preprocessing, seed, optimizer budget, feature anchor, strict probe, and shuffle control. The
2025-07 onward holdout is excluded physically by the training runner.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import train_ssl_local as train  # noqa: E402
from futures_foundation.finetune import ssl as finetune_ssl  # noqa: E402

TRAIN = ROOT / 'scripts' / 'train_ssl_local.py'
OBJECTIVES = ('elapsed_time_v2', 'vicreg_v1')
TARGETS = ('vol', 'trend_eff', 'range_expand', 'fwd_absmove', 'direction', 'fwd_dir')
PRIMARY = ('trend_eff', 'range_expand')


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=float) + '\n')
    os.replace(tmp, path)


def build_matrix(*, data_dir, output_dir, seed=17, epochs=5, steps=50, batch=32,
                 device='cuda', feature_anchor_weight=.1):
    rows = []
    for objective in OBJECTIVES:
        name = f'mantis_v2_stage2_{objective}_seed{int(seed)}_256bar_dev'
        output = Path(output_dir).resolve() / f'{name}.pt'
        command = [
            sys.executable, str(TRAIN), '--stage', 'contrastive', '--lineage', 'vanilla',
            '--protocol', 'foundation_5y1y1y_v1', '--train-start', train.TOURNAMENT_TRAIN_START,
            '--val-start', train.TOURNAMENT_VAL_START,
            '--holdout-start', train.TOURNAMENT_OOS_START,
            '--data-dir', str(Path(data_dir).resolve()), '--output', str(output),
            '--model-id', 'paris-noah/MantisV2', '--model-version', 'v2', '--seq', '256',
            '--preprocessing', 'per_window_per_channel_zscore_v1',
            '--contrastive-objective', objective,
            '--contrastive-reserve-contexts', '2', '--feature-anchor-weight',
            str(float(feature_anchor_weight)), '--device', device, '--seed', str(int(seed)),
            '--probe-seed', '20260704', '--epochs', str(int(epochs)),
            '--steps-per-epoch', str(int(steps)), '--batch', str(int(batch)),
            '--patience', str(int(epochs)), '--probe-folds', '5', '--controls', 'shuffle',
        ]
        rows.append({'name': name, 'objective': objective, 'seed': int(seed),
                     'output': str(output), 'epochs': int(epochs),
                     'steps_per_epoch': int(steps), 'batch': int(batch),
                     'feature_anchor_weight': float(feature_anchor_weight),
                     'command': command, 'status': 'pending'})
    return rows


def _validate_run(row, experiment):
    output = Path(row['output'])
    run_path, report_path = (Path(str(output) + '.run.json'),
                             Path(str(output) + '.report.json'))
    if not run_path.is_file() or not report_path.is_file():
        raise RuntimeError(f"missing run/report artifacts for {row['name']}")
    run, report = json.loads(run_path.read_text()), json.loads(report_path.read_text())
    required_run = {
        'status': 'complete', 'source_tree_sha256': experiment['source_tree_sha256'],
        'corpus_manifest_sha256': experiment['corpus_manifest_sha256'],
        'train_start': train.TOURNAMENT_TRAIN_START,
        'val_start': train.TOURNAMENT_VAL_START,
        'holdout_start': train.TOURNAMENT_OOS_START, 'lineage': 'vanilla',
    }
    for key, expected in required_run.items():
        if run.get(key) != expected:
            raise RuntimeError(f"{row['name']}: {key}={run.get(key)!r}, expected {expected!r}")
    cfg = run.get('config') or {}
    required_cfg = {
        'contrastive_objective': row['objective'], 'seq': 256,
        'preprocessing': 'per_window_per_channel_zscore_v1',
        'contrastive_reserve_contexts': 2.0, 'epochs': row['epochs'],
        'steps_per_epoch': row['steps_per_epoch'], 'batch': row['batch'],
        'probe_folds': 5, 'feature_anchor_weight': row['feature_anchor_weight'],
    }
    for key, expected in required_cfg.items():
        if cfg.get(key) != expected:
            raise RuntimeError(
                f"{row['name']}: config {key}={cfg.get(key)!r}, expected {expected!r}")
    for label, result in (('real', report.get('probe') or {}),
                          ('shuffle', (report.get('control_probe') or {}).get('shuffle') or {})):
        if result.get('probe_protocol') != 'expanding_walk_forward':
            raise RuntimeError(f"{row['name']}: {label} probe is not expanding walk-forward")
        for target in TARGETS:
            folds = ((result.get('per_target') or {}).get(target) or {}).get('fold_delta') or []
            if len(folds) != 5:
                raise RuntimeError(f"{row['name']}: {label}/{target} has {len(folds)} folds")
    return run, report


def analyze(rows, experiment):
    summaries = []
    for row in rows:
        run, report = _validate_run(row, experiment)
        probe = report['probe']
        control = report['control_probe']['shuffle']
        real = {target: float(probe['per_target'][target]['delta']) for target in TARGETS}
        shuffled = {target: float(control['per_target'][target]['delta']) for target in TARGETS}
        strict = report.get('verdict') or {}
        history = (report.get('history') or [{}])[0]
        gate, gate_detail = finetune_ssl._passes(
            probe, float(history.get('std', 0.0)), pretext='contrastive')
        beats_shuffle = {
            target: real[target] > shuffled[target] for target in TARGETS
        }
        eligible = bool(gate and
                        float(probe['mean_core_delta']) > float(control['mean_core_delta']) and
                        all(beats_shuffle[target] for target in PRIMARY))
        summaries.append({
            'name': row['name'], 'objective': row['objective'], 'seed': row['seed'],
            'recorded_gate_passed': bool(history.get('gate_ok')),
            'current_gate_passed': bool(gate), 'current_gate': gate_detail,
            'verdict': strict, 'real_delta': real,
            'shuffle_delta': shuffled, 'beats_shuffle': beats_shuffle,
            'mean_core_delta': float(probe['mean_core_delta']),
            'shuffle_mean_core_delta': float(control['mean_core_delta']),
            'eligible_for_downstream_scoring': eligible,
            'checkpoint': row['output'], 'checkpoint_sha256': run['checkpoint_sha256'],
        })
    eligible = [row['objective'] for row in summaries if row['eligible_for_downstream_scoring']]
    return {
        'schema_version': 'ffm_mantis_v2_ssl_pilot_comparison_v1',
        'experiment': experiment, 'runs': summaries,
        'eligible_for_downstream_scoring': eligible,
        'decision': ('score_eligible_checkpoints_on_forward_path_and_trading_rulers'
                     if eligible else 'stop_stage2_pilot_no_arm_passed'),
        'note': ('This bounded one-seed screen may reject an arm, but cannot promote a checkpoint. '
                 'Promotion requires downstream lift and a second seed.'),
    }


def run_matrix(rows, manifest_path, experiment):
    manifest = {'schema_version': 'ffm_mantis_v2_ssl_pilot_matrix_v1',
                'purpose': 'bounded_development_screen_not_oos_not_full_training',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'experiment': experiment, 'runs': rows}
    _atomic_json(manifest_path, manifest)
    for row in rows:
        output = Path(row['output'])
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f'refusing to overwrite comparison cell: {output}')
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
            raise RuntimeError(f"pilot cell failed: {row['name']} (see {log})")
        _validate_run(row, experiment)
    comparison = analyze(rows, experiment)
    _atomic_json(Path(manifest_path).with_name('comparison.json'), comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='output/foundation_tournament/data_cache')
    parser.add_argument('--output-dir', default='output/mantis_v2_ssl_pilot')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--steps-per-epoch', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--feature-anchor-weight', type=float, default=.1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.steps_per_epoch, args.batch) <= 0:
        parser.error('training budget values must be positive')
    data_dir = Path(args.data_dir).resolve()
    corpus = data_dir / 'TOURNAMENT_CACHE.json'
    if not corpus.is_file():
        raise FileNotFoundError(f'sealed tournament corpus missing: {corpus}')
    _, files = train._experiment_source_files()
    experiment = {
        'source_tree_sha256': train._source_tree_sha256(files),
        'corpus_manifest_sha256': train._sha256(corpus),
        'protocol': 'foundation_5y1y1y_v1',
        'train_start': train.TOURNAMENT_TRAIN_START,
        'val_start': train.TOURNAMENT_VAL_START,
        'holdout_start': train.TOURNAMENT_OOS_START,
        'lineage': 'vanilla', 'seq': 256, 'reserve_contexts': 2.0,
        'preprocessing': 'per_window_per_channel_zscore_v1', 'probe_folds': 5,
        'control': 'shuffle', 'seed': int(args.seed),
        'feature_anchor_weight': float(args.feature_anchor_weight),
        'tickers': list(train.ROOTS), 'timeframes': list(train.TFS),
    }
    output_dir = Path(args.output_dir).resolve()
    rows = build_matrix(data_dir=data_dir, output_dir=output_dir, seed=args.seed,
                        epochs=args.epochs, steps=args.steps_per_epoch, batch=args.batch,
                        device=args.device, feature_anchor_weight=args.feature_anchor_weight)
    manifest = output_dir / 'matrix.json'
    if args.execute:
        result = run_matrix(rows, manifest, experiment)
        print(json.dumps(result, indent=2))
    else:
        _atomic_json(manifest, {'schema_version': 'ffm_mantis_v2_ssl_pilot_matrix_v1',
                               'purpose': 'dry_run', 'experiment': experiment, 'runs': rows})
        print(json.dumps({'manifest': str(manifest), 'runs': rows}, indent=2))


if __name__ == '__main__':
    main()
