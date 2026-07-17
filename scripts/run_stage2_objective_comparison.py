#!/usr/bin/env python3
"""Run the locked Stage-2 legacy-vs-elapsed development comparison.

This is a development screen, not full training and not an OOS benchmark. Every cell uses the
same pre-2024 train period, 2024-through-June-2025 validation period, source/corpus snapshot,
eligible anchor universe, preprocessing, context, optimizer budget, strict probe, and shuffle
control. July 2025 onward is physically excluded by ``train_ssl_local.py``.
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


TRAIN = ROOT / 'scripts' / 'train_ssl_local.py'
OBJECTIVES = ('bar_offset_v1', 'elapsed_time_v2')
TARGETS = ('vol', 'trend_eff', 'range_expand', 'fwd_absmove', 'direction', 'fwd_dir')
PRIMARY = ('trend_eff', 'range_expand')


def _atomic_json(path, value):
    path = Path(path)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=float) + '\n')
    os.replace(tmp, path)


def build_matrix(*, data_dir, output_dir, seeds=(17, 29), epochs=5, steps=50, batch=32,
                 device='cuda'):
    rows = []
    for objective in OBJECTIVES:
        for seed in seeds:
            name = f'stage2_{objective}_seed{int(seed)}_dev'
            output = Path(output_dir).resolve() / f'{name}.pt'
            command = [
                sys.executable, str(TRAIN), '--stage', 'contrastive', '--lineage', 'vanilla',
                '--data-dir', str(Path(data_dir).resolve()), '--output', str(output),
                '--model-version', 'v2', '--seq', '64',
                '--preprocessing', 'per_window_per_channel_zscore_v1',
                '--contrastive-objective', objective,
                '--contrastive-reserve-contexts', '3', '--device', device,
                '--seed', str(int(seed)), '--epochs', str(int(epochs)),
                '--steps-per-epoch', str(int(steps)), '--batch', str(int(batch)),
                '--patience', str(int(epochs)), '--probe-folds', '5', '--controls', 'shuffle',
            ]
            rows.append({
                'name': name, 'objective': objective, 'seed': int(seed), 'output': str(output),
                'epochs': int(epochs), 'steps_per_epoch': int(steps), 'batch': int(batch),
                'command': command, 'status': 'pending',
            })
    return rows


def _validate_run(row, experiment):
    output = Path(row['output'])
    run_path = Path(str(output) + '.run.json')
    report_path = Path(str(output) + '.report.json')
    if not run_path.is_file() or not report_path.is_file():
        raise RuntimeError(f"missing run/report artifacts for {row['name']}")
    run = json.loads(run_path.read_text())
    report = json.loads(report_path.read_text())
    expected = {
        'status': 'complete', 'source_tree_sha256': experiment['source_tree_sha256'],
        'corpus_manifest_sha256': experiment['corpus_manifest_sha256'],
        'val_start': train.CANONICAL_VAL_START, 'holdout_start': train.CANONICAL_OOS_START,
        'lineage': 'vanilla',
    }
    for key, value in expected.items():
        if run.get(key) != value:
            raise RuntimeError(f"{row['name']}: {key}={run.get(key)!r}, expected {value!r}")
    cfg = run.get('config') or {}
    required_cfg = {
        'contrastive_objective': row['objective'], 'seq': 64,
        'preprocessing': 'per_window_per_channel_zscore_v1',
        'contrastive_reserve_contexts': 3.0, 'epochs': row['epochs'],
        'steps_per_epoch': row['steps_per_epoch'], 'batch': row['batch'], 'probe_folds': 5,
    }
    for key, value in required_cfg.items():
        if cfg.get(key) != value:
            raise RuntimeError(f"{row['name']}: config {key}={cfg.get(key)!r}, expected {value!r}")
    probe = report.get('probe') or {}
    control = (report.get('control_probe') or {}).get('shuffle') or {}
    if probe.get('probe_protocol') != 'expanding_walk_forward':
        raise RuntimeError(f"{row['name']}: real probe is not expanding walk-forward")
    if control.get('probe_protocol') != 'expanding_walk_forward':
        raise RuntimeError(f"{row['name']}: shuffle probe is not expanding walk-forward")
    for label, result in (('real', probe), ('shuffle', control)):
        per = result.get('per_target') or {}
        for target in TARGETS:
            folds = (per.get(target) or {}).get('fold_delta') or []
            if len(folds) != 5:
                raise RuntimeError(
                    f"{row['name']}: {label}/{target} has {len(folds)} folds, expected 5")
    return run, report


def analyze(rows, experiment):
    summaries = []
    by_key = {}
    for row in rows:
        run, report = _validate_run(row, experiment)
        probe = report['probe']; control = report['control_probe']['shuffle']
        real_delta = {t: float(probe['per_target'][t]['delta']) for t in TARGETS}
        ctrl_delta = {t: float(control['per_target'][t]['delta']) for t in TARGETS}
        diag = ((report.get('history') or [{}])[0].get('task_diagnostics') or {})
        control_wins = {t: real_delta[t] > ctrl_delta[t] for t in PRIMARY}
        sampling_ok = (row['objective'] != 'elapsed_time_v2' or
                       (float(diag.get('positive_valid_fraction', 0)) >= 0.70 and
                        float(diag.get('valid_rows_fraction', 0)) >= 0.80 and
                        float(diag.get('valid_negatives_min', 0)) >= 16 and
                        float(diag.get('positive_overlap_max', 1)) <= 0.500001 and
                        float(diag.get('weight_min', 0)) == 1.0 and
                        float(diag.get('weight_max', 0)) == 1.0))
        summary = {
            'name': row['name'], 'objective': row['objective'], 'seed': row['seed'],
            'gate_passed': bool((report.get('history') or [{}])[0].get('gate_ok')),
            'real_delta': real_delta, 'shuffle_delta': ctrl_delta,
            'mean_core_delta': float(probe['mean_core_delta']),
            'shuffle_mean_core_delta': float(control['mean_core_delta']),
            'beats_shuffle_primary': control_wins,
            'beats_shuffle_mean_core': float(probe['mean_core_delta']) > float(control['mean_core_delta']),
            'sampling_ok': bool(sampling_ok), 'sampling_diagnostics': diag,
            'checkpoint_sha256': run['checkpoint_sha256'],
        }
        summaries.append(summary); by_key[(row['objective'], row['seed'])] = summary
    paired = []
    for seed in sorted({r['seed'] for r in rows}):
        legacy, elapsed = by_key[('bar_offset_v1', seed)], by_key[('elapsed_time_v2', seed)]
        primary = {t: elapsed['real_delta'][t] > legacy['real_delta'][t] for t in PRIMARY}
        paired.append({'seed': seed, 'elapsed_beats_legacy_primary': primary,
                       'elapsed_beats_legacy_mean_core':
                           elapsed['mean_core_delta'] > legacy['mean_core_delta']})
    elapsed_runs = [s for s in summaries if s['objective'] == 'elapsed_time_v2']
    promote = bool(all(
        s['gate_passed'] and s['sampling_ok'] and s['beats_shuffle_mean_core'] and
        all(s['beats_shuffle_primary'].values()) for s in elapsed_runs
    ) and all(p['elapsed_beats_legacy_mean_core'] and
              all(p['elapsed_beats_legacy_primary'].values()) for p in paired))
    return {'schema_version': 'ffm_stage2_objective_comparison_v1',
            'experiment': experiment, 'runs': summaries, 'paired': paired,
            'promotion_rule': ('both V2 seeds pass the strict gate and sampling checks, beat '
                               'shuffle on mean/core primary targets, and beat paired legacy on '
                               'mean/core primary targets'),
            'promote_elapsed_time_v2': promote}


def run_matrix(rows, manifest_path, experiment):
    manifest = {'schema_version': 'ffm_stage2_objective_matrix_v1',
                'purpose': 'bounded_development_screen_not_oos_not_full_training',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'experiment': experiment, 'runs': rows}
    _atomic_json(manifest_path, manifest)
    for row in rows:
        output = Path(row['output']); output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing comparison cell: {output}")
        log = Path(str(output) + '.log')
        row.update(status='running', started_utc=datetime.now(timezone.utc).isoformat(), log=str(log))
        _atomic_json(manifest_path, manifest)
        with log.open('w') as stream:
            result = subprocess.run(row['command'], cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT)
        row.update(status=('complete' if result.returncode == 0 else 'failed'),
                   returncode=int(result.returncode),
                   finished_utc=datetime.now(timezone.utc).isoformat())
        _atomic_json(manifest_path, manifest)
        if result.returncode:
            raise RuntimeError(f"comparison cell failed: {row['name']} (see {log})")
        _validate_run(row, experiment)
    comparison = analyze(rows, experiment)
    comparison_path = Path(manifest_path).with_name('comparison.json')
    _atomic_json(comparison_path, comparison)
    return comparison


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', default='output/stage2_objective_comparison')
    p.add_argument('--seeds', default='17,29')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--steps-per-epoch', type=int, default=50)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--device', default='cuda')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    seeds = tuple(int(x) for x in a.seeds.split(',') if x)
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError('comparison requires exactly two distinct seeds')
    if min(a.epochs, a.steps_per_epoch, a.batch) <= 0:
        raise ValueError('training budget values must be positive')
    data_dir = Path(a.data_dir).resolve()
    corpus = data_dir / 'MANIFEST.json'
    if not corpus.is_file():
        raise FileNotFoundError(f'sealed corpus manifest missing: {corpus}')
    _, files = train._experiment_source_files()
    experiment = {
        'source_tree_sha256': train._source_tree_sha256(files),
        'corpus_manifest_sha256': train._sha256(corpus),
        'val_start': train.CANONICAL_VAL_START, 'holdout_start': train.CANONICAL_OOS_START,
        'lineage': 'vanilla', 'seq': 64, 'reserve_contexts': 3.0,
        'preprocessing': 'per_window_per_channel_zscore_v1', 'probe_folds': 5,
        'control': 'shuffle', 'tickers': list(train.ROOTS), 'timeframes': list(train.TFS),
    }
    output_dir = Path(a.output_dir).resolve()
    rows = build_matrix(data_dir=data_dir, output_dir=output_dir, seeds=seeds,
                        epochs=a.epochs, steps=a.steps_per_epoch, batch=a.batch,
                        device=a.device)
    manifest = output_dir / 'matrix.json'
    if a.execute:
        comparison = run_matrix(rows, manifest, experiment)
        print(json.dumps({'comparison': str(output_dir / 'comparison.json'),
                          'promote_elapsed_time_v2': comparison['promote_elapsed_time_v2']},
                         indent=2))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, {'schema_version': 'ffm_stage2_objective_matrix_v1',
                                'purpose': 'plan_only', 'experiment': experiment, 'runs': rows})
        print(f'wrote {len(rows)}-cell comparison plan -> {manifest}')


if __name__ == '__main__':
    main()
