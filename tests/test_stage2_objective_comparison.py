import json
from pathlib import Path

from futures_foundation.finetune import ssl
from scripts import run_stage2_objective_comparison as comparison


def test_comparison_matrix_locks_shared_protocol(tmp_path):
    rows = comparison.build_matrix(data_dir=tmp_path, output_dir=tmp_path / 'out')
    assert len(rows) == 4
    assert {(r['objective'], r['seed']) for r in rows} == {
        (objective, seed) for objective in comparison.OBJECTIVES for seed in (17, 29)}
    for row in rows:
        cmd = row['command']
        assert cmd[cmd.index('--contrastive-reserve-contexts') + 1] == '3'
        assert cmd[cmd.index('--probe-folds') + 1] == '5'
        assert cmd[cmd.index('--controls') + 1] == 'shuffle'
        assert cmd[cmd.index('--lineage') + 1] == 'vanilla'
        assert '--no-probe' not in cmd and '--smoke' not in cmd


def test_common_reserve_holds_anchor_universe_fixed():
    base = ssl._base_cfg(pretext='contrastive', seq=64, contrastive_reserve_contexts=3.0)
    legacy = {**base, 'contrastive_objective': 'bar_offset_v1'}
    elapsed = {**base, 'contrastive_objective': 'elapsed_time_v2'}
    assert ssl.get_pretext('contrastive').reserve(legacy) == 192
    assert ssl.get_pretext('contrastive').reserve(elapsed) == 192


def _probe(delta):
    per = {target: {'delta': float(delta), 'fold_delta': [float(delta)] * 5}
           for target in comparison.TARGETS}
    return {'probe_protocol': 'expanding_walk_forward', 'per_target': per,
            'mean_core_delta': float(delta)}


def test_analysis_requires_paired_seed_control_and_sampling_wins(tmp_path):
    experiment = {
        'source_tree_sha256': 'source', 'corpus_manifest_sha256': 'corpus',
        'val_start': '2024-01-01', 'holdout_start': '2025-07-01', 'lineage': 'vanilla',
    }
    rows = comparison.build_matrix(data_dir=tmp_path, output_dir=tmp_path / 'out')
    for row in rows:
        out = Path(row['output']); out.parent.mkdir(parents=True, exist_ok=True)
        delta = 0.03 if row['objective'] == 'elapsed_time_v2' else 0.01
        run = {
            'status': 'complete', 'source_tree_sha256': 'source',
            'corpus_manifest_sha256': 'corpus', 'val_start': '2024-01-01',
            'holdout_start': '2025-07-01', 'lineage': 'vanilla', 'checkpoint_sha256': 'x',
            'config': {'contrastive_objective': row['objective'], 'seq': 64,
                       'preprocessing': 'per_window_per_channel_zscore_v1',
                       'contrastive_reserve_contexts': 3.0, 'epochs': 5,
                       'steps_per_epoch': 50, 'batch': 32, 'probe_folds': 5},
        }
        diag = {'positive_valid_fraction': 0.9, 'valid_rows_fraction': 0.9,
                'valid_negatives_min': 32, 'positive_overlap_max': 0.49,
                'weight_min': 1.0, 'weight_max': 1.0}
        report = {'probe': _probe(delta), 'control_probe': {'shuffle': _probe(0.0)},
                  'history': [{'gate_ok': True, 'task_diagnostics': diag}]}
        Path(str(out) + '.run.json').write_text(json.dumps(run))
        Path(str(out) + '.report.json').write_text(json.dumps(report))
    result = comparison.analyze(rows, experiment)
    assert result['promote_elapsed_time_v2'] is True

    # One paired-seed primary regression is sufficient to block promotion.
    victim = next(r for r in rows if r['objective'] == 'elapsed_time_v2' and r['seed'] == 17)
    path = Path(victim['output'] + '.report.json')
    report = json.loads(path.read_text())
    report['probe']['per_target']['trend_eff']['delta'] = -0.01
    path.write_text(json.dumps(report))
    assert comparison.analyze(rows, experiment)['promote_elapsed_time_v2'] is False
