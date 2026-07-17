import json
from pathlib import Path

from scripts import run_mantis_v2_ssl_pilot as pilot


def test_pilot_matrix_locks_single_factor_protocol(tmp_path):
    rows = pilot.build_matrix(data_dir=tmp_path, output_dir=tmp_path / 'out')
    assert [row['objective'] for row in rows] == list(pilot.OBJECTIVES)
    assert len({row['seed'] for row in rows}) == 1
    for row in rows:
        cmd = row['command']
        assert cmd[cmd.index('--seq') + 1] == '256'
        assert cmd[cmd.index('--protocol') + 1] == 'foundation_5y1y1y_v1'
        assert cmd[cmd.index('--lineage') + 1] == 'vanilla'
        assert cmd[cmd.index('--contrastive-reserve-contexts') + 1] == '2'
        assert cmd[cmd.index('--preprocessing') + 1] == \
            'per_window_per_channel_zscore_v1'
        assert cmd[cmd.index('--probe-folds') + 1] == '5'
        assert cmd[cmd.index('--controls') + 1] == 'shuffle'


def _probe(delta, gate=True):
    per = {target: {'delta': float(delta), 'fold_delta': [float(delta)] * 5}
           for target in pilot.TARGETS}
    return {'probe_protocol': 'expanding_walk_forward', 'per_target': per,
            'mean_core_delta': float(delta), 'descriptive_delta': float(delta),
            'fwd_absmove_delta': float(delta), 'fwd_dir_delta': float(delta),
            'forward_score': float(2 * delta),
            'learns_regime_vol_structure': bool(gate)}


def test_pilot_analysis_requires_strict_and_shuffle_lift(tmp_path):
    experiment = {
        'source_tree_sha256': 'source', 'corpus_manifest_sha256': 'corpus',
        'train_start': '2019-07-01', 'val_start': '2024-07-01',
        'holdout_start': '2025-07-01', 'lineage': 'vanilla',
    }
    rows = pilot.build_matrix(data_dir=tmp_path, output_dir=tmp_path / 'out')
    for row in rows:
        output = Path(row['output'])
        output.parent.mkdir(parents=True, exist_ok=True)
        Path(str(output) + '.run.json').write_text(json.dumps({
            'status': 'complete', 'source_tree_sha256': 'source',
            'corpus_manifest_sha256': 'corpus', 'train_start': '2019-07-01',
            'val_start': '2024-07-01', 'holdout_start': '2025-07-01',
            'lineage': 'vanilla', 'checkpoint_sha256': 'sha',
            'config': {'contrastive_objective': row['objective'], 'seq': 256,
                       'preprocessing': 'per_window_per_channel_zscore_v1',
                       'contrastive_reserve_contexts': 2.0, 'epochs': row['epochs'],
                       'steps_per_epoch': row['steps_per_epoch'], 'batch': row['batch'],
                       'probe_folds': 5,
                       'feature_anchor_weight': row['feature_anchor_weight']},
        }))
        Path(str(output) + '.report.json').write_text(json.dumps({
            'verdict': {'all_pass': True}, 'probe': _probe(.03),
            'control_probe': {'shuffle': _probe(.01)},
            'history': [{'gate_ok': True, 'std': .5}],
        }))
    result = pilot.analyze(rows, experiment)
    assert result['eligible_for_downstream_scoring'] == list(pilot.OBJECTIVES)

    failed = rows[1]
    report_path = Path(failed['output'] + '.report.json')
    report = json.loads(report_path.read_text())
    report['history'][0]['std'] = 0.0
    report_path.write_text(json.dumps(report))
    assert pilot.analyze(rows, experiment)['eligible_for_downstream_scoring'] == [
        'elapsed_time_v2']
