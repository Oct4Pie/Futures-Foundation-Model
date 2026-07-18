from pathlib import Path

import pytest

from scripts import run_ssl_ablation_matrix as matrix
from scripts import train_ssl_local


def test_matrix_is_bounded_direct_vanilla_smoke(tmp_path):
    admission = tmp_path / 'admission.json'
    rows = matrix.build_matrix(
        data_dir=tmp_path, output_dir=tmp_path / 'out', admission_report=admission,
    )
    assert len(rows) == 6
    assert {(r['seq'], r['preprocessing']) for r in rows} == {
        (seq, prep) for seq in (64, 128, 256) for prep in matrix.PREPROCESSING.values()}
    for row in rows:
        command = row['command']
        assert command[command.index('--lineage') + 1] == 'vanilla'
        assert command[command.index('--contrastive-objective') + 1] == 'elapsed_time_v2'
        assert row['contrastive_objective'] == 'elapsed_time_v2'
        assert '--smoke' in command and '--no-probe' in command
        assert command[command.index('--admission-report') + 1] == str(admission.resolve())
        assert row['status'] == 'pending' and Path(row['output']).parent == tmp_path / 'out'


def test_lineage_resolution_is_explicit():
    assert train_ssl_local._resolve_lineage('mask', 'auto', None) == 'vanilla'
    assert train_ssl_local._resolve_lineage('contrastive', 'vanilla', None) == 'vanilla'
    with pytest.raises(ValueError, match='requires --warm-checkpoint'):
        train_ssl_local._resolve_lineage('contrastive', 'canonical', None)
    with pytest.raises(ValueError, match='cannot also specify'):
        train_ssl_local._resolve_lineage('contrastive', 'vanilla', Path('/x.pt'))
    with pytest.raises(ValueError, match='no predecessor'):
        train_ssl_local._resolve_lineage('mask', 'canonical', None)
    assert train_ssl_local._resolve_lineage(
        'contrastive', 'diagnostic', Path('/x.pt')) == 'diagnostic'
    with pytest.raises(ValueError, match='requires a staged predecessor'):
        train_ssl_local._resolve_lineage('contrastive', 'diagnostic', None)
    with pytest.raises(ValueError, match='no predecessor'):
        train_ssl_local._resolve_lineage('mask', 'diagnostic', Path('/x.pt'))
