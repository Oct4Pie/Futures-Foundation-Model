"""Signal-contract sidecar producer + BaseChronosStrategy base.

Torch-free: produce/strategy import only the torch-free backbone seam.
Verifies the <base>_signal.json contract is assembled correctly from a
bundle + an inheriting labeler, the feature-width guard fires on mismatch,
and a hook-less labeler still yields a valid minimal contract.
"""
import json
from pathlib import Path

from futures_foundation.pipeline.strategy import BaseChronosStrategy
from futures_foundation.pipeline.produce import write_signal_contract


class _Labeler(BaseChronosStrategy):
    n_classes = 2
    FLIP_SCHEME = 'fake'
    FLIP_PARAMS = {'a': 1, 'b': 2}
    DIRECTION_RULE = 'rule'
    MIN_GAP = 20
    HANDCRAFT_NAMES = ('x', 'y')
    STOP_ATR, RR, VERT = 0.5, 3.0, 120
    VERSION = 'fake-1.0'
    TRAIN_SCOPE = {'tickers': ['NQ']}

    def __init__(self, feat_cols=('f0', 'f1', 'f2', 'f3', 'f4', 'f5')):
        self._b = {'NQ': {'feat_cols': list(feat_cols)}}


def _bundle(feat_dim):
    return {'feat_dim': feat_dim, 'embed_dim': 256, 'ctx_window': 128,
            'chronos_ckpt': 'amazon/chronos-bolt-tiny', 'n_classes': 2,
            'calibrated': False, 'pool': 'mean', 'locscale': False,
            'training_metadata': {'train_date': '2026-06-23'}}


def test_contract_assembled_and_written(tmp_path):
    lab = _Labeler()
    path, c, ok = write_signal_contract(lab, _bundle(8 + 256),
                                        str(tmp_path / 'm.joblib'))
    assert ok
    assert c['contract_version'] == '1.0'
    assert c['triplet_id'] == 'm@2026-06-23'
    assert c['flip_scheme'] == 'fake' and c['flip_params'] == {'a': 1, 'b': 2}
    assert c['direction_rule'] == 'rule' and c['entry_timing'] == 'next_bar_open'
    assert c['handcraft_features'] == ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'x', 'y']
    assert c['handcraft_dim'] == 8
    assert c['label_def'] == 'triple_barrier SL=0.5ATR TP=3.0R VERT=120'
    assert c['ctx_window'] == 128
    assert c['chronos_ckpt'] == 'amazon/chronos-bolt-tiny'
    assert c['content_sha'] is None            # no ONNX next to it
    # actually written + valid JSON
    on_disk = json.loads(Path(path).read_text())
    assert on_disk['flip_scheme'] == 'fake'


def test_feature_width_mismatch_flagged(tmp_path):
    # bundle says handcraft width 9, labeler declares 8 names -> ok False
    lab = _Labeler()
    _, c, ok = write_signal_contract(lab, _bundle(9 + 256),
                                     str(tmp_path / 'm.joblib'))
    assert ok is False
    assert c['handcraft_dim'] == 8             # reports the labeler's actual names


def test_hookless_labeler_minimal_contract(tmp_path):
    class Bare:                                  # no signal_contract/feature_names
        pass
    _, c, ok = write_signal_contract(Bare(), _bundle(8 + 256),
                                     str(tmp_path / 'm.joblib'))
    assert ok                                    # names is None -> no width check
    assert c['flip_scheme'] is None              # minimal but valid
    assert c['handcraft_features'] is None
    assert c['handcraft_dim'] == 8               # falls back to bundle width
    assert c['contract_version'] == '1.0'


def test_calibrated_flag_passthrough(tmp_path):
    b = _bundle(8 + 256); b['calibrated'] = True
    _, c, _ = write_signal_contract(_Labeler(), b, str(tmp_path / 'm.joblib'))
    assert c['calibrated'] is True


def test_contract_no_return_shape_by_default(tmp_path):
    _, c, _ = write_signal_contract(_Labeler(), _bundle(8 + 256),
                                    str(tmp_path / 'm.joblib'))
    assert c['return_shape'] is False
    assert c['return_shape_features'] is None and c['return_shape_fn'] is None
    assert c['embed_layout'] == [['chronos_pool', 256]]


def test_contract_declares_return_shape_for_serve_parity(tmp_path):
    from futures_foundation.extractors.chronos.window_features import (
        return_shape_feature_names, RETURN_SHAPE_DIM)
    b = _bundle(8 + 256 + RETURN_SHAPE_DIM)            # embed now 263, +8 handcraft
    b['embed_dim'] = 256 + RETURN_SHAPE_DIM
    b['return_shape'] = True
    _, c, ok = write_signal_contract(_Labeler(), b, str(tmp_path / 'm.joblib'))
    assert ok and c['return_shape'] is True
    assert c['return_shape_features'] == return_shape_feature_names()
    assert c['return_shape_fn'].endswith('window_features.return_shape_features')
    assert c['embed_layout'][0] == ['chronos_pool', 256]
    assert ['return_shape', RETURN_SHAPE_DIM] in c['embed_layout']
    assert c['handcraft_dim'] == 8                     # 271 - 263, consumer-checkable
