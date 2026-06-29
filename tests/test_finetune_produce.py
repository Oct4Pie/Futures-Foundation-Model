"""Final-model train + held-out OOS (finetune/produce.py) — torch-free.

Uses a synthetic labeler + the 'logistic' classifier so the produce step (train <
cutoff, score the holdout once, SHUFFLE control) is exercised end-to-end through
the subprocess worker, no torch.
"""
import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune import produce


class SyntheticLabeler:
    n_classes = 2

    def __init__(self, n_bars=1600, seq=32, C=4, seed=0):
        rng = np.random.default_rng(seed)
        self.ts = pd.date_range('2020-01-01', periods=n_bars, freq='1D', tz='UTC')
        self.y = rng.integers(0, 2, n_bars)
        self.W = rng.standard_normal((n_bars, C, seq)).astype(np.float32)
        self.W[self.y == 1, 0, -6:] += 2.5
        self.seq, self.C = seq, C

    def calendar(self):
        return pd.DataFrame({'item_id': 'SYN', 'timestamp': self.ts, 'target': self.y})

    def build(self, lo, hi, test_start):
        idx = np.flatnonzero((self.ts >= lo) & (self.ts < hi))
        return [self.W[i, 0] for i in idx], self.y[idx], [(int(i),) for i in idx]

    def mv_contexts(self, keys):
        return (np.stack([self.W[k[0]] for k in keys]) if keys
                else np.zeros((0, self.C, self.seq)))

    def mv_feature_names(self):
        return [f'ch{i}' for i in range(self.C)]

    def evaluate(self, keys, preds):
        return np.array([(2.0 if self.y[k[0]] == 1 else -1.0)
                         for k, p in zip(keys, preds) if p == 1])


def test_train_final_oos_beats_shuffle():
    lab = SyntheticLabeler(n_bars=1600, seed=0)
    out = produce.train_final(lab, classifier='logistic', holdout_start='2023-06-01',
                              seed=0, verbose=False)
    for k in ('oos_auc', 'best_val_auc', 'oos_meanR', 'shuffle_meanR', 'edge_shuffle',
              'n_train', 'n_oos', 'oos_trades', 'beats_shuffle'):
        assert k in out
    assert out['n_oos'] > 0 and out['oos_trades'] > 0
    assert out['oos_auc'] > 0.8                     # learnable signal holds OOS
    assert out['beats_shuffle'] is True
    assert out['edge_shuffle'] >= produce.PASS_LIFT_MARGIN_R


def test_train_final_insufficient_oos_raises():
    lab = SyntheticLabeler(n_bars=1600, seed=0)
    with pytest.raises(ValueError):
        produce.train_final(lab, classifier='logistic', holdout_start='2099-01-01',
                            verbose=False)


def test_train_final_writes_signal_contract(tmp_path):
    import json
    lab = SyntheticLabeler(n_bars=1600, seed=0)
    out = produce.train_final(lab, classifier='logistic', holdout_start='2023-06-01',
                              seed=0, export_onnx=True,
                              output_path=str(tmp_path / 'mantis_fractal'), verbose=False)
    assert 'artifacts' in out
    cpath = out['artifacts']['contract']
    c = json.loads(open(cpath).read())
    # self-describing contract shape (mirrors pipeline.produce's signal.json)
    for k in ('contract_version', 'role', 'input', 'channel_names', 'ft_config',
              'n_classes', 'oos_metrics', 'train_scope', 'content_sha'):
        assert k in c
    assert c['input']['channels'] == lab.C
    assert c['channel_names'] == [f'ch{i}' for i in range(lab.C)]
    assert c['oos_metrics']['oos_auc'] is not None
    # logistic has no ONNX export -> onnx/sha are null but the contract still writes
    assert c['onnx'] is None and c['content_sha'] is None
