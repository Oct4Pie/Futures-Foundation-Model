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


# ---- operating-point / WR-by-score readouts (the '1-2 A+ trades/day' deploy view) -----------
class _RLabeler:
    """Minimal labeler: each key carries its realized R at k[4]; evaluate returns R for taken."""
    def evaluate(self, keys, preds):
        return np.array([k[4] for k, p in zip(keys, preds) if p == 1], float)


def _ranked_oos():
    # 5 wins (+2R) then 5 losses (-1R); proba ranks the wins first; 10 trades across 5 days
    R = [2.0, 2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    keys = [('ES@3min', i, 1, 0.0, r) for i, r in enumerate(R)]
    proba = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.40, 0.30, 0.20, 0.10, 0.05])
    ts = pd.to_datetime([f'2026-01-0{d}' for d in [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]], utc=True)
    return _RLabeler(), keys, proba, ts


def test_operating_points_top_slice_is_all_winners():
    lab, keys, proba, ts = _ranked_oos()
    rows = produce.operating_points(lab, keys, proba, ts, rates=(5, 2, 1))
    by = {r['rate']: r for r in rows}
    assert by[1]['days'] == 5
    assert by[1]['n'] == 5 and by[1]['wr3R'] == 1.0 and by[1]['meanR'] == pytest.approx(2.0)
    assert by[2]['n'] == 10 and by[2]['wr3R'] == pytest.approx(0.5)   # 2/day*5 -> all 10
    # tighter rate = higher (or equal) WR — the whole point of the operating-point view
    assert by[1]['wr3R'] >= by[2]['wr3R']


def test_wr_by_score_bands_are_monotone():
    lab, keys, proba, ts = _ranked_oos()
    bands = produce.wr_by_score(lab, keys, proba, ts)
    assert sum(b['n'] for b in bands) == len(keys)          # bands partition the OOS set
    wrs = [b['wr3R'] for b in bands if b['n'] > 0]
    assert wrs[0] >= wrs[-1]                                 # top band cleaner than bottom
    assert bands[0]['wr3R'] == 1.0                           # top decile = pure winners here
    for b in bands:                                          # trades/day = n / days(=5)
        assert b['per_day'] == pytest.approx(b['n'] / 5)


def test_alignment_breakdown_counter_vs_aligned():
    # counter pivots: model score separates winners (score high -> R=+2) from losers; aligned all
    # mediocre. The readout must show counter's TOP-score WR >> counter's base WR (sighted gating).
    class _AlignLab(_RLabeler):
        def htf_alignment(self, keys):
            return np.array([k[5] for k in keys])            # alignment stashed at k[5]
    n = 120
    keys, proba, ts = [], [], []
    for i in range(n):
        counter = i < 60
        good = (i % 2 == 0)
        r = 2.0 if good else -1.0
        keys.append(('ES@3min', i, 1, 0.0, r, -1 if counter else 1))
        proba.append((0.9 if good else 0.1) if counter else 0.5)   # score separates ONLY counter
        ts.append(np.datetime64('2026-01-0%d' % ((i % 5) + 1)))
    ab = produce.alignment_breakdown(_AlignLab(), keys, np.array(proba), ts)
    assert set(ab) == {'counter', 'aligned'}
    assert ab['counter']['base_wr3R'] == pytest.approx(0.5)
    top = ab['counter']['ops'][-1]                           # tightest rate = highest scores
    assert top['wr3R'] == 1.0                                # sighted: top counter picks all win
    assert ab['aligned']['base_wr3R'] == pytest.approx(0.5)
    # no htf_alignment on the labeler -> None (backward-compatible)
    assert produce.alignment_breakdown(_RLabeler(), keys, np.array(proba), ts) is None


def test_operating_points_empty_is_safe():
    assert produce.operating_points(_RLabeler(), [], np.array([]), []) == []
    assert produce.wr_by_score(_RLabeler(), [], np.array([]), []) == []


def test_ladder_signal_contract_shape(tmp_path):
    # the reach-ladder produce must write a ladder-shaped signal.json: entry_signal=p_3r, both head
    # outputs, reach_targets, and calibration baked into the onnx (bot reads p_3r as-is).
    import json
    out = dict(oos_auc=0.6, oos_meanR=0.63, shuffle_meanR=None, edge_shuffle=None,
               oos_trades=1000, n_train=100, n_oos=50, platt=None,
               entry_thresholds={'top0.1pct': 4.2, 'top1pct': 3.3, 'top10pct': 2.8},
               val_percentiles={'ES@3min': {'p50': 1.7, 'p90': 3.1, 'p99': 4.4}})
    ck = {'rank': 'expected_reach', 'reach_targets': [2.0, 3.0, 4.0, 6.0, 8.0], 'head': 'mlp'}
    lab = SyntheticLabeler(n_bars=200)
    produce._emit(out, 'mantis_frozen', ck, lab, None, None, 5, 1, ['c0'], ['ES'], ['3min'],
                  '2026-01-01', True, str(tmp_path / 'model'), verbose=False)
    c = json.loads((tmp_path / 'model_signal.json').read_text())
    assert c['head_type'] == 'reach_ladder'
    assert c['entry_signal'] == 'expected_reach'           # the VALIDATED entry signal (80% WR tiers)
    assert c['entry_rule'] == 'enter if expected_reach >= entry_thresholds[tier]'
    assert c['entry_thresholds'] == {'top0.1pct': 4.2, 'top1pct': 3.3, 'top10pct': 2.8}
    assert c['head_outputs'] == ['p_3r', 'expected_reach']
    assert c['reach_targets'] == [2.0, 3.0, 4.0, 6.0, 8.0]
    assert c['calibration']['baked_into_onnx'] is True
    # STANDARDIZED 0-100 SCORE (2026-07-17): the ladder MUST carry the per-stream percentile scale
    # too — same 0-100 axis as the single head, ranking expected_reach, so a head swap needs no bot
    # change. Regression guard for the wiring gap where only the single head emitted it.
    ss = c['score_scale']
    assert ss['kind'] == 'per_stream_val_percentile' and ss['signal'] == 'expected_reach'
    assert ss['percentiles'] == {'ES@3min': {'p50': 1.7, 'p90': 3.1, 'p99': 4.4}}
    assert ss['p_min'] == 1.7                               # VAL-median E[R] backstop
    assert 'expected_reach' in ss['rule']


def test_contract_window_recipe_for_multi_tf(tmp_path):
    # a labeler that declares MV_AGG=(1,5) gets an explicit window_recipe in signal.json (how the
    # bot builds the [10, seq] input); single-TF labelers get None (backward-compatible).
    import json
    out = dict(oos_auc=0.6, oos_meanR=0.5, shuffle_meanR=None, edge_shuffle=None,
               oos_trades=10, n_train=100, n_oos=50, platt=None)
    lab = SyntheticLabeler(n_bars=200)
    lab.MV_AGG, lab.MV_SEQ, lab.MV_MODE = (1, 5), 64, 'ohlcv_agg1-5'
    produce._emit(out, 'mantis_frozen', {'head': 'mlp'}, lab, None, None, 10, 64,
                  None, ['ES'], ['3min'], '2026-01-01', True, str(tmp_path / 'm'), verbose=False)
    c = json.loads((tmp_path / 'm_signal.json').read_text())
    r = c['window_recipe']
    assert r['factors'] == [1, 5] and r['mv_seq'] == 64
    assert 'O=first' in r['aggregation'] and 'anchored' in r['aggregation']
    # single-TF -> no recipe (bot keeps feeding the plain [5, seq] window)
    lab2 = SyntheticLabeler(n_bars=200)
    produce._emit(dict(out), 'mantis_frozen', {'head': 'mlp'}, lab2, None, None, 5, 64,
                  None, ['ES'], ['3min'], '2026-01-01', True, str(tmp_path / 'm2'), verbose=False)
    assert json.loads((tmp_path / 'm2_signal.json').read_text())['window_recipe'] is None


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
