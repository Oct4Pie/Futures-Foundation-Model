"""RL pipeline tests (generic — no proprietary strategy logic)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.rl import RLStrategy, register, get_strategy, RL_STRATEGIES
from pipelines.rl.base import RL_STRATEGIES as _REG


# ── a synthetic generic strategy (stands in for any plug-in; NOT CRT) ────────
class _SyntheticStrategy(RLStrategy):
    name = "synthetic"
    entry_filter = True

    def detect_entries(self, df_raw, ctx_df, ticker):
        n = len(df_raw)
        idx = list(range(10, n - 5, 7))
        return pd.DataFrame({
            "bar_idx": idx,
            "direction": 1,
            "sl_distance": 1.0,
            "tp_rr": 2.0,
        })


def _raw(n=120):
    base = 100.0 + np.arange(n) * 0.5
    idx = pd.date_range("2023-01-01", periods=n, freq="3min",
                        tz="America/New_York")
    return pd.DataFrame(
        {"open": base, "high": base + 1, "low": base - 1,
         "close": base, "volume": 100.0}, index=idx)


def test_abc_cannot_instantiate_without_detect_entries():
    with pytest.raises(TypeError):
        RLStrategy()


def test_synthetic_strategy_emits_valid_events():
    s = _SyntheticStrategy()
    ev = s.detect_entries(_raw(120), _raw(120), "ES")
    assert {"bar_idx", "direction", "sl_distance", "tp_rr"} <= set(ev.columns)
    assert (ev["tp_rr"] >= 1.0).all() and (ev["sl_distance"] > 0).all()
    assert len(ev) > 0


def test_default_knobs_and_entry_filter_toggle():
    s = _SyntheticStrategy()
    assert s.entry_filter is True            # default: PPO learns chop-veto
    assert s.trail_atr_k == 2.0 and s.activate_r == 1.0 and s.max_hold == 130
    assert s.config_dict() == {}

    class _NoFilter(_SyntheticStrategy):
        name = "nofilter"
        entry_filter = False                 # SuperTrend-style: pure exit-RL
    assert _NoFilter().entry_filter is False


def test_registry_register_get_and_dup_guard():
    @register("unit_test_strat")
    class _S(_SyntheticStrategy):
        name = "unit_test_strat"
    try:
        got = get_strategy("unit_test_strat")
        assert isinstance(got, RLStrategy) and got.name == "unit_test_strat"
        with pytest.raises(ValueError):           # dup name rejected
            @register("unit_test_strat")
            class _S2(_SyntheticStrategy):
                name = "unit_test_strat"
        with pytest.raises(KeyError):             # unknown name
            get_strategy("does_not_exist")
    finally:
        _REG.pop("unit_test_strat", None)


def test_get_strategy_type_checks():
    RL_STRATEGIES["bad"] = lambda **_: object()
    try:
        with pytest.raises(TypeError):
            get_strategy("bad")
    finally:
        RL_STRATEGIES.pop("bad", None)


# ── device helper ────────────────────────────────────────────────────────────
from pipelines.rl.device import get_device, device_str
import torch


def test_device_auto_and_explicit():
    d = get_device("auto")
    assert isinstance(d, torch.device)
    assert device_str("auto") in ("cuda", "mps", "cpu")
    assert get_device("cpu").type == "cpu"


# ── SingleTradeEnv ───────────────────────────────────────────────────────────
from pipelines.rl.env import SingleTradeEnv


def _arrs(n=40, trend=1.0, base=100.0):
    px = base + np.arange(n) * trend
    ctx = np.tile(np.array([[0.1, 0.2, 0.3]], np.float32), (n, 1))
    return ctx, px.copy(), (px + 1).copy(), (px - 1).copy(), px.copy()


def test_env_untradable_signal_at_end():
    ctx, o, h, l, c = _arrs(10)
    e = SingleTradeEnv(ctx, o, h, l, c, entry_bar=9, direction=1,
                       sl_distance=1.0)
    e.reset()
    obs, r, term, _, info = e.step(1)
    assert term and r == 0.0 and info.get("untradable")


def test_env_obs_dim_and_veto_is_negative():
    ctx, o, h, l, c = _arrs(40, trend=1.0)
    e = SingleTradeEnv(ctx, o, h, l, c, entry_bar=5, direction=1,
                       sl_distance=1.0, entry_filter=True, veto_cost=0.02)
    obs = e.reset()
    assert obs.shape == (e.ctx_dim + 4,) == (7,)
    obs, r, term, _, info = e.step(0)            # veto
    assert term and info.get("veto") and r == pytest.approx(-0.02)


def test_env_take_then_exit_uptrend_positive_R():
    ctx, o, h, l, c = _arrs(40, trend=2.0)        # strong uptrend
    e = SingleTradeEnv(ctx, o, h, l, c, entry_bar=5, direction=1,
                       sl_distance=1.0, entry_filter=True, max_hold=20)
    e.reset()
    e.step(1)                                     # take → enter at bar 6 open
    r = None
    for _ in range(5):
        obs, r, term, _, info = e.step(0)         # hold
        if term:
            break
    obs, r, term, _, info = e.step(1)             # exit
    assert term and r > 0                         # uptrend long profit in R


def test_env_hard_sl_stops_at_minus_1R():
    ctx, o, h, l, c = _arrs(40, trend=1.0)
    c2 = c.copy(); l2 = l.copy()
    l2[7] = 90.0                                  # crash below stop after entry@6
    e = SingleTradeEnv(ctx, o, h, l2, c2, entry_bar=5, direction=1,
                       sl_distance=1.0, entry_filter=False)  # pure-exit start
    e.reset()
    obs, r, term, _, info = e.step(0)             # hold into the crash bar
    assert term and info.get("sl") and r == pytest.approx(-1.0, abs=1e-6)


def test_env_pure_exit_starts_in_trade():
    ctx, o, h, l, c = _arrs(40, trend=2.0)
    e = SingleTradeEnv(ctx, o, h, l, c, entry_bar=5, direction=1,
                       sl_distance=1.0, entry_filter=False, max_hold=3)
    e.reset()
    assert e.state == 1                           # IN_TRADE immediately
    term = False
    while not term:
        obs, r, term, _, info = e.step(0)         # hold → timeout close
    assert info.get("timeout") or info.get("sl")


# ── causal-parity harness ────────────────────────────────────────────────────
from pipelines.rl.causal import check_causal, assert_causal


def _causal_detector(df):
    # event when close > open on the SAME bar (uses only that bar — causal)
    m = df["close"].values > df["open"].values
    idx = np.flatnonzero(m)
    return pd.DataFrame({"bar_idx": idx, "direction": 1,
                         "sl_distance": 1.0, "tp_rr": 2.0})


def _lookahead_detector(df):
    # event when NEXT bar's close is higher → peeks ahead (NON-causal)
    c = df["close"].values
    m = np.zeros(len(c), bool)
    m[:-1] = c[1:] > c[:-1]
    idx = np.flatnonzero(m)
    return pd.DataFrame({"bar_idx": idx, "direction": 1,
                         "sl_distance": 1.0, "tp_rr": 2.0})


def _df(n=60):
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.standard_normal(n))
    return pd.DataFrame({"open": base, "high": base + 1, "low": base - 1,
                         "close": base + rng.standard_normal(n) * 0.3})


def test_causal_detector_passes():
    ok, mm = check_causal(_causal_detector, _df(60))
    assert ok and mm == []
    assert_causal(_causal_detector, _df(60))      # no raise


def test_lookahead_detector_is_rejected():
    ok, mm = check_causal(_lookahead_detector, _df(60))
    assert not ok and len(mm) > 0
    with pytest.raises(AssertionError, match="NON-CAUSAL"):
        assert_causal(_lookahead_detector, _df(60))


# ── run_walkforward driver (injected trainer — no SB3 dep) ───────────────────
from pipelines.rl.pipeline import run_walkforward, RLConfig, ScriptedPolicy


def _take_then_exit(obs):
    # obs = [ctx(3), in_trade, bars_held_norm, unreal_R, room]
    if obs[3] == 0.0:           # PRE_ENTRY → take
        return 1
    return 1 if obs[4] >= 0.04 else 0      # hold ~few bars then exit


class _StubTrainer:
    def __init__(self, fn): self.fn = fn
    def train(self, episodes, seed): return ScriptedPolicy(self.fn)


def _wf_data(n=2600, trend=0.6):
    base = 100.0 + np.arange(n) * trend
    idx = pd.date_range("2023-01-01", periods=n, freq="1h",
                         tz="America/New_York")
    df = pd.DataFrame({"open": base, "high": base + 0.5, "low": base - 0.5,
                       "close": base, "volume": 100.0}, index=idx)
    ctx = np.tile(np.array([[0.1, 0.2, 0.3]], np.float32), (n, 1))
    return {"ES": (df, ctx)}


class _WFStrategy(RLStrategy):
    name = "wf"
    entry_filter = True
    max_hold = 130

    def detect_entries(self, df_raw, ctx_df, ticker):
        idx = list(range(30, len(df_raw) - 10, 13))
        return pd.DataFrame({"bar_idx": idx, "direction": 1,
                             "sl_distance": 2.0, "tp_rr": 2.0})


def test_run_walkforward_returns_verdict_structure():
    res = run_walkforward(_WFStrategy(), _wf_data(),
                          RLConfig(seeds=(0, 1)),
                          trainer=_StubTrainer(_take_then_exit))
    assert set(res) == {"verdict", "multiseed", "per_seed"}
    assert isinstance(res["verdict"], bool)
    assert res["multiseed"]["n"] == 2
    for p in res["per_seed"]:
        assert {"agg", "gate", "robust", "n"} <= set(p)
        assert p["agg"]["trades"] > 0          # OOS trades were produced


def test_shape_reward_override_changes_pnl():
    base = run_walkforward(_WFStrategy(), _wf_data(),
                           RLConfig(seeds=(0,), shuffle_control=False),
                           trainer=_StubTrainer(_take_then_exit))

    class _Scaled(_WFStrategy):
        name = "scaled"
        def shape_reward(self, realized_r, run_state):
            return realized_r * 3.0            # plug-in custom (e.g. sizing)

    scaled = run_walkforward(_Scaled(), _wf_data(),
                             RLConfig(seeds=(0,), shuffle_control=False),
                             trainer=_StubTrainer(_take_then_exit))
    assert scaled["per_seed"][0]["agg"]["pnl"] == pytest.approx(
        base["per_seed"][0]["agg"]["pnl"] * 3.0, rel=1e-6)


def test_shape_reward_stopiteration_blows_account():
    class _Blown(_WFStrategy):
        name = "blown"
        def shape_reward(self, realized_r, run_state):
            if len(run_state["cum_r"]) >= 3:   # MLL-style: stop after 3
                raise StopIteration
            return realized_r

    res = run_walkforward(_Blown(), _wf_data(),
                          RLConfig(seeds=(0,), shuffle_control=False),
                          trainer=_StubTrainer(_take_then_exit))
    # each window's rollout terminates at <=3 trades (account blown)
    assert res["per_seed"][0]["agg"]["trades"] <= 3 * 20   # generous upper bound
    assert res["per_seed"][0]["agg"]["trades"] > 0


# ── augment_obs: account/MLL-aware observation (generic hook) ─────────────────
from pipelines.rl.env import SingleTradeEnv as _STE


class _MLLStrategy(_WFStrategy):
    """Plug-in style: appends a 'buffer-to-zero' feature so PPO can SEE how
    close the account is to blowing (generic; the real MLL math is IP)."""
    name = "mll"
    extra_obs_dim = 1
    start_balance_R = 5.0                         # external MLL buffer (in R)

    def _buffer(self, run_state):
        return self.start_balance_R + float(np.sum(run_state.get("cum_r", []) or [0.0]))

    def augment_obs(self, obs, run_state):
        return np.concatenate([obs, [np.float32(self._buffer(run_state))]])

    def shape_reward(self, realized_r, run_state):
        if self._buffer(run_state) + realized_r <= 0.0:
            raise StopIteration                  # account blown → terminate
        return realized_r


def test_augment_obs_grows_obs_dim_and_reflects_run_state():
    ctx = np.zeros((20, 3), np.float32)
    o = 100 + np.arange(20) * 1.0
    s = _MLLStrategy()
    rs = {"cum_r": [-1.0, -1.0]}                  # 2 losses → buffer 5-2 = 3
    env = _STE(ctx, o, o + 0.5, o - 0.5, o, entry_bar=2, direction=1,
               sl_distance=1.0, strategy=s, run_state=rs)
    assert env.obs_dim == 3 + 4 + 1
    obs = env.reset()
    assert obs.shape == (8,)
    assert obs[-1] == pytest.approx(3.0)         # buffer-to-zero feature
    rs["cum_r"].append(-1.0)                      # account moves
    assert env.reset()[-1] == pytest.approx(2.0)  # obs tracks it live


def test_augment_obs_wrong_length_raises():
    class _Bad(_MLLStrategy):
        def augment_obs(self, obs, run_state):
            return obs                            # forgot the extra feature
    ctx = np.zeros((20, 3), np.float32); o = 100 + np.arange(20) * 1.0
    env = _STE(ctx, o, o + .5, o - .5, o, entry_bar=2, direction=1,
               sl_distance=1.0, strategy=_Bad(), run_state={"cum_r": []})
    with pytest.raises(ValueError, match="augment_obs returned"):
        env.reset()


def test_run_walkforward_with_account_aware_strategy():
    res = run_walkforward(_MLLStrategy(), _wf_data(),
                          RLConfig(seeds=(0,), shuffle_control=False),
                          trainer=_StubTrainer(_take_then_exit))
    assert isinstance(res["verdict"], bool)
    assert res["per_seed"][0]["agg"]["trades"] > 0   # ran end-to-end w/ aug obs
