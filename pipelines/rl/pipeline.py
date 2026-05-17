"""RL walk-forward driver — generic, model-agnostic, dependency-light.

Reuses the validated spine (pipelines.common: walk_forward_windows,
robustness) exactly like xgboost. The PPO trainer is an INJECTED seam
(`trainer.train(episodes, seed) -> policy`); the default lazily imports
SB3 only when actually training, so this module + its tests need no RL
deps. Strategy customization (incl. prop-firm/MLL) is ONLY via
strategy.shape_reward — the pipeline has no such concept.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipelines.common.walkforward import walk_forward_windows
from pipelines.common.robustness import shuffle_robust, multiseed_verdict
from .env import SingleTradeEnv


@dataclass
class RLConfig:
    train_months: int = 3
    test_months: int = 1
    seeds: tuple = (0, 1, 2)
    min_median: float = 0.0
    shuffle_control: bool = True


class ScriptedPolicy:
    """Deterministic obs->action policy (tests / baselines)."""
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, obs):
        return int(self.fn(obs))


def _agg(rs) -> dict:
    r = np.asarray(rs, float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {"trades": 0, "pnl": 0.0, "profit_factor": 0.0, "mean_r": 0.0}
    gp = float(r[r > 0].sum()); gl = float(-r[r < 0].sum())
    return {"trades": int(len(r)), "pnl": float(r.sum()),
            "profit_factor": (gp / gl) if gl > 0 else float("inf"),
            "mean_r": float(r.mean())}


def _every_oos_month_pf_gt1(dated_r) -> bool:
    if not dated_r:
        return False
    df = pd.DataFrame(dated_r, columns=["dt", "r"])
    _dt = pd.to_datetime(df["dt"])
    if getattr(_dt.dt, "tz", None) is not None:     # tz-immaterial for month
        _dt = _dt.dt.tz_localize(None)
    df["m"] = _dt.dt.to_period("M")
    for _, g in df.groupby("m"):
        gp = g.loc[g.r > 0, "r"].sum(); gl = -g.loc[g.r < 0, "r"].sum()
        if not (gp > gl):                       # monthly PF must exceed 1
            return False
    return True


def _episodes(strategy, df, ctx, mask, run_state):
    """SingleTradeEnv per detected entry whose signal bar is in `mask`. All
    envs share `run_state` so augment_obs sees the live account state."""
    ev = strategy.detect_entries(df, df, "T")
    o = df["open"].values; h = df["high"].values
    l = df["low"].values;  c = df["close"].values
    out = []
    for _, e in ev.iterrows():
        bi = int(e["bar_idx"])
        if bi < 0 or bi >= len(df) or not mask[bi]:
            continue
        out.append((df.index[bi], SingleTradeEnv(
            ctx[bi:], o, h, l, c, entry_bar=bi, direction=int(e["direction"]),
            sl_distance=float(e["sl_distance"]), tp_rr=float(e["tp_rr"]),
            entry_filter=strategy.entry_filter, max_hold=strategy.max_hold,
            strategy=strategy, run_state=run_state)))
    return out


def _rollout(strategy, episodes, policy, rng, shuffle, run_state):
    """Returns (dated_trades, terminated_early). terminated_early=True when
    the strategy raised StopIteration mid-run (a self-abort, e.g. an
    account-blown signal) — a run that self-terminated did NOT complete and
    must FAIL the verdict, not merely score the partial trades."""
    dated = []
    terminated = False
    for i in range(len(episodes)):
        dt, env = episodes[i]
        obs = env.reset(); done = False; r = 0.0
        while not done:
            obs, r, done, _, _ = env.step(policy(obs))
        if shuffle:                              # break entry↔outcome link
            r = float(rng.standard_normal()) * 0.0  # shuffled = no signal
        try:
            r = float(strategy.shape_reward(r, run_state))
        except StopIteration:                    # strategy self-aborted run
            terminated = True
            break
        run_state["cum_r"].append(r)
        dated.append((dt, r))
    return dated, terminated


def _run_seed(strategy, data, cfg, trainer, seed, shuffle):
    rng = np.random.default_rng(seed)
    oos = []
    terminated = False
    for tk, (df, ctx) in data.items():
        for tr_mask, te_mask in walk_forward_windows(
                df.index, cfg.train_months, cfg.test_months):
            rs_train = {"cum_r": []}            # account state during training
            rs_test = {"cum_r": []}             # fresh account for the OOS run
            train_eps = _episodes(strategy, df, ctx, tr_mask, rs_train)
            test_eps = _episodes(strategy, df, ctx, te_mask, rs_test)
            if not test_eps:
                continue
            policy = trainer.train(train_eps, seed)
            d, term = _rollout(strategy, test_eps, policy, rng, shuffle,
                               rs_test)
            oos += d
            if term:
                terminated = True
    agg = _agg([r for _, r in oos])
    agg_gate = _every_oos_month_pf_gt1(oos) if not shuffle else False
    return {"agg": agg, "gate": agg_gate, "n": len(oos),
            "terminated": terminated}


def run_walkforward(strategy, data: dict, cfg: RLConfig = None,
                    trainer=None) -> dict:
    """data = {ticker: (df_raw[DatetimeIndex, OHLC], ctx[ndarray T×d])}.
    trainer.train(train_episodes, seed) -> policy(obs)->action. Returns the
    consolidated verdict (multi-seed + shuffle + every-OOS-month-PF>1)."""
    cfg = cfg or RLConfig()
    if trainer is None:                          # lazy default — no test dep
        from .ppo import make_ppo_trainer
        trainer = make_ppo_trainer()
    seed_pnls, per = [], []
    for sd in cfg.seeds:
        real = _run_seed(strategy, data, cfg, trainer, sd, shuffle=False)
        if cfg.shuffle_control:
            shuf = _run_seed(strategy, data, cfg, trainer, sd, shuffle=True)
            real["robust"] = shuffle_robust(real["agg"], shuf["agg"])
        else:
            real["robust"] = True
        seed_pnls.append(real["agg"]["pnl"])
        per.append(real)
    ms = multiseed_verdict(seed_pnls, min_median=cfg.min_median)
    verdict = bool(ms["pass"]
                   and all(p["robust"] for p in per)
                   and all(p["gate"] for p in per)
                   and not any(p["terminated"] for p in per))  # blown = FAIL
    return {"verdict": verdict, "multiseed": ms, "per_seed": per}
