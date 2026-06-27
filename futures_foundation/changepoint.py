"""Bayesian Online Change-Point Detection (BOCPD) — a causal "the dynamics just
shifted" signal for trend-start detection.

A trend start IS a change-point in the price dynamics. Where the regime HMM
describes the ambient *state*, BOCPD flags the *transition* — the probability
that a regime break just occurred at this bar. It's a market-state signal, so it
lives in the foundation (reusable by any strategy) and is appended as features
to the head, exactly like the regime posteriors.

WHY BOCPD (not a hand-tuned trigger): it's fully Bayesian/online — the only knob
is a SOFT hazard prior (expected run length), not a hard threshold to overfit.
It outputs a probability, updated causally per bar (Adams & MacKay 2007).

CAUSALITY: the feature at bar t uses only observations x[0..t] (online forward
recursion) — no future leak, by construction.

Conjugate model: Normal-Gamma prior on each run-length hypothesis -> Student-t
predictive. Run-length distribution truncated at `rmax` for bounded cost.
"""
from __future__ import annotations

import numpy as np


def bocpd_features(x, hazard_lambda=250.0, rmax=300, warmup=20):
    """Run BOCPD over a 1-D series `x` (e.g. returns). CAUSAL: row t uses only
    x[0..t]. Returns [T, 2]:
       col 0 = changepoint probability (posterior mass on run-length 0 at t)
       col 1 = normalized expected run-length (0=just changed, 1=long-stable)
    hazard_lambda = expected run length (soft prior; H = 1/lambda).

    The input is standardized with a CAUSAL expanding z-score (scale-invariant —
    works on any return scale, no future leak) and the run-length hypotheses use
    a fixed weakly-informative Normal-Gamma prior (predictive scale ~1)."""
    from scipy.special import gammaln
    x = np.asarray(x, float)
    T = len(x)
    out = np.zeros((T, 2), float)
    if T < 2:
        return out
    # causal expanding z-score (row t uses only x[0..t]) -> scale-invariant
    nn = np.arange(1, T + 1)
    m = np.cumsum(x) / nn
    v = np.maximum(np.cumsum(x * x) / nn - m * m, 1e-12)
    xz = (x - m) / np.sqrt(v)
    xz[:warmup] = 0.0                              # unstable warmup -> neutral
    x = xz
    # fixed weakly-informative prior on the standardized series (pred. scale ~1)
    mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 0.5
    H = 1.0 / hazard_lambda
    # sufficient stats per active run-length hypothesis (start: just the prior)
    mu = np.array([mu0]); kap = np.array([kappa0])
    al = np.array([alpha0]); be = np.array([beta0])
    R = np.array([1.0])                                   # run-length posterior
    for t in range(T):
        xt = x[t]
        # Student-t predictive prob of xt under each run-length hypothesis
        df = 2 * al
        scale2 = be * (kap + 1.0) / (al * kap)
        z = (xt - mu) ** 2 / (df * scale2)
        logp = (gammaln((df + 1) / 2) - gammaln(df / 2)
                - 0.5 * np.log(df * np.pi * scale2) - (df + 1) / 2 * np.log1p(z))
        pred = np.exp(logp)
        # growth (no change) + changepoint (reset to run-length 0)
        growth = R * pred * (1.0 - H)
        cp = float((R * pred * H).sum())
        newR = np.empty(len(R) + 1)
        newR[0] = cp
        newR[1:] = growth
        s = newR.sum()
        newR /= (s if s > 0 else 1e-300)
        # conjugate Normal-Gamma update (prepend prior for the new run-length 0)
        mu1 = (kap * mu + xt) / (kap + 1.0)
        kap1 = kap + 1.0
        al1 = al + 0.5
        be1 = be + 0.5 * kap * (xt - mu) ** 2 / (kap + 1.0)
        mu = np.concatenate(([mu0], mu1))
        kap = np.concatenate(([kappa0], kap1))
        al = np.concatenate(([alpha0], al1))
        be = np.concatenate(([beta0], be1))
        # truncate to rmax for bounded cost/memory
        if len(newR) > rmax:
            newR = newR[:rmax]; newR /= newR.sum()
            mu = mu[:rmax]; kap = kap[:rmax]; al = al[:rmax]; be = be[:rmax]
        R = newR
        rl = np.arange(len(R))
        # col0 = P(run-length <= kshort): "recently changed" score — spikes when
        # the posterior collapses to short run-lengths after a break. (R[0]
        # itself is always == H, so it carries no signal; the DISTRIBUTION does.)
        kshort = 10
        out[t, 0] = float(R[:min(kshort + 1, len(R))].sum())
        out[t, 1] = min(float((R * rl).sum()), rmax) / rmax   # norm E[run-length]
    return np.clip(out, 0.0, 1.0)


N_CHANGEPOINT_FEATS = 2


CHANGEPOINT_RETURN_COL = 'ret_close_1'


def change_point_from_feature(keys, feat, feat_names, col=CHANGEPOINT_RETURN_COL,
                              hazard_lambda=250.0):
    """Change-point features [N, 2] computed from an EXISTING return column in the
    labeler's feature matrix (default 'ret_close_1'), per stream — the pipeline
    entry point. Mirrors the validated combo study: BOCPD on the per-bar returns,
    causal, scattered to both directions. Additive market-state signal; strongest
    WITH the regime HMM ('we just shifted into this regime')."""
    from .regime import _stream_index
    feat = np.asarray(feat, np.float32)
    if col not in feat_names:
        raise ValueError(f"change-point needs '{col}' in labeler.feature_names()")
    ret = feat[:, feat_names.index(col)]
    out = np.zeros((len(keys), N_CHANGEPOINT_FEATS), np.float32)
    for s in _stream_index(keys).values():
        out[s['rows']] = bocpd_features(
            ret[s['uniq_rows']], hazard_lambda=hazard_lambda)[s['pos']]
    return out


def change_point_features(keys, series, hazard_lambda=250.0):
    """Per-signal change-point features [N, 2] from a per-signal scalar `series`
    (e.g. the decision-bar log-close). Groups into per-stream time-ordered
    sequences (reusing the regime stream index), runs BOCPD on each stream's
    RETURNS, and scatters the causal features back to every signal on each bar.
    keys[i] = (item_id, bar_index, ...). Leak-safe: BOCPD is online/causal."""
    from .regime import _stream_index
    series = np.asarray(series, float)
    idx = _stream_index(keys)
    out = np.zeros((len(keys), N_CHANGEPOINT_FEATS), np.float32)
    for d in idx.values():
        u = d['uniq_rows']
        s = series[u]
        if len(s) < 2:
            continue
        r = np.diff(s, prepend=s[0])          # returns; first = 0 (no change)
        f = bocpd_features(r, hazard_lambda=hazard_lambda)
        out[d['rows']] = f[d['pos']]          # vectorized scatter (both dirs share)
    return out
