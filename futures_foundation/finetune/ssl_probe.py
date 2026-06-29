"""Linear-probe validation — does the SSL backbone learn regime / volatility /
structure useful for downstream classification? (the BERT 'probing' check).

Freeze the encoder, take its embedding of clean held-out windows, and fit a LINEAR
probe to predict OHLCV-derived properties of each window. If the SSL-adapted encoder
beats the VANILLA Mantis encoder on these, the contrastive pretraining demonstrably
encoded more downstream-useful structure — BEFORE any strategy head is trained.

Probe targets (computed from the raw window, no strategy labels, no future leak):
  * vol          — realized volatility (std of log-returns)            [regression R2]
  * trend_eff    — directional efficiency |net|/sum|step| (trend<->chop)[regression R2]
  * range_expand — log(2nd-half range / 1st-half range) (compression-> [regression R2]
                   expansion = momentum compression/expansion)
  * direction    — sign of net return                                  [binary AUC]

Target computation is torch-free (numpy) + testable; the embedding extraction lives in
_ssl_torch.embed_encoder (lazy torch).
"""
import numpy as np

_TARGET_KIND = {'vol': 'reg', 'trend_eff': 'reg', 'range_expand': 'reg',
                'direction': 'bin'}
# regime/vol/structure targets that define a "useful" representation (direction is
# reported but excluded from the pass criterion — Mantis is known weak on direction).
_CORE_TARGETS = ('vol', 'trend_eff', 'range_expand')


def targets_from_windows(big, starts, seq):
    """Compute the probe targets for each window [start, start+seq). big: [T, 5]
    (O,H,L,C,V). Returns dict name -> float array [M]."""
    big = np.asarray(big, np.float64)
    s = np.asarray(starts, np.int64)
    rows = s[:, None] + np.arange(seq)[None, :]              # [M, seq]
    win = big[rows]                                          # [M, seq, 5]
    high, low, close = win[:, :, 1], win[:, :, 2], win[:, :, 3]
    logret = np.diff(np.log(np.clip(close, 1e-9, None)), axis=1)   # [M, seq-1]
    net = logret.sum(1)
    vol = logret.std(1)
    trend_eff = np.abs(net) / (np.abs(logret).sum(1) + 1e-9)
    h = seq // 2
    r1 = high[:, :h].max(1) - low[:, :h].min(1)
    r2 = high[:, h:].max(1) - low[:, h:].min(1)
    range_expand = np.log((r2 + 1e-9) / (r1 + 1e-9))
    return {'vol': vol.astype(np.float32), 'trend_eff': trend_eff.astype(np.float32),
            'range_expand': range_expand.astype(np.float32),
            'direction': (net > 0).astype(np.int32)}


def probe_embedding(emb, y, kind, seed=0, test_frac=0.3):
    """Fit a linear probe (Ridge for 'reg', Logistic for 'bin') on a train split of
    `emb`, score on the held-out split. Returns R2 (reg) or AUC (bin)."""
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, roc_auc_score
    emb = np.asarray(emb, np.float32)
    n = len(emb)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    nt = int(n * (1 - test_frac))
    tr, te = idx[:nt], idx[nt:]
    sc = StandardScaler().fit(emb[tr])
    Xtr, Xte = sc.transform(emb[tr]), sc.transform(emb[te])
    if kind == 'bin':
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            return 0.5
        m = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, y[tr])
        return float(roc_auc_score(y[te], m.predict_proba(Xte)[:, 1]))
    m = Ridge(alpha=1.0).fit(Xtr, y[tr])
    return float(r2_score(y[te], m.predict(Xte)))


def compare(emb_ssl, emb_vanilla, targets, seed=0):
    """Probe both embeddings on every target; return per-target {ssl, vanilla, delta}
    plus a pass flag (SSL >= vanilla on the core regime/vol/structure targets)."""
    res, core_deltas = {}, []
    for name, y in targets.items():
        kind = _TARGET_KIND[name]
        a = probe_embedding(emb_ssl, y, kind, seed)
        b = probe_embedding(emb_vanilla, y, kind, seed)
        res[name] = {'ssl': a, 'vanilla': b, 'delta': a - b, 'kind': kind}
        if name in _CORE_TARGETS:
            core_deltas.append(a - b)
    mean_core = float(np.mean(core_deltas)) if core_deltas else 0.0
    return {'per_target': res, 'mean_core_delta': mean_core,
            'learns_regime_vol_structure': bool(mean_core > 0.0)}


def run_probe(big, starts, seq, ssl_ckpt, *, model_id='paris-noah/Mantis-8M',
              device=None, max_windows=20000, batch=512, seed=0, verbose=True):
    """Extract SSL-adapted vs vanilla encoder embeddings for held-out windows and
    compare on the probe targets. Returns the compare() dict."""
    from . import _ssl_torch
    emb_ssl, used = _ssl_torch.embed_encoder(big, starts, seq, ckpt=ssl_ckpt,
                                             model_id=model_id, device=device,
                                             batch=batch, max_windows=max_windows, seed=seed)
    emb_van, _ = _ssl_torch.embed_encoder(big, used, seq, ckpt=None, model_id=model_id,
                                          device=device, batch=batch,
                                          max_windows=len(used), seed=seed)
    tgt = targets_from_windows(big, used, seq)
    out = compare(emb_ssl, emb_van, tgt, seed=seed)
    if verbose:
        for name, d in out['per_target'].items():
            print(f"  [probe] {name:>12} ({d['kind']}) ssl={d['ssl']:.4f} "
                  f"vanilla={d['vanilla']:.4f} delta={d['delta']:+.4f}", flush=True)
        print(f"  [probe] mean_core_delta={out['mean_core_delta']:+.4f} "
              f"learns_regime_vol_structure={out['learns_regime_vol_structure']}", flush=True)
    return out
