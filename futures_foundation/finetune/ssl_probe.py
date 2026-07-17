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
                'fwd_absmove': 'reg', 'direction': 'bin', 'fwd_dir': 'bin'}
# CORE targets define a "useful for buy/sell classification" representation: regime/vol/
# structure (descriptive) + fwd_absmove (is there a tradeable move next — buy/sell-relevant
# and learnable). direction (in-window) and fwd_dir (forward direction) are reported but
# NOT in the pass criterion — directional prediction is genuinely hard / noisy as a gate.
_CORE_TARGETS = ('vol', 'trend_eff', 'range_expand', 'fwd_absmove')


def non_overlapping_rows(starts, span, group_ids=None):
    """Return row indices for a deterministic maximal subset of non-overlapping windows.

    Probe train/test rows must not share raw bars. The SSL validation start set is dense, and a
    random row split otherwise puts near-identical overlapping windows on both sides. ``span``
    includes the embedded context and every forward target bar.
    """
    starts = np.asarray(starts, np.int64)
    if len(starts) == 0:
        return np.array([], np.int64)
    groups = (np.zeros(len(starts), np.int64) if group_ids is None
              else np.asarray(group_ids, np.int64))
    if len(groups) != len(starts):
        raise ValueError("group_ids must have one entry per start")
    keep = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        rows = rows[np.argsort(starts[rows], kind='stable')]
        next_start = None
        for row in rows:
            start = int(starts[row])
            if next_start is None or start >= next_start:
                keep.append(int(row))
                next_start = start + int(span)
    return np.asarray(sorted(keep), np.int64)


def walk_forward_splits(starts, group_ids, folds=5, span=80, *, timestamps=None,
                        span_ns=None):
    """Expanding, stream-aware probe splits with a full-window embargo.

    Each symbol/timeframe stream is ordered independently.  Fold ``i`` trains only on
    that stream's earlier blocks and tests on its next block.  A train example is kept
    only when its complete context+forward span ends at least another ``span`` bars
    before the first test example.  Thus neither raw-bar overlap nor later-to-earlier
    probe fitting can inflate the checkpoint comparison.
    """
    starts = np.asarray(starts, np.int64)
    groups = np.asarray(group_ids, np.int64)
    folds, span = int(folds), int(span)
    if len(starts) != len(groups) or len(starts) == 0:
        raise ValueError("starts/group_ids must be non-empty and have equal length")
    if folds < 1 or span < 1:
        raise ValueError("folds and span must be positive")

    if timestamps is not None:
        times = np.asarray(timestamps, np.int64)
        if len(times) != len(starts) or span_ns is None or int(span_ns) < 1:
            raise ValueError("calendar walk-forward requires aligned timestamps and span_ns")
        # Shared wall-clock edges prevent the same market event appearing in probe training at
        # one timeframe and probe testing at another.  Reserve both the complete maximum
        # context+target duration and an equally long embargo before every test block.
        lo, hi = int(times.min()), int(times.max()) + 1
        edges = np.linspace(lo, hi, folds + 2).astype(np.int64)
        splits = []
        for fold in range(1, folds + 1):
            test_lo, test_hi = int(edges[fold]), int(edges[fold + 1])
            tr = np.flatnonzero(times + 2 * int(span_ns) <= test_lo)
            te = np.flatnonzero((times >= test_lo) & (times < test_hi))
            for group in np.unique(groups):
                if not np.any(groups[tr] == group) or not np.any(groups[te] == group):
                    raise ValueError(f"probe group {group} fold {fold} is empty after calendar embargo")
            if np.intersect1d(tr, te).size:
                raise RuntimeError("walk-forward probe train/test rows overlap")
            splits.append((tr, te))
        return splits

    blocks = {}
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        rows = rows[np.argsort(starts[rows], kind='stable')]
        if len(rows) < folds + 1:
            raise ValueError(f"probe group {group} has only {len(rows)} rows for {folds} folds")
        blocks[int(group)] = np.array_split(rows, folds + 1)

    splits = []
    for fold in range(1, folds + 1):
        train_parts, test_parts = [], []
        for group, chunks in blocks.items():
            test = chunks[fold]
            earlier = np.concatenate(chunks[:fold])
            first_test = int(starts[test].min())
            # Train window [s,s+span) plus an equally long embargo must end by first_test.
            train = earlier[starts[earlier] + 2 * span <= first_test]
            if len(train) == 0 or len(test) == 0:
                raise ValueError(f"probe group {group} fold {fold} is empty after embargo")
            train_parts.append(train)
            test_parts.append(test)
        tr, te = np.concatenate(train_parts), np.concatenate(test_parts)
        if np.intersect1d(tr, te).size:
            raise RuntimeError("walk-forward probe train/test rows overlap")
        splits.append((tr, te))
    return splits


def targets_from_windows(big, starts, seq, fwd_k=16):
    """Compute the probe targets for each window [start, start+seq). big: [T, 5]
    (O,H,L,C,V). Descriptive targets come from the window; FORWARD targets (buy/sell-
    relevant) come from the next `fwd_k` bars AFTER the window — strictly future relative
    to the embedded window (no leak). Caller keeps fwd_k <= max_jitter so the forward bars
    stay in-stream. Returns dict name -> float array [M]."""
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
    # FORWARD (buy/sell): log-return over the next fwd_k bars after the window end
    end = np.clip(s + seq - 1, 0, len(big) - 1)
    fwd = np.clip(s + seq - 1 + fwd_k, 0, len(big) - 1)
    fwd_ret = (np.log(np.clip(big[fwd, 3], 1e-9, None))
               - np.log(np.clip(big[end, 3], 1e-9, None)))
    return {'vol': vol.astype(np.float32), 'trend_eff': trend_eff.astype(np.float32),
            'range_expand': range_expand.astype(np.float32),
            'fwd_absmove': np.abs(fwd_ret).astype(np.float32),
            'direction': (net > 0).astype(np.int32),
            'fwd_dir': (fwd_ret > 0).astype(np.int32)}


def _fit_score(emb, y, kind, tr, te):
    """Fit a linear probe on emb[tr] -> score on emb[te]. R2 (reg) or AUC (bin)."""
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, roc_auc_score
    sc = StandardScaler().fit(emb[tr])
    Xtr, Xte = sc.transform(emb[tr]), sc.transform(emb[te])
    if kind == 'bin':
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            return 0.5
        # 1280-dim correlated embeddings -> lbfgs hits its iter cap; bump it, and suppress the
        # (harmless, diagnostic) ConvergenceWarning so it doesn't bury the run's progress lines.
        # A probe only needs a stable AUC, not exact convergence — the score is valid regardless.
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
        return float(roc_auc_score(y[te], m.predict_proba(Xte)[:, 1]))
    # Contrastive embeddings can be strongly collinear.  The default Cholesky path emits
    # ill-conditioned-matrix warnings and can make tiny checkpoint deltas solver artifacts.
    # LSQR is deterministic here and handles the regularized least-squares system directly.
    m = Ridge(alpha=1.0, solver='lsqr', tol=1e-6).fit(Xtr, y[tr])
    return float(r2_score(y[te], m.predict(Xte)))


def _probe_scores(emb, y, kind, seed=0, test_frac=0.3, folds=1, splits=None):
    """Return one score per evaluation fold instead of hiding dispersion in a mean."""
    emb = np.asarray(emb, np.float32)
    n = len(emb)
    if splits is not None:
        return [_fit_score(emb, y, kind, tr, te) for tr, te in splits]
    # Dense SSL windows overlap heavily. The default must stay temporal; shuffled probes are
    # retained only to reproduce explicitly requested historical results.
    import os
    legacy = os.environ.get('PROBE_SPLIT', 'contiguous') == 'random'
    if folds and folds > 1:
        from sklearn.model_selection import KFold
        kf = (KFold(n_splits=int(folds), shuffle=True, random_state=seed) if legacy
              else KFold(n_splits=int(folds), shuffle=False))
        return [_fit_score(emb, y, kind, tr, te) for tr, te in kf.split(emb)]
    idx = np.random.default_rng(seed).permutation(n) if legacy else np.arange(n)
    nt = int(n * (1 - test_frac))
    return [_fit_score(emb, y, kind, idx[:nt], idx[nt:])]


def probe_embedding(emb, y, kind, seed=0, test_frac=0.3, folds=1, splits=None):
    """Fit a linear probe and score it. folds<=1 -> a single random train/test split (fast).
    folds>1 -> k-fold CV, returning the MEAN score over folds -> a robust, low-variance estimate
    (use this to reliably RANK candidates, e.g. channel-weight configs, where a single split is
    too noisy to trust the ordering)."""
    return float(np.mean(_probe_scores(emb, y, kind, seed, test_frac, folds, splits)))


def compare(emb_ssl, emb_vanilla, targets, seed=0, folds=1, splits=None):
    """Probe both embeddings on every target; return per-target {ssl, vanilla, delta} plus
    aggregate deltas that the gate uses. We separate DESCRIPTIVE content (vol/trend_eff/
    range_expand — easy in-window stats) from FORWARD predictive content (fwd_absmove size,
    fwd_dir direction), because a shortcut embedding can lift the easy descriptive stats while
    the forward (genuinely predictive) targets barely move. The gate keys on the FORWARD ones.
    folds>1 -> k-fold CV per probe (robust deltas for ranking candidates)."""
    res, core_deltas, desc_deltas = {}, [], []
    for name, y in targets.items():
        kind = _TARGET_KIND[name]
        af = _probe_scores(emb_ssl, y, kind, seed, folds=folds, splits=splits)
        bf = _probe_scores(emb_vanilla, y, kind, seed, folds=folds, splits=splits)
        fold_delta = np.asarray(af, dtype=float) - np.asarray(bf, dtype=float)
        a, b = float(np.mean(af)), float(np.mean(bf))
        delta_std = float(np.std(fold_delta, ddof=1)) if len(fold_delta) > 1 else 0.0
        delta_se = delta_std / np.sqrt(len(fold_delta)) if len(fold_delta) > 1 else 0.0
        res[name] = {'ssl': a, 'vanilla': b, 'delta': a - b, 'kind': kind,
                     'fold_delta': fold_delta.tolist(), 'delta_std': delta_std,
                     'delta_lcb_95': float((a - b) - 1.96 * delta_se)}
        if name in _CORE_TARGETS:
            core_deltas.append(a - b)
        if name in ('vol', 'trend_eff', 'range_expand'):
            desc_deltas.append(a - b)
    mean_core = float(np.mean(core_deltas)) if core_deltas else 0.0
    fwd_absmove_delta = float(res['fwd_absmove']['delta']) if 'fwd_absmove' in res else 0.0
    fwd_dir_delta = float(res['fwd_dir']['delta']) if 'fwd_dir' in res else 0.0
    return {'per_target': res, 'mean_core_delta': mean_core,
            'descriptive_delta': float(np.mean(desc_deltas)) if desc_deltas else 0.0,
            'fwd_absmove_delta': fwd_absmove_delta,   # forward MOVE SIZE (R2) vs vanilla
            'fwd_dir_delta': fwd_dir_delta,           # forward DIRECTION (AUC) vs vanilla
            'forward_score': fwd_absmove_delta + fwd_dir_delta,   # combined forward relevance
            'probe_protocol': ('expanding_walk_forward' if splits is not None else
                               ('random_kfold' if folds and folds > 1 else 'random_holdout')),
            'probe_folds': (len(splits) if splits is not None else int(folds or 1)),
            'learns_regime_vol_structure': bool(mean_core > 0.0)}


def run_probe(big, starts, seq, ssl_ckpt, *, model_id='paris-noah/Mantis-8M', model_version=None,
              device=None, max_windows=20000, batch=512, seed=0, fwd_k=16, folds=1,
              group_ids=None, timestamps=None, span_ns=None, verbose=True, preprocessing=None):
    """Extract SSL-adapted vs vanilla encoder embeddings for held-out windows and
    compare on the probe targets (regime/vol/structure + forward buy/sell move). Returns
    the compare() dict. Production scoring is stream-aware expanding walk-forward with
    non-overlapping examples and a full context+forward-span embargo."""
    from . import _ssl_torch
    starts = np.asarray(starts, np.int64)
    legacy_probe = group_ids is None or timestamps is None or span_ns is None
    if legacy_probe:
        # Compatibility only for a trainer process started before the walk-forward caller was
        # installed.  It is loudly tagged and cannot attest a parent checkpoint for the next
        # stage; scripts/train_ssl_local.py requires a separate strict_probe attestation.
        print("  [probe] LEGACY RANDOM SPLIT — DIAGNOSTIC ONLY; STRICT ATTESTATION REQUIRED",
              flush=True)
        group_ids = np.zeros(len(starts), np.int64)
        timestamps = np.arange(len(starts), dtype=np.int64)
    else:
        group_ids = np.asarray(group_ids, np.int64)
        timestamps = np.asarray(timestamps, np.int64)
    if len(group_ids) != len(starts) or len(timestamps) != len(starts):
        raise ValueError("group_ids/timestamps must have one entry per probe start")
    emb_ssl, used = _ssl_torch.embed_encoder(big, starts, seq, ckpt=ssl_ckpt,
                                             model_id=model_id, model_version=model_version,
                                             device=device,
                                             batch=batch, max_windows=max_windows, seed=seed,
                                             preprocessing=preprocessing)
    if not np.array_equal(used, starts):
        raise RuntimeError("probe embedding sampler changed the sealed start rows")
    keep = non_overlapping_rows(used, seq + fwd_k, group_ids=group_ids)
    emb_ssl, used = emb_ssl[keep], np.asarray(used)[keep]
    used_groups = group_ids[keep]
    used_timestamps = timestamps[keep]
    emb_van, _ = _ssl_torch.embed_encoder(big, used, seq, ckpt=None, model_id=model_id,
                                          model_version=model_version, device=device, batch=batch,
                                          max_windows=len(used), seed=seed,
                                          preprocessing=preprocessing)
    tgt = targets_from_windows(big, used, seq, fwd_k=fwd_k)
    splits = (None if legacy_probe else walk_forward_splits(
        used, used_groups, folds=folds, span=seq + fwd_k,
        timestamps=used_timestamps, span_ns=span_ns))
    out = compare(emb_ssl, emb_van, tgt, seed=seed, folds=folds, splits=splits)
    if legacy_probe:
        out['probe_protocol'] = 'legacy_random_diagnostic_only'
        out['strict_attestation_required'] = True
    out['probe_embargo_bars'] = int(seq + fwd_k)
    out['probe_max_span_ns'] = None if span_ns is None else int(span_ns)
    if verbose:
        for name, d in out['per_target'].items():
            print(f"  [probe] {name:>12} ({d['kind']}) ssl={d['ssl']:.4f} "
                  f"vanilla={d['vanilla']:.4f} delta={d['delta']:+.4f}", flush=True)
        print(f"  [probe] mean_core_delta={out['mean_core_delta']:+.4f} "
              f"learns_regime_vol_structure={out['learns_regime_vol_structure']}", flush=True)
    return out
