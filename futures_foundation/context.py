"""Context heads — named market-understanding handles on the foundation.

Seven forward-looking context targets (close-only labels — a head must be
able to see the inputs that define its target), and `ContextHeads`:
XGBoost probes trained ONCE on pre-cutoff data, frozen thereafter,
exposing `ctx_*` features downstream models can fuse by name.

FFM 2.1 enriched inputs (2026-06-10, temp/probe_ff68_full.json): heads
consume [Bolt embedding | 68-feature library] instead of the embedding
alone — the EMB+FF68 arm beat the emb-only arm on every surviving head
(e.g. vol_expansion AUC 0.824 vs 0.775, volatility r 0.636 vs 0.521,
quiet_persist AUC 0.744 vs 0.693). Older emb-only bundles keep working
via `meta['inputs']`/`input_dim` back-compat.

Capability evidence (Phase-0 probe, full pre-2023 corpus, shuffle +
trivial-baseline controls): the frozen embedding knows future VOLATILITY
beyond trivial trailing stats (vol percentile r=0.52 vs 0.41 trivial;
expansion AUC 0.78 vs 0.70); knows structure/range at trivial-matching
level; does not know direction. See scripts/probe_context_heads.py.

Leak discipline (hard requirement): heads train only on bars whose
FORWARD label window ends before HEADS_CUTOFF. Downstream signal training
that consumes ctx_* features must use folds at/after HEADS_CUTOFF — the
chronos evaluate/produce seam enforces this.

Pre-registered gates (decided before any training run): a head ships in
`transform()` only if it clears its gate on the pre-cutoff validation
slice. Failing heads stay in the bundle (metrics recorded) but are
excluded from transform by default.

Process contract: this module is torch-free; XGBoost only. Embeddings
arrive via `futures_foundation.foundation.embed_bars` (subprocess seam).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HEADS_CUTOFF = pd.Timestamp('2023-01-01', tz='UTC')
MAX_LABEL_HORIZON = 20            # bars; longest forward window of any head

# Pre-registered probe gates — change only BEFORE a run, never after.
GATE_REG_PEARSON = 0.05
GATE_CLF_AUC = 0.55

#: (name, kind) — kind: 'reg' | 'clf'. Feature name downstream = 'ctx_<name>'.
#: FFM 2.1 enriched set (user-approved prune, 2026-06-10 FF68 probe).
#: fwd_return is the weakest seat (r .099, first time past gate) —
#: flagged marginal. range_pos REMOVED (never beat trivial); trend_start
#: REMOVED (dead in all arms).
HEAD_SPECS = [
    ('fwd_return', 'reg'),
    ('vol_expansion', 'clf'),
    ('volatility', 'reg'),
    ('structure', 'clf'),
    ('quiet_persist', 'clf'),
    ('trendiness', 'reg'),       # trend vs chop: fwd 20-bar efficiency ratio
    ('range_bound', 'clf'),      # ranging: fwd 10-bar closes stay in range
]

#: Candidate heads under probe evaluation — promoted into HEAD_SPECS only
#: after beating BOTH their gate AND the trivial baseline on the probe
#: (scripts/probe_context_heads.py). Currently empty.
#: Graveyard: trendiness/range_bound PROMOTED 2026-06-10 via the FF68
#: enriched arm (temp/probe_ff68_full.json); range_pos/trend_start
#: REMOVED the same day (range_pos never beat trivial; trend_start dead
#: in every arm).
CANDIDATE_HEAD_SPECS = []


def compute_context_labels(close: pd.Series) -> pd.DataFrame:
    """All seven forward-looking labels from a close series. NaN where a
    trailing or forward window is unavailable — never filled.

      fwd_return     reg  20-bar fwd log-return / trailing 200-bar std of
                          20-bar returns, clipped +/-4
      vol_expansion  clf  fwd 20-bar realized vol > 1.5x trailing 200-bar
                          median of 20-bar realized vol
      volatility     reg  fwd 10-bar realized-vol percentile vs trailing
                          100 bars' 10-bar vols, continuous [0,1]
      structure      clf  fwd 20-bar close max/min vs trailing 12-bar close
                          max/min: both higher = 1, both lower = 0, mixed NaN
      trendiness     reg  fwd 20-bar efficiency ratio (|net|/path), [0,1]
      range_bound    clf  fwd 10-bar closes stay in trailing 20-bar range
      quiet_persist  clf  on currently-quiet bars only: quiet persists
                          (fwd vol <= 1.25x trailing median); else NaN
    """
    lc = np.log(close)
    r1 = lc.diff()

    out = pd.DataFrame(index=close.index)

    fwd20 = lc.shift(-20) - lc
    sigma20 = lc.diff(20).rolling(200, min_periods=50).std()
    out['fwd_return'] = (fwd20 / sigma20.replace(0, np.nan)).clip(-4, 4)

    v10 = r1.rolling(10).std()
    v20 = r1.rolling(20).std()

    fwd_v20 = v20.shift(-20)
    med_v20 = v20.rolling(200, min_periods=50).median()
    ve = (fwd_v20 > 1.5 * med_v20).astype(float)
    ve[fwd_v20.isna() | med_v20.isna()] = np.nan
    out['vol_expansion'] = ve

    # percentile of fwd 10-bar vol within the trailing 100 bars' v10 dist
    fwd_v10 = v10.shift(-10).to_numpy()
    v10a = v10.to_numpy()
    pct = np.full(len(v10a), np.nan)
    W = 100
    if len(v10a) > W:
        sw = np.lib.stride_tricks.sliding_window_view(v10a, W)
        tgt = fwd_v10[W - 1:]
        with np.errstate(invalid='ignore'):
            ranks = np.nanmean(sw < tgt[:, None], axis=1)
        bad = np.isnan(tgt) | np.isnan(sw).any(axis=1)
        ranks[bad] = np.nan
        pct[W - 1:] = ranks
    out['volatility'] = pct

    ref_hi = close.rolling(12).max()
    ref_lo = close.rolling(12).min()
    fwd_hi = close.rolling(20).max().shift(-20)   # covers t+1..t+20
    fwd_lo = close.rolling(20).min().shift(-20)
    st = pd.Series(np.nan, index=close.index)
    valid = ref_hi.notna() & ref_lo.notna() & fwd_hi.notna() & fwd_lo.notna()
    st[valid & (fwd_hi > ref_hi) & (fwd_lo > ref_lo)] = 1.0
    st[valid & (fwd_hi < ref_hi) & (fwd_lo < ref_lo)] = 0.0
    out['structure'] = st                          # mixed = NaN sentinel

    rh = close.rolling(20).max()
    rl = close.rolling(20).min()

    # trendiness: efficiency ratio of the NEXT 20 bars — |net move| over
    # path length. 1 = clean trend (either direction), ~0 = chop.
    net = (lc.shift(-20) - lc).abs()
    path = r1.abs().rolling(20).sum().shift(-20)      # covers t+1..t+20
    out['trendiness'] = (net / path.replace(0, np.nan)).clip(0, 1)

    # range_bound: next 10 bars' closes stay inside the current 20-bar
    # close range (the "ranging" state, direction-agnostic).
    fwd_hi10 = close.rolling(10).max().shift(-10)
    fwd_lo10 = close.rolling(10).min().shift(-10)
    rb = ((fwd_hi10 <= rh) & (fwd_lo10 >= rl)).astype(float)
    rb[fwd_hi10.isna() | fwd_lo10.isna() | rh.isna() | rl.isna()] = np.nan
    out['range_bound'] = rb

    # quiet_persist: defined ONLY on currently-quiet bars (trailing 20-bar
    # vol <= trailing median); does quiet persist — no meaningful expansion
    # (fwd vol <= 1.25x median) — over the next 20 bars? The 1.25x buffer
    # keeps the label clean inside a uniform quiet regime (vs comparing to
    # the median itself, which flips ~half the bars by construction).
    # Conditional label — non-quiet bars are NaN (dropped per-head).
    quiet_now = v20 <= med_v20
    qp = (fwd_v20 <= 1.25 * med_v20).astype(float)
    qp[~quiet_now | fwd_v20.isna() | med_v20.isna() | v20.isna()] = np.nan
    out['quiet_persist'] = qp

    return out


def context_features(df: pd.DataFrame, instrument: str) -> np.ndarray:
    """The 68-feature library matrix for enriched heads — strictly trailing.

    Wraps `futures_foundation.features.derive_features(df, instrument)` and
    returns the float32 matrix of the `get_model_feature_columns()` columns
    present, in that fixed order, aligned to `df` rows. NaNs are left as-is
    (XGBoost handles missing values natively).

    df: DataFrame with datetime/open/high/low/close/volume columns.
    -> float32 [len(df), n_features]
    """
    from .features import derive_features, get_model_feature_columns
    fdf = derive_features(df, instrument)
    cols = [c for c in get_model_feature_columns() if c in fdf.columns]
    return fdf[cols].to_numpy(np.float32)


def _fit_head(kind, X, y, seed, n_estimators):
    import xgboost as xgb
    common = dict(n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                  random_state=seed, n_jobs=0)
    if kind == 'reg':
        return xgb.XGBRegressor(objective='reg:squarederror', **common).fit(X, y)
    return xgb.XGBClassifier(objective='binary:logistic',
                             eval_metric='logloss', **common).fit(X, y)


def _score_head(kind, model, X, y):
    if kind == 'reg':
        p = model.predict(X)
        if p.std() == 0 or y.std() == 0:
            return 0.0
        return float(np.corrcoef(p, y)[0, 1])
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


@dataclass
class ContextHeads:
    """Frozen XGBoost context heads on foundation inputs.

    fit() trains every head and applies the pre-registered gate on the
    validation slice; transform() emits ONLY the heads that passed (or an
    explicit `include` override for ablations). save()/load() round-trip
    via joblib with full training metadata.

    fit/transform are matrix-agnostic: X may be the embedding alone
    (legacy emb-only bundles) or [embedding | 68-feature library] (FFM 2.1
    enriched bundles — the trainer records meta['inputs']='emb+ff68' and
    meta['feature_cols']). `input_dim` is captured in fit and enforced in
    transform so a bundle can never be silently fed the wrong matrix.
    """
    seed: int = 0
    n_estimators: int = 400
    models: dict = field(default_factory=dict)      # name -> fitted model
    metrics: dict = field(default_factory=dict)     # name -> metrics dict
    meta: dict = field(default_factory=dict)
    input_dim: int = None                           # set in fit from X width

    @property
    def active_names(self):
        """ctx_* feature names emitted by transform(), in HEAD_SPECS order."""
        return [f'ctx_{n}' for n, _ in HEAD_SPECS
                if self.metrics.get(n, {}).get('passed')]

    def fit(self, E_tr, labels_tr, E_va, labels_va, verbose=True):
        """Train all heads on (E_tr, labels_tr); gate on (E_va, labels_va).
        Rows with NaN labels are dropped per head (structure's mixed
        sentinel). Returns self."""
        self.input_dim = int(np.asarray(E_tr).shape[1])
        for name, kind in HEAD_SPECS:
            ytr = np.asarray(labels_tr[name], np.float32)
            yva = np.asarray(labels_va[name], np.float32)
            m_tr, m_va = ~np.isnan(ytr), ~np.isnan(yva)
            if m_tr.sum() < 500 or m_va.sum() < 100:
                self.metrics[name] = dict(kind=kind, passed=False,
                                          skipped='too few rows',
                                          n_train=int(m_tr.sum()),
                                          n_val=int(m_va.sum()))
                continue
            model = _fit_head(kind, E_tr[m_tr], ytr[m_tr], self.seed,
                              self.n_estimators)
            score = _score_head(kind, model, E_va[m_va], yva[m_va])
            gate = GATE_REG_PEARSON if kind == 'reg' else GATE_CLF_AUC
            passed = bool(score > gate)
            self.models[name] = model
            self.metrics[name] = dict(
                kind=kind, metric='pearson_r' if kind == 'reg' else 'auc',
                score=score, gate=gate, passed=passed,
                n_train=int(m_tr.sum()), n_val=int(m_va.sum()))
            if verbose:
                flag = '✅ PASS' if passed else '❌ FAIL (excluded)'
                print(f"  [ctx_{name:<13}] "
                      f"{self.metrics[name]['metric']}={score:+.3f} "
                      f"gate>{gate}  {flag}")
        return self

    def transform(self, E, include=None):
        """[N, input_dim] input matrix -> [N, n_active] ctx features
        (float32), column order = active_names. `include` overrides the
        gate (list of bare head names) for ablation studies."""
        E = np.asarray(E)
        if self.input_dim is not None and E.shape[1] != self.input_dim:
            raise ValueError(
                f"input width {E.shape[1]} != bundle input_dim "
                f"{self.input_dim} (inputs={self.meta.get('inputs', 'emb')!r})"
                " — enriched bundles need [embedding | ff68 features],"
                " emb-only bundles need the embedding alone.")
        names = (include if include is not None
                 else [n for n, _ in HEAD_SPECS
                       if self.metrics.get(n, {}).get('passed')])
        if not names:
            return np.zeros((len(E), 0), np.float32)
        cols = []
        for name in names:
            kind = dict(HEAD_SPECS)[name]
            model = self.models[name]
            if kind == 'reg':
                cols.append(model.predict(E).astype(np.float32))
            else:
                cols.append(model.predict_proba(E)[:, 1].astype(np.float32))
        return np.column_stack(cols)

    def context_at(self, df, indices, instrument, ctx: int = 128,
                   batch: int = 64, include=None) -> pd.DataFrame:
        """Per-candle market readout — the live-inference entry point.

        For each decision bar index: causal log-close window -> foundation
        embedding (subprocess), hstacked with the 68-feature library row
        for enriched bundles (meta inputs=='emb+ff68') -> named ctx_*
        features. The bot can call this every bar to know the current
        regime/volatility/structure state without ever touching
        embeddings directly.

        df:         DataFrame with datetime/open/high/low/close/volume.
        indices:    decision-bar integer positions into df.
        instrument: symbol for the feature library (e.g. 'ES').

        Emb-only bundles (old format, no meta inputs) use the embedding
        alone — full back-compat via input_dim/meta.

        -> DataFrame indexed by `indices`, columns = active_names.
        """
        from .foundation import embed_bars
        idx = np.asarray(indices)
        E = embed_bars(df['close'].to_numpy(), idx, ctx=ctx, batch=batch)
        if self.meta.get('inputs') == 'emb+ff68':
            X = np.hstack([E, context_features(df, instrument)[idx]])
        else:
            X = E
        names = ([f'ctx_{n}' for n in include] if include is not None
                 else self.active_names)
        return pd.DataFrame(self.transform(X, include=include),
                            index=idx, columns=names)

    def htf_context_at(self, ts, close, indices, htf: str = '1h',
                       ctx: int = 128, batch: int = 64,
                       include=None) -> pd.DataFrame:
        """Per-candle HTF market readout for intraday trading — the same
        heads, fed a higher-timeframe close series, STRICTLY CAUSAL.

        For a decision bar at base-TF time ts[i], the HTF window contains
        only HTF buckets that FULLY ENDED at or before ts[i] (the
        cross-TF lookahead bug class: never read an HTF bar that hasn't
        closed). Conservative by construction: ts is treated as bar-open
        time, so the current (still-forming) HTF bucket is never visible.

        ts:      tz-aware timestamps of the base bars (1m/3m/5m), sorted.
        close:   base-TF closes aligned to ts.
        indices: decision-bar integer positions into ts/close.
        htf:     pandas offset ('1h', '4h', ...). Columns are suffixed,
                 e.g. 'ctx_volatility_1h'.

        -> DataFrame indexed by `indices`; rows with insufficient HTF
        history are NaN.

        NOTE: heads are trained on base-TF windows; HTF windows are a
        transfer application. Bolt embeds log-close shapes scale-free and
        the labels are bar-count-relative, but validate per strategy on
        the honest ruler before relying on it.

        Emb-only bundles only: enriched (emb+ff68) heads need the feature
        library on HTF bars, which is not built yet.
        """
        if self.meta.get('inputs') == 'emb+ff68':
            raise NotImplementedError(
                'HTF readout not yet supported for enriched heads — train '
                'an emb-only bundle for HTF use')
        ts = pd.DatetimeIndex(ts)
        c = np.asarray(close, dtype=np.float64)
        idx = np.asarray(indices, dtype=np.int64)
        # bucket each base bar; HTF close = last base close in bucket
        bucket = ts.floor(htf)
        df = pd.DataFrame({'b': bucket, 'c': c})
        htf_close = df.groupby('b', sort=True)['c'].last()
        bucket_end = (htf_close.index + pd.Timedelta(htf)).asi8
        hc = np.log(htf_close.to_numpy())
        # completed buckets at decision bar i: bucket_end <= ts[i]
        n_done = np.searchsorted(bucket_end, ts.asi8[idx], side='right')
        names = ([f'ctx_{n}' for n in include] if include is not None
                 else self.active_names)
        out = pd.DataFrame(np.nan, index=idx,
                           columns=[f'{n}_{htf}' for n in names])
        ok = n_done >= ctx
        if ok.any():
            windows = np.stack([hc[d - ctx:d] for d in n_done[ok]]).astype(
                np.float32)
            # dedupe identical windows (many base bars share an HTF state)
            uniq, inv = np.unique(windows, axis=0, return_inverse=True)
            from .foundation import embed
            E = embed(uniq, batch=batch)
            out.loc[ok] = self.transform(E, include=include)[inv]
        return out

    def save(self, path):
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(models=self.models, metrics=self.metrics,
                         meta=self.meta, seed=self.seed,
                         n_estimators=self.n_estimators,
                         input_dim=self.input_dim), path)
        return str(path)

    @classmethod
    def load(cls, path):
        import joblib
        blob = joblib.load(path)
        obj = cls(seed=blob.get('seed', 0),
                  n_estimators=blob.get('n_estimators', 400))
        obj.models = blob['models']
        obj.metrics = blob['metrics']
        obj.meta = blob.get('meta', {})
        obj.input_dim = blob.get('input_dim')   # None on pre-2.1 bundles
        return obj

    def describe(self) -> str:
        active = ', '.join(self.active_names) or '(none passed)'
        return (f"ContextHeads[active: {active}] "
                f"meta={json.dumps(self.meta, default=str)[:200]}")
