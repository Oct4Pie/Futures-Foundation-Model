"""Production training — fit ONE signal head + ONE risk head on the full
corpus for live deployment. The walk-forward evaluator (evaluate.py) is
what certifies the methodology; this trains ONCE on all-but-last-N-months
and saves a single joblib bundle the bot loads.

Strategy-agnostic: any labeler conforming to the same protocol the
walk-forward evaluator uses (calendar / build / features / evaluate)
plugs in unchanged.

The last `holdout_months` months are HELD OUT — the model never sees
them, and they're evaluated as a final sanity check at the end. The
in-sample work has already been validated by 58-fold walk-forward; the
holdout is the last unprejudiced look before the artifact ships.

Bundle contents (joblib):
  signal_head         XGBHead — predict_proba(X) -> [P(skip), P(take)]
  risk_head           XGBRiskHead (binary labelers only) — peak-R̂ regression
  feat_dim            X.shape[1] sanity at inference
  ctx_window          foundation context length (128)
  d_model             foundation embedding dim (256)
  chronos_ckpt        backbone identifier the bot must load identically
  labeler_name        class name (e.g. 'SuperTrendChronos')
  labeler_config      labeler.config_dict() — strategy params (SL/RR/VERT/...)
  n_classes           2 (binary selection) or 3 (direction)
  training_metadata   train/holdout span, n_signals, label_dist, seed, date
  holdout_eval        dict of WR/meanR/PF/sumR on the unseen months
"""
from __future__ import annotations
import datetime as _dt
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from futures_foundation.extractors.chronos import backbone
from . import context_fusion
from .head_xgb import XGBHead, XGBRiskHead


def train(labeler, *, holdout_months: int = 1, seed: int = 0,
          n_estimators: int = 600, max_depth: int = 5,
          output_path: Optional[str | Path] = None,
          context_heads_path: Optional[str] = None, emb_mode: str = 'both',
          export_onnx: bool = False, verbose: bool = True) -> dict:
    """Fit on all signals strictly before `cal_max - holdout_months`
    (with the same leak-purge the walk-forward uses), evaluate on the
    unseen holdout, save bundle. Returns metadata dict.

    Args
    ----
    labeler         a StrategyLabeler (calendar/build/features/evaluate)
    holdout_months  N months at the tail held out as final sanity check.
                    0 = train on every bar (no holdout). Default 1.
    seed            XGBoost random_state for deterministic fits.
    n_estimators    XGB capacity for the SIGNAL head. Production datasets
                    are ~20× per-fold size; the walk-forward default of
                    200 underfits at production scale (probas collapse to
                    the marginal class rate, nothing exceeds 0.5). 600
                    is the empirically-restored sweet spot for ~100k rows.
    max_depth       XGB depth for the signal head. 5 (vs walk-forward 4)
                    pairs with the n_estimators bump.
    output_path     explicit joblib path; default is auto-named in CWD.
    """
    # Stamp the active backbone BEFORE any work — production training
    # bakes the backbone choice into the joblib bundle; surfacing it at
    # run-START prevents the 2026-05-19 wiring gap (fine-tuned ckpt
    # silently ignored because CHRONOS_FT_CKPT was unset).
    backbone.stamp_active_source(context='production training')
    heads = context_fusion.resolve_heads(context_heads_path)
    cal = labeler.calendar()
    cal_min = pd.Timestamp(cal['timestamp'].min())
    cal_max = pd.Timestamp(cal['timestamp'].max())
    if heads is not None:
        # LEAK GUARD: heads trained on pre-cutoff bars; the signal head
        # may not train on them too (double-dip). Trim the span, loudly.
        from futures_foundation.context import HEADS_CUTOFF
        if cal_min < HEADS_CUTOFF:
            print(f"[context-heads] leak guard: training span trimmed "
                  f"{cal_min} -> {HEADS_CUTOFF} (heads saw earlier bars)")
            cal_min = HEADS_CUTOFF
        context_fusion.enforce_cutoff(heads, cal_min, what='production train')
    holdout_start = (cal_max - pd.DateOffset(months=holdout_months)
                     if holdout_months > 0 else cal_max + pd.Timedelta('1ns'))

    nc = int(labeler.n_classes)
    binary = (nc == 2)

    if verbose:
        print(f"=== production train | "
              f"labeler={type(labeler).__name__} ===")
        print(f"  data span : {cal_min}  →  {cal_max}")
        print(f"  train     : {cal_min}  →  {holdout_start}  "
              f"(leak-purged at boundary)")
        if holdout_months > 0:
            print(f"  holdout   : {holdout_start}  →  {cal_max}  "
                  f"({holdout_months}mo unseen)")
        else:
            print(f"  holdout   : (none — training on all data)")
        print(f"  n_classes : {nc} "
              f"({'binary selection' if binary else 'multi-class'})")

    # ---- Stage 1: build training signals (leak-purged at holdout_start) ---
    t0 = time.time()
    Ctr, Ytr, Ktr = labeler.build(
        cal_min, holdout_start,
        test_start=(holdout_start if holdout_months > 0 else None))
    Ytr = np.asarray(Ytr)
    if not len(Ytr):
        raise RuntimeError("no training signals — labeler returned empty")
    if verbose:
        dist = np.bincount(Ytr, minlength=nc).tolist()
        print(f"\n[train signals] {len(Ytr)} signals  "
              f"label_dist={dist}  ({time.time()-t0:.1f}s)")

    # ---- Stage 2: batch-embed ----
    t0 = time.time()
    if verbose:
        print(f"\n[embed] {len(Ctr):,} contexts → foundation backbone ...")
    Etr = backbone.embed(Ctr)
    if verbose:
        print(f"[embed] done. shape={Etr.shape}  ({time.time()-t0:.1f}s)")

    # ---- Stage 3: fuse with ctx heads + hand-crafted features ----
    feats_fn = getattr(labeler, 'features', None)
    Ftr = (np.asarray(feats_fn(Ktr), np.float32) if feats_fn is not None
           else None)
    Xtr = context_fusion.fuse(Etr, Ftr, heads, emb_mode)
    feat_dim = Xtr.shape[1]
    if verbose:
        n_ctx = len(heads.active_names) if heads is not None else 0
        print(f"\n[fuse] X_train: {Xtr.shape}  "
              f"(emb_mode={emb_mode}, ctx {n_ctx}, hand-craft "
              f"{0 if Ftr is None else Ftr.reshape(len(Etr), -1).shape[1]})")

    # ---- Stage 4: fit signal head ----
    t0 = time.time()
    signal_head = XGBHead(n_classes=nc, n_estimators=n_estimators,
                          max_depth=max_depth).fit(Xtr, Ytr, seed=seed)
    if verbose:
        print(f"\n[signal-head] fit XGBHead({nc}-class) on {len(Ytr)} rows  "
              f"(n_est={n_estimators}, depth={max_depth})  "
              f"({time.time()-t0:.1f}s)")

    # ---- Stage 5: fit risk head (binary labelers only) ----
    risk_head = None
    if binary:
        max_rr = np.asarray(
            [k[3] if isinstance(k, tuple) and len(k) > 3 else 0.0
             for k in Ktr], np.float32)
        if max_rr.any():
            t0 = time.time()
            risk_head = XGBRiskHead().fit(Xtr, max_rr, seed=seed)
            if verbose:
                print(f"[risk-head] fit XGBRiskHead on max_rr "
                      f"(mean target {max_rr.mean():.2f}R)  "
                      f"({time.time()-t0:.1f}s)")
        elif verbose:
            print("[risk-head] skipped — max_rr not in labeler keys")

    # ---- Stage 6: holdout evaluation ----
    holdout_eval = None
    if holdout_months > 0:
        if verbose:
            print(f"\n=== HOLDOUT EVAL ({holdout_months}mo unseen) ===")
        Cte, Yte, Kte = labeler.build(
            holdout_start, cal_max + pd.Timedelta('1ns'), test_start=None)
        Yte = np.asarray(Yte)
        if not len(Yte):
            if verbose:
                print("  [holdout] no signals in window — skipped")
        else:
            Ete = backbone.embed(Cte)
            Fte = (np.asarray(feats_fn(Kte), np.float32)
                   if feats_fn is not None else None)
            Xte = context_fusion.fuse(Ete, Fte, heads, emb_mode)
            # threshold sweep mirrors the walk-forward dashboard — argmax
            # `predict()` is wrong here because class imbalance (~70/30)
            # parks the model below 0.50 on most signals; what matters in
            # production is which proba threshold the BOT picks, so we
            # show the sweep and let the user choose.
            proba = (signal_head.predict_proba(Xte)[:, 1].astype(np.float32)
                     if binary else None)
            risk_pred = (risk_head.predict(Xte) if risk_head is not None
                         else None)

            def _row(R):
                if not len(R):
                    return None
                wins = R[R > 0]
                losses = R[R < 0]
                pf = (float(wins.sum() / -losses.sum())
                      if (R < 0).any() and losses.sum() < 0
                      else float('inf'))
                return {'trades': int(len(R)),
                        'wr': float((R > 0).mean()),
                        'meanR': float(R.mean()),
                        'sumR': float(R.sum()),
                        'pf': pf}

            HOLDOUT_THRS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
            fixed_rows, dyn_rows = {}, {}
            if proba is not None:
                for thr in HOLDOUT_THRS:
                    preds = (proba >= thr).astype(int)
                    R_f = labeler.evaluate(Kte, preds)
                    r_f = _row(R_f)
                    if r_f is not None:
                        fixed_rows[thr] = r_f
                    if risk_pred is not None:
                        R_d = labeler.evaluate(Kte, preds,
                                               risk_preds=risk_pred)
                        r_d = _row(R_d)
                        if r_d is not None:
                            dyn_rows[thr] = r_d
            holdout_eval = {
                'n_signals': int(len(Yte)),
                'positive_rate': float(Yte.mean()),
                'proba_stats': ({
                    'mean': float(proba.mean()),
                    'median': float(np.median(proba)),
                    'max': float(proba.max()),
                    'ge_50': int((proba >= 0.50).sum()),
                    'ge_65': int((proba >= 0.65).sum()),
                    'ge_75': int((proba >= 0.75).sum()),
                } if proba is not None else None),
                'fixed_tp_by_thr': fixed_rows,
                'dynamic_tp_by_thr': dyn_rows,
            }
            if verbose:
                pst = holdout_eval['proba_stats']
                print(f"  signals={holdout_eval['n_signals']}  "
                      f"positive_rate={100*holdout_eval['positive_rate']:.1f}%")
                if pst:
                    print(f"  proba: mean={pst['mean']:.3f}  "
                          f"max={pst['max']:.3f}  ≥0.50:{pst['ge_50']}  "
                          f"≥0.65:{pst['ge_65']}  ≥0.75:{pst['ge_75']}")
                if fixed_rows:
                    print(f"\n  🎯 FIXED-TP sweep")
                    print(f"     Thr   Trades    WR    meanR    PF")
                    for thr, r in fixed_rows.items():
                        pf_s = (f"{r['pf']:.2f}" if r['pf'] < 999 else "inf")
                        print(f"     {thr:>4.2f}  {r['trades']:>6}  "
                              f"{100*r['wr']:>5.1f}%  "
                              f"{r['meanR']:+5.2f}R  {pf_s}")
                if dyn_rows:
                    print(f"\n  💎 DYNAMIC-TP sweep")
                    print(f"     Thr   Trades    WR    meanR    PF")
                    for thr, r in dyn_rows.items():
                        pf_s = (f"{r['pf']:.2f}" if r['pf'] < 999 else "inf")
                        print(f"     {thr:>4.2f}  {r['trades']:>6}  "
                              f"{100*r['wr']:>5.1f}%  "
                              f"{r['meanR']:+5.2f}R  {pf_s}")
                if not fixed_rows and not dyn_rows:
                    print("  [holdout] no signals passed any threshold — "
                          "the model is severely underfit; bump capacity.")

    # ---- Stage 7: save bundle ----
    if output_path is None:
        date = _dt.date.today().strftime('%Y%m%d')
        name = type(labeler).__name__.lower()
        output_path = Path(f"chronos_{name}_production_{date}.joblib")
    output_path = Path(output_path)

    ckpt = os.environ.get('CHRONOS_FT_CKPT') or 'amazon/chronos-bolt-tiny'

    import joblib
    bundle = {
        'signal_head': signal_head,
        'risk_head': risk_head,
        'feat_dim': feat_dim,
        'embed_dim': int(Etr.shape[1]),
        'ctx_window': len(Ctr[0]) if len(Ctr) else None,
        'd_model': backbone.D_MODEL,
        'chronos_ckpt': ckpt,
        'labeler_name': type(labeler).__name__,
        'labeler_config': (labeler.config_dict()
                           if hasattr(labeler, 'config_dict') else {}),
        'n_classes': nc,
        'context_heads': ({
            'path': str(context_heads_path
                        or os.environ.get(context_fusion.ENV_VAR)),
            'emb_mode': emb_mode,
            'active_names': heads.active_names,
            'metrics': heads.metrics,
            'meta': heads.meta,
        } if heads is not None else None),
        'training_metadata': {
            'train_span': (str(cal_min), str(holdout_start)),
            'holdout_span': ((str(holdout_start), str(cal_max))
                             if holdout_months > 0 else None),
            'n_train_signals': int(len(Ytr)),
            'label_dist': np.bincount(Ytr, minlength=nc).tolist(),
            'train_date': _dt.date.today().isoformat(),
            'seed': int(seed),
        },
        'holdout_eval': holdout_eval,
    }
    joblib.dump(bundle, output_path)
    size_mb = output_path.stat().st_size / 1e6

    if verbose:
        print(f"\n[save] {output_path}  ({size_mb:.1f} MB)")

    # ---- Stage 8 (optional): export ONNX + parity-check vs the joblib ----
    # Any produce/pipeline script gets ONNX via export_onnx=True OR the
    # FFM_EXPORT_ONNX=1 env (universal switch, no per-script wiring needed).
    onnx_results = None
    if export_onnx or os.environ.get('FFM_EXPORT_ONNX') == '1':
        from . import onnx_export
        onnx_results = onnx_export.export_bundle_onnx(bundle, output_path,
                                                      verbose=verbose)

    if verbose:
        print("\n=== DONE ===")

    return {
        'bundle_path': str(output_path),
        'training_metadata': bundle['training_metadata'],
        'holdout_eval': holdout_eval,
        'size_mb': size_mb,
        'onnx': onnx_results,
    }
