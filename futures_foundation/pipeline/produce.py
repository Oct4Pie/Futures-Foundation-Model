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


def write_signal_contract(labeler, bundle, output_path):
    """Emit the self-describing signal contract `<base>_signal.json` (v1.0) next
    to the ONNX triplet — the consumer (bot) reads it to VALIDATE it can run the
    model (flip scheme implemented? ctx match? triplet match? feature contract?)
    instead of silently feeding a head the wrong bars/features.

    Bundle supplies the encoder/output/integrity fields; the labeler's optional
    `signal_contract()` (flip/trigger + label) and `feature_names()` (EXACT
    ordered handcraft columns) supply the strategy-specific contract. Labelers
    without those hooks still get a valid minimal sidecar (the required subset).
    """
    import hashlib
    import json
    base = Path(output_path).with_suffix('')
    sc = labeler.signal_contract() if hasattr(labeler, 'signal_contract') else {}
    names = (labeler.feature_names() if hasattr(labeler, 'feature_names')
             else sc.get('handcraft_features'))
    td = (bundle.get('training_metadata') or {}).get('train_date')
    embed_dim = int(bundle.get('embed_dim') or backbone.D_MODEL)
    handcraft_dim = int(bundle['feat_dim']) - embed_dim
    sig_onnx = Path(f"{base}_signal_head.onnx")
    sha = (hashlib.sha256(sig_onnx.read_bytes()).hexdigest()
           if sig_onnx.exists() else None)
    # embed sub-layout the SERVE path must reproduce, in order: ONNX is the
    # chronos_pool block only; the consumer appends loc/scale + return-shape
    # itself (both computed from the SAME ctx window, both pure-numpy).
    rs_on = bool(bundle.get('return_shape', False))
    from .. extractors.chronos.window_features import return_shape_feature_names
    rs_names = return_shape_feature_names() if rs_on else None
    embed_layout = [['chronos_pool', int(backbone.D_MODEL)]]
    if bundle.get('locscale'):
        embed_layout.append(['locscale', 2])
    if rs_on:
        embed_layout.append(['return_shape', len(return_shape_feature_names())])
    contract = {
        'contract_version': '1.0',
        'triplet_id': f"{base.name}@{td}" if td else base.name,
        'role': 'signal_head',
        # 1. trigger / flip (labeler-supplied)
        'flip_scheme': sc.get('flip_scheme'),
        'flip_params': sc.get('flip_params', {}),
        'direction_rule': sc.get('direction_rule'),
        'entry_timing': sc.get('entry_timing', 'next_bar_open'),
        'min_gap_bars': sc.get('min_gap_bars'),
        # 2. features (EXACT ordered handcraft names + integrity dim)
        'handcraft_features': names,
        'handcraft_dim': (len(names) if names is not None else handcraft_dim),
        'feature_lib': sc.get('feature_lib'),
        'nan_policy': {'posinf': 0, 'neginf': 0},
        # 3. encoder / input
        'chronos_ckpt': bundle.get('chronos_ckpt'),
        'ctx_window': (int(bundle['ctx_window'])
                       if bundle.get('ctx_window') else None),
        'pool': bundle.get('pool', 'mean'),
        'locscale': bool(bundle.get('locscale', False)),
        # return-shape: 7 direction/momentum scalars the SERVE path must compute
        # from the ctx window via `return_shape_fn` and append per `embed_layout`.
        'return_shape': rs_on,
        'return_shape_features': rs_names,
        'return_shape_fn': ('futures_foundation.extractors.chronos.window_features'
                            '.return_shape_features' if rs_on else None),
        'embed_layout': embed_layout,
        'embed_dim': embed_dim,
        # 4. output / label
        'n_classes': int(bundle['n_classes']),
        'calibrated': bool(bundle.get('calibrated', False)),
        'label_def': sc.get('label_def'),
        'proba_meaning': sc.get('proba_meaning',
                                'P(trade reaches TP before SL)'),
        # 5. provenance / integrity
        'version': sc.get('version'),
        'train_date': td,
        'train_scope': sc.get('train_scope'),
        'content_sha': sha,
    }
    path = Path(f"{base}_signal.json")
    path.write_text(json.dumps(contract, indent=2))
    # guard: the ordered handcraft names MUST match the bundle's actual width,
    # else the consumer would build a mismatched vector (the silent-bug class).
    ok = names is None or len(names) == handcraft_dim
    return str(path), contract, ok


def group_importance(imp, embed_dim, handcraft_names=None, locscale=False,
                     return_shape=False, d_model=None):
    """Reusable feature-importance review: split a fitted head's importances into
    chronos_embed / return_shape / handcraft groups + name the top fields. The
    embedding block layout mirrors the signal-contract (chronos_pool [+ locscale]
    [+ return_shape]); the handcraft block (labeler features) follows. Returns
    (groups: dict, named: list[(name, importance)] sorted desc)."""
    from ..extractors.chronos.window_features import return_shape_feature_names
    imp = np.asarray(imp, float)
    d_model = int(d_model or backbone.D_MODEL)
    names, o = [], 0
    chronos_w = embed_dim - (2 if locscale else 0) - (len(return_shape_feature_names()) if return_shape else 0)
    names += [f'chronos_{i}' for i in range(chronos_w)]
    if locscale:
        names += ['loc', 'scale']
    if return_shape:
        names += return_shape_feature_names()
    names += list(handcraft_names or [f'feat_{i}' for i in range(len(imp) - embed_dim)])
    names = names[:len(imp)]
    grp = {'chronos_embed': float(imp[:chronos_w].sum())}
    o = chronos_w
    if locscale:
        grp['locscale'] = float(imp[o:o + 2].sum()); o += 2
    if return_shape:
        rs = len(return_shape_feature_names()); grp['return_shape'] = float(imp[o:o + rs].sum()); o += rs
    grp['handcraft'] = float(imp[o:].sum())
    named = sorted(zip(names, imp.tolist()), key=lambda x: -x[1])
    return grp, named


def train(labeler, *, holdout_months: int = 1, seed: int = 0,
          n_estimators: int = 600, max_depth: int = 5,
          head_params: Optional[dict] = None,
          holdout_start_date: Optional[str] = None,
          output_path: Optional[str | Path] = None,
          context_heads_path: Optional[str] = None, emb_mode: str = 'both',
          export_onnx: bool = False, calibrate: bool = False,
          verbose: bool = True) -> dict:
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
    # holdout boundary: explicit DATE (e.g. '2026-01-01' -> clean 2026 OOS)
    # takes precedence over months-from-tail.
    if holdout_start_date is not None:
        holdout_start = pd.Timestamp(holdout_start_date)
        if holdout_start.tzinfo is None and cal_max.tzinfo is not None:
            holdout_start = holdout_start.tz_localize(cal_max.tzinfo)
        has_holdout = holdout_start < cal_max
    else:
        has_holdout = holdout_months > 0
        holdout_start = (cal_max - pd.DateOffset(months=holdout_months)
                         if has_holdout else cal_max + pd.Timedelta('1ns'))

    nc = int(labeler.n_classes)
    binary = (nc == 2)

    if verbose:
        print(f"=== production train | "
              f"labeler={type(labeler).__name__} ===")
        print(f"  data span : {cal_min}  →  {cal_max}")
        print(f"  train     : {cal_min}  →  {holdout_start}  "
              f"(leak-purged at boundary)")
        if has_holdout:
            print(f"  holdout   : {holdout_start}  →  {cal_max}  (unseen OOS)")
        else:
            print(f"  holdout   : (none — training on all data)")
        print(f"  n_classes : {nc} "
              f"({'binary selection' if binary else 'multi-class'})")

    # ---- Stage 1: build training signals (leak-purged at holdout_start) ---
    t0 = time.time()
    Ctr, Ytr, Ktr = labeler.build(
        cal_min, holdout_start,
        test_start=(holdout_start if has_holdout else None))
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
    _hp = dict(n_estimators=n_estimators, max_depth=max_depth)
    if head_params:                                  # e.g. Optuna-tuned config
        _hp.update(head_params)
    signal_head = XGBHead(n_classes=nc, **_hp).fit(Xtr, Ytr, seed=seed)
    if verbose:
        print(f"\n[signal-head] fit XGBHead({nc}-class) on {len(Ytr)} rows  "
              f"({_hp})  ({time.time()-t0:.1f}s)")
    # feature-importance review (reusable): chronos / return-shape / handcraft
    fi_groups, fi_named = group_importance(
        signal_head.feature_importances(), int(Etr.shape[1]),
        handcraft_names=(labeler.feature_names() if hasattr(labeler, 'feature_names') else None),
        locscale=(os.environ.get('CHRONOS_POOL_LOCSCALE') == '1'),
        return_shape=backbone._return_shape_on())
    if verbose:
        print("[feature-importance] " + " ".join(f"{g}={v:.1%}" for g, v in
              sorted(fi_groups.items(), key=lambda x: -x[1])))
        print("  top: " + ", ".join(f"{n}={v:.2%}" for n, v in fi_named[:8]))

    # ---- Stage 4b: calibrate signal-head proba (Platt, OUT-OF-FOLD) ----
    # Maps raw XGB proba -> P(win). Binary heads only. Off by default so the
    # generic pipeline / live bundles are byte-unchanged unless opted in.
    if calibrate and nc == 2:
        t0 = time.time()
        signal_head.fit_calibration(Xtr, Ytr, seed=seed)
        if verbose:
            A, B = signal_head._platt
            print(f"[calibrate] Platt OOF fit: A={A:+.4f} B={B:+.4f}  "
                  f"({time.time()-t0:.1f}s)  "
                  f"⚠ proba scale changed (≈P(win)) — re-tune sizing thresholds")
    elif calibrate and verbose:
        print(f"[calibrate] skipped — head is {nc}-class (binary only)")

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
    if has_holdout:
        if verbose:
            print(f"\n=== HOLDOUT EVAL ({holdout_start.date()} → {cal_max.date()} OOS) ===")
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
            # recall curve: target catch-rate (recall of TREND label) -> proba
            # threshold, from the holdout. The bot sets catch_rate (e.g. 0.85)
            # and filters signals above the looked-up threshold. per-TF (1/3/5/
            # 15min differ) + global; the triple-barrier label IS the trend, so
            # this is calibrated on confirmed catches.
            recall_curve = None
            if proba is not None and (Yte == 1).any():
                tgts = [round(0.50 + 0.05 * k, 2) for k in range(11)]   # 0.50..1.00
                tfs = np.array([str(k[0]).split('@')[1] if '@' in str(k[0])
                                else 'all' for k in Kte])

                def _curve(mask):
                    pos = proba[mask & (Yte == 1)]
                    if not len(pos):
                        return {}
                    return {f'{t:.2f}': float(np.quantile(pos, 1.0 - t))
                            for t in tgts}
                recall_curve = {'global': _curve(np.ones(len(proba), bool))}
                for tf in sorted(set(tfs)):
                    recall_curve[tf] = _curve(tfs == tf)
                if verbose:
                    g = recall_curve['global']
                    print(f"  [recall-curve] catch 80%→thr {g.get('0.80')}  "
                          f"85%→{g.get('0.85')}  90%→{g.get('0.90')}")
            holdout_eval = {
                'n_signals': int(len(Yte)),
                'positive_rate': float(Yte.mean()),
                'recall_curve': recall_curve,
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
        'pool': 'mean',
        'locscale': os.environ.get('CHRONOS_POOL_LOCSCALE') == '1',
        'return_shape': backbone._return_shape_on(),   # +7 return-shape dims baked into embed
        'calibrated': signal_head._platt is not None,
        'platt': signal_head._platt,            # (A, B) or None — for transparency
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
                             if has_holdout else None),
            'head_params': head_params,
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
    contract_path = None
    if export_onnx or os.environ.get('FFM_EXPORT_ONNX') == '1':
        from ..extractors.chronos import onnx_export   # Chronos-specific (encoder export)
        onnx_results = onnx_export.export_bundle_onnx(bundle, output_path,
                                                      verbose=verbose)
        # self-describing signal contract sidecar (next to the ONNX triplet)
        contract_path, contract, ok = write_signal_contract(
            labeler, bundle, output_path)
        if verbose:
            print(f"\n[contract] {('✓' if ok else '⚠ feature-dim MISMATCH')} "
                  f"{contract_path}  "
                  f"(flip={contract['flip_scheme']}, "
                  f"handcraft={contract['handcraft_dim']}, "
                  f"triplet={contract['triplet_id']})")

    if verbose:
        print("\n=== DONE ===")

    return {
        'bundle_path': str(output_path),
        'training_metadata': bundle['training_metadata'],
        'holdout_eval': holdout_eval,
        'size_mb': size_mb,
        'onnx': onnx_results,
        'contract': contract_path,
    }
