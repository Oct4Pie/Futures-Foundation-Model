"""Honest-ruler evaluation — strategy- and head-agnostic.

Frozen Chronos embedding (+ optional strategy features) -> a pluggable
head, scored leak-free over walk-forward x {REAL, SHUFFLE, RANDOM} x seeds.
A number is believed ONLY if REAL clearly beats SHUFFLE and RANDOM on
realized R (cost lives inside the strategy's evaluate()). No strategy or
head specifics here — only the StrategyLabeler protocol and a head with
fit(X,y,seed)/predict(X).
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from .data import walk_forward_folds
from futures_foundation import foundation as backbone
from . import context_fusion
from .head_xgb import XGBHead, XGBRiskHead

# ===========================================================================
# Pre-registered PASS criteria. The auto-verdict at the end of run() prints
# PASS/FAIL based ONLY on these numbers — no human interpretation, no goal-
# post moving. Change these constants deliberately (and ideally pre-register
# the change before the next run, not after seeing results).
# ===========================================================================
PASS_TARGET_WR        = 0.50   # WR at the chosen dynamic-TP threshold
                               # (revised down from 0.60 after seeing that
                               # AvgR is the more informative combine-pass
                               # metric under dynamic-TP; AvgR target stays
                               # at 1.5R, so the combine math is preserved
                               # even if hit-rate is lower)
PASS_TARGET_MEAN_R    = 1.50   # AvgR per trade at dynamic-TP (combine math)
PASS_MIN_TRADES_AGG   = 100    # min total trades (sig + RL bandwidth)
PASS_LIFT_MARGIN_R    = 0.10   # REAL must beat each control by this margin
PASS_MAX_RISK_MAE_R   = 2.50   # risk-head useful if MAE <= this (R-units)
PASS_MIN_RISK_CORR    = 0.10   # Pearson r(predicted, actual max_rr) >= this
PASS_MIN_TICKERS_LIFT = 2      # >= this many tickers above NAIVE on meanR

# ── Auto-regularization (overfit DETECTION + REMEDIATION) ──────────────────
# Overfit signal = in-sample (TRAIN) meanR exceeds VALIDATION meanR by more than
# OVERFIT_GAP_R → the head memorized train. When that fires, re-fit a
# progressively more-regularized head and keep the one with the best VALIDATION
# meanR (selection NEVER touches test). The XGBoost analog of the original FFM
# FoldHealthMonitor's "reduce epochs / increase regularisation" remediation —
# but as a CLOSED LOOP (detect → adjust → re-check), not just a printed suggestion.
# Ladder is most→least capable (XGBHead kwargs); default head is rung 0.
OVERFIT_GAP_R = 0.30
# Generalization gate: VAL→TEST meanR gap above this = the operating point does
# NOT generalize (edge held on validation but decayed on test = fake/fragile
# edge). A HARD verdict failure — we want real OOS generalization, not a number
# that only looks good on the data the threshold was tuned on.
GEN_GAP_TOL = 0.30
REG_LADDER = [
    dict(max_depth=3, min_child_weight=20, reg_lambda=5.0),
    dict(max_depth=3, min_child_weight=50, reg_lambda=10.0, subsample=0.6),
    dict(max_depth=2, min_child_weight=80, reg_lambda=15.0, n_estimators=150),
]


def _overfit_trigger(train_meanR, val_meanR):
    """True if the head overfit TRAIN relative to VALIDATION by more than
    OVERFIT_GAP_R → the auto-regularize remediation should fire."""
    return (train_meanR - val_meanR) > OVERFIT_GAP_R


def _best_rung(default_val_meanR, rung_val_means):
    """Pick the regularization rung with the best VALIDATION meanR, but only if
    it beats the default head's validation meanR; otherwise None (keep default).
    rung_val_means: iterable of (cfg, val_meanR). Selection never sees TEST."""
    best_vm, best_cfg = default_val_meanR, None
    for cfg, vm in rung_val_means:
        if vm > best_vm:
            best_vm, best_cfg = vm, cfg
    return best_cfg


def _operating_verdict(v_at, t_at, gap, real_m, shuf_m, rand_m, naive_m,
                       thr=None, records=None):
    """Build the pre-registered operating-point verdict (pure; unit-tested).
    Returns (checks, all_pass, verdict_dict). checks: list of (ok, msg). The
    VAL→TEST gap is a HARD generalization criterion — fake edge fails here."""
    checks = [
        (t_at is not None and t_at['wr'] >= PASS_TARGET_WR
         and t_at['n'] >= PASS_MIN_TRADES_AGG,
         f"TEST@val-thr: WR≥{int(PASS_TARGET_WR*100)}% + trades≥"
         f"{PASS_MIN_TRADES_AGG}"
         + (f" (WR={100*t_at['wr']:.1f}%, n={t_at['n']})" if t_at else "")),
        (real_m - shuf_m >= PASS_LIFT_MARGIN_R,
         f"edge REAL−SHUFFLE ≥{PASS_LIFT_MARGIN_R}R (got {real_m-shuf_m:+.2f}R)"),
        (real_m - rand_m >= PASS_LIFT_MARGIN_R,
         f"edge REAL−RANDOM ≥{PASS_LIFT_MARGIN_R}R (got {real_m-rand_m:+.2f}R)"),
        (real_m - naive_m >= PASS_LIFT_MARGIN_R,
         f"edge REAL−NAIVE ≥{PASS_LIFT_MARGIN_R}R (got {real_m-naive_m:+.2f}R)"),
        (gap is not None and gap <= GEN_GAP_TOL,
         f"GENERALIZES: VAL→TEST gap ≤{GEN_GAP_TOL}R"
         + (f" (got {gap:+.2f}R)" if gap is not None else " (no val/test)")),
    ]
    all_pass = all(ok for ok, _ in checks)
    verdict = dict(
        all_pass=all_pass,
        generalizes=(gap is not None and gap <= GEN_GAP_TOL),
        gap=gap, thr=thr,
        test_meanR=(t_at['meanR'] if t_at else None),
        test_wr=(t_at['wr'] if t_at else None),
        test_n=(t_at['n'] if t_at else 0),
        val_meanR=(v_at['meanR'] if v_at else None),
        edge_shuffle=real_m - shuf_m, edge_random=real_m - rand_m,
        edge_naive=real_m - naive_m, records=records)
    return checks, all_pass, verdict


def _stats(R):
    R = np.asarray(R, float)
    if not len(R):
        return "trades=0"
    return (f"trades={len(R)} win={np.mean(R > 0):.1%} "
            f"sumR={R.sum():+.1f} meanR={R.mean():+.3f}")


def _featurize(labeler, contexts, keys):
    """Frozen Chronos embedding, fused with the strategy's own features if
    the labeler exposes an optional features(keys) hook."""
    X = backbone.embed(contexts)
    feats = getattr(labeler, 'features', None)
    if feats is not None:
        extra = np.asarray(feats(keys), np.float32)
        if extra.size:
            X = np.hstack([X, extra.reshape(len(X), -1)])
    return X


def _taken_tickers(keys, preds):
    """Per-trade ticker list aligned with the labeler.evaluate R array
    (convention: preds==0 = skip; else = a trade; key[0] = ticker by
    convention in concrete labelers). Returns [] if keys aren't tuples
    (e.g., synthetic test labelers using int keys) — gracefully skips
    the per-ticker breakdown for those."""
    taken = [keys[i] for i in np.flatnonzero(np.asarray(preds) != 0)]
    return [k[0] for k in taken if isinstance(k, tuple) and len(k)]


def _agg_stats(name, R, tks=None):
    """Aggregate stats line; optional per-ticker breakdown."""
    R = np.asarray(R, float)
    if not len(R):
        print(f"  [{name:<7}] trades=0")
        return
    print(f"  [{name:<7}] {_stats(R)}")
    if tks is None:
        return
    by_tk = defaultdict(list)
    for r, t in zip(R, tks):
        by_tk[t].append(r)
    for tk in sorted(by_tk):
        Rt = np.asarray(by_tk[tk])
        print(f"            {tk:<5} {_stats(Rt)}")


def run(labeler, head_factory=None, seeds=(0, 1, 2), train_m=3, val_m=1, test_m=1,
        max_folds=None, context_heads_path=None, emb_mode='both',
        min_train_start=None, auto_regularize=True, return_verdict=False):
    """labeler: a StrategyLabeler. head_factory: nc -> head (default
    XGBHead). max_folds=None -> sweep every available OOS month-pair
    (XGBoost-pipeline convention). Prints REAL/SHUFFLE/RANDOM per
    fold-seed, an aggregate stats block + per-ticker breakdown for
    REAL, AND (binary labelers only) a WR-by-confidence-threshold sweep
    — the entry-signal calibration test for downstream RL/account
    management. Returns the per-(fold,seed) records.

    Context-heads fusion (additive, default-off): `context_heads_path`
    (or $CONTEXT_HEADS_BUNDLE) loads a frozen ContextHeads bundle whose
    ctx_* outputs are fused per `emb_mode` ('both'|'emb'|'heads' — the
    pre-registered A/B arms). With heads active, folds whose train window
    starts before HEADS_CUTOFF are EXCLUDED (leak guard: the heads
    trained on those bars). `min_train_start` filters folds in any mode —
    set it identically across A/B arms so all arms see the same folds."""
    # Stamp the active backbone BEFORE any work — surfaces a wiring
    # gap (fine-tuned ckpt sitting unused, CHRONOS_FT_CKPT not exported)
    # at run-START, not after we've burnt 15 min on the wrong backbone.
    backbone.stamp_active_source(context='walk-forward eval')
    heads = context_fusion.resolve_heads(context_heads_path)
    if heads is not None and min_train_start is None:
        from futures_foundation.context import HEADS_CUTOFF
        min_train_start = HEADS_CUTOFF
        print(f"[context-heads] leak guard: folds restricted to train "
              f"windows starting >= {min_train_start.date()}")
    _default_head = head_factory is None      # auto-regularize only the default head
    head_factory = head_factory or (lambda nc: XGBHead(nc))
    binary = labeler.n_classes == 2
    out = []
    verdict = None              # set in the binary operating-point block below
    pool = {'REAL': ([], []), 'SHUFFLE': ([], []), 'RANDOM': ([], [])}
    proba_records = []
    val_records = []                   # (keys, proba) on VALIDATION — for thr select
    # Phase 1 — collect all (Ctr, Ytr, Ktr, Cte, Yte, Kte) across folds
    # WITHOUT calling backbone.embed. Pure pandas/numpy; fast.
    fold_data = []
    n_excluded = 0
    for fold, tr, val, te in walk_forward_folds(labeler.calendar(), train_m,
                                                val_m, test_m):
        if max_folds is not None and len(fold_data) >= max_folds:
            break
        if min_train_start is not None \
                and tr['timestamp'].min() < pd.Timestamp(min_train_start):
            n_excluded += 1
            continue
        if heads is not None:
            context_fusion.enforce_cutoff(
                heads, tr['timestamp'].min(), what=f'fold {fold} train')
        val0 = val['timestamp'].min()
        te0 = te['timestamp'].min()
        # TRAIN purged so its outcome window can't reach VAL; VAL purged so its
        # outcome window can't reach TEST; TEST unpurged (outcomes resolve forward,
        # never used for training). Strict train < val < test separation.
        Ctr, Ytr, Ktr = labeler.build(tr['timestamp'].min(),
                                       tr['timestamp'].max(), val0)
        Cval, Yval, Kval = labeler.build(val0, val['timestamp'].max(), te0)
        Cte, Yte, Kte = labeler.build(
            te['timestamp'].min(),
            te['timestamp'].max() + np.timedelta64(1, 'ns'), None)
        if len(Ytr) < 50 or len(Cte) < 50 or len(Cval) < 10:
            continue
        fold_data.append(dict(fold=fold, Ctr=Ctr, Ytr=np.asarray(Ytr), Ktr=Ktr,
                              Cval=Cval, Yval=np.asarray(Yval), Kval=Kval,
                              Cte=Cte, Yte=np.asarray(Yte), Kte=Kte))
    if n_excluded:
        print(f"[folds] {n_excluded} fold(s) excluded by min_train_start="
              f"{pd.Timestamp(min_train_start).date()}")
    if not fold_data:
        print("No productive folds — nothing to evaluate.")
        return out
    # Phase 2 — ONE batched Chronos embed across all productive folds.
    # Replaces ~2*N_folds subprocess loads with 1. Major speedup on long
    # walk-forwards; numpy/torch parallelize the single big inference call.
    flat_contexts = []
    for d in fold_data:
        flat_contexts.extend(d['Ctr']); flat_contexts.extend(d['Cval'])
        flat_contexts.extend(d['Cte'])
    print(f"\n[batch-embed] {len(flat_contexts):,} contexts across "
          f"{len(fold_data)} folds in ONE Chronos call...")
    flat_embed = backbone.embed(flat_contexts)
    # slice back per fold + fuse with labeler features
    o = 0
    feats_fn = getattr(labeler, 'features', None)

    def _fuse(keys, E):
        extra = (np.asarray(feats_fn(keys), np.float32)
                 if feats_fn is not None else None)
        return context_fusion.fuse(E, extra, heads, emb_mode)

    for d in fold_data:
        ntr, nva, nte = len(d['Ctr']), len(d['Cval']), len(d['Cte'])
        Etr = flat_embed[o:o + ntr]; o += ntr
        Eva = flat_embed[o:o + nva]; o += nva
        Ete = flat_embed[o:o + nte]; o += nte
        d['Xtr'] = _fuse(d['Ktr'], Etr)
        d['Xval'] = _fuse(d['Kval'], Eva)
        d['Xte'] = _fuse(d['Kte'], Ete)
    mode_note = (f" (emb_mode={emb_mode}, ctx heads: "
                 f"{len(heads.active_names)})" if heads is not None else "")
    print(f"[batch-embed] done. feat_dim={fold_data[0]['Xtr'].shape[1]}"
          f"{mode_note}\n")

    # Phase 3 — per-(fold, seed) XGBoost work, parallelized via threads.
    # XGBoost releases the GIL in its C tree-builder + numpy ops, so a
    # ThreadPoolExecutor extracts multi-core speedup without pickling the
    # labeler's large bar arrays (which a ProcessPool would force).
    # Risk-head training target: max_rr_realized embedded in keys (4th
    # element). Falls back to 0 if labeler keys are 3-tuples (legacy).
    def _max_rr_targets(keys):
        return np.asarray(
            [k[3] if len(k) > 3 else 0.0 for k in keys], np.float32)

    def _work(d, seed):
        Ytr, Xtr, Xte, Kte = d['Ytr'], d['Xtr'], d['Xte'], d['Kte']
        Yte_np = d['Yte']
        nc = labeler.n_classes
        # signal head — binary take/skip. FIT ON TRAIN ONLY.
        head = head_factory(nc).fit(Xtr, Ytr, seed)

        # ---- overfit DETECT + auto-REGULARIZE (REAL head; default head only) --
        # If the head memorized train (train meanR ≫ val meanR), step down the
        # regularization ladder and keep the rung with the best VALIDATION meanR.
        # Selection uses TRAIN + VAL only — TEST is never consulted.
        remediated = None
        if binary and auto_regularize and _default_head:
            tr_R = labeler.evaluate(d['Ktr'], head.predict(Xtr))
            vl_R = labeler.evaluate(d['Kval'], head.predict(d['Xval']))
            tr_m = float(tr_R.mean()) if len(tr_R) else 0.0
            vl_m = float(vl_R.mean()) if len(vl_R) else 0.0
            if _overfit_trigger(tr_m, vl_m):            # overfit to train → remediate
                cands = []                               # (cfg, val_meanR, head)
                for cfg in REG_LADDER:
                    h = XGBHead(nc, **cfg).fit(Xtr, Ytr, seed)
                    vR = labeler.evaluate(d['Kval'], h.predict(d['Xval']))
                    vm = float(vR.mean()) if len(vR) else -1e9
                    cands.append((cfg, vm, h))
                best_cfg = _best_rung(vl_m, [(c, vm) for c, vm, _ in cands])
                if best_cfg is not None:                 # a rung beat the default
                    head = next(h for c, vm, h in cands if c is best_cfg)
                    remediated = best_cfg

        p_real = head.predict(Xte)
        R = labeler.evaluate(Kte, p_real)               # fixed-TP baseline (final head)
        proba = (head.predict_proba(Xte)[:, 1].astype(np.float32)
                 if binary else None)
        # VALIDATION proba — for threshold SELECTION (never on test). Same head.
        proba_val = (head.predict_proba(d['Xval'])[:, 1].astype(np.float32)
                     if binary else None)
        # risk head — predicts max_rr_realized per signal (binary only;
        # 3-class direction strategies don't have a single risk regression
        # target). Trained alongside on the same Xtr.
        risk_pred = None
        if binary:
            mrr_tr = _max_rr_targets(d['Ktr'])
            risk_head = XGBRiskHead().fit(Xtr, mrr_tr, seed)
            risk_pred = risk_head.predict(Xte)
        # shuffle + random controls (unchanged)
        ysh = Ytr.copy()
        np.random.default_rng(seed + 1).shuffle(ysh)
        p_sh = head_factory(nc).fit(Xtr, ysh, seed).predict(Xte)
        Rs = labeler.evaluate(Kte, p_sh)
        p_rd = np.random.default_rng(seed + 2).integers(0, nc, len(Kte))
        Rr = labeler.evaluate(Kte, p_rd)
        # NAIVE baseline (binary only): take EVERY signal. Computed at
        # dynamic-TP when risk_preds available — that's the apples-to-
        # apples comparison the verdict checks REAL against.
        Rn = None
        if binary:
            all_take = np.ones(len(Kte), int)
            Rn = labeler.evaluate(Kte, all_take, risk_preds=risk_pred)
        return dict(fold=d['fold'], seed=seed, R=R, Rs=Rs, Rr=Rr, Rn=Rn,
                    p_real=p_real, p_sh=p_sh, p_rd=p_rd,
                    proba=proba, risk_pred=risk_pred,
                    Yte=Yte_np, Kte=Kte, remediated=remediated,
                    proba_val=proba_val, Kval=d['Kval'], Yval=d['Yval'])

    import concurrent.futures as cf
    import os
    n_workers = min(os.cpu_count() or 4, 8)
    print(f"[parallel] running {len(fold_data) * len(seeds)} (fold, seed) "
          f"work units across {n_workers} threads...")
    work_units = [(d, s) for d in fold_data for s in seeds]
    results = [None] * len(work_units)
    with cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_work, d, s): i
                   for i, (d, s) in enumerate(work_units)}
        for fut in cf.as_completed(futures):
            results[futures[fut]] = fut.result()

    # Print in fold/seed order + aggregate.
    by_fold = defaultdict(list)
    for r in results:
        by_fold[r['fold']].append(r)
    for fold in sorted(by_fold):
        d = next(x for x in fold_data if x['fold'] == fold)
        print(f"== fold {fold} | ntr={len(d['Ytr'])} nte={len(d['Kte'])} "
              f"| feat_dim={d['Xtr'].shape[1]} | classes={labeler.n_classes} ==")
        for r in sorted(by_fold[fold], key=lambda x: x['seed']):
            print(f"  seed {r['seed']}  [REAL   ] {_stats(r['R'])}")
            print(f"          [SHUFFLE] {_stats(r['Rs'])}")
            print(f"          [RANDOM ] {_stats(r['Rr'])}")
            out.append({'fold': r['fold'], 'seed': r['seed'],
                        'REAL': r['R'], 'SHUFFLE': r['Rs'],
                        'RANDOM': r['Rr']})
            pool['REAL'][0].append(r['R'])
            pool['REAL'][1].extend(_taken_tickers(r['Kte'], r['p_real']))
            pool['SHUFFLE'][0].append(r['Rs'])
            pool['SHUFFLE'][1].extend(_taken_tickers(r['Kte'], r['p_sh']))
            pool['RANDOM'][0].append(r['Rr'])
            pool['RANDOM'][1].extend(_taken_tickers(r['Kte'], r['p_rd']))
            if r.get('Rn') is not None:
                pool.setdefault('NAIVE', ([], []))
                pool['NAIVE'][0].append(r['Rn'])
                pool['NAIVE'][1].extend(
                    [k[0] for k in r['Kte']
                     if isinstance(k, tuple) and len(k)])
            if binary and r['proba'] is not None:
                proba_records.append(
                    (r['Kte'], r['proba'], r['Yte'].astype(np.int8),
                     r['risk_pred']))
                if r.get('proba_val') is not None:
                    val_records.append((r['Kval'], r['proba_val']))
    print(f"\n== AGGREGATE | folds={len({r['fold'] for r in out})} "
          f"seeds={len(seeds)} | pooled across all (fold,seed) ==")
    for name in ('REAL', 'SHUFFLE', 'RANDOM'):
        R = (np.concatenate(pool[name][0]) if pool[name][0]
             else np.array([], float))
        _agg_stats(name, R, tks=pool[name][1] if name == 'REAL' else None)
    print("\n-> Believe a result only if REAL clearly beats SHUFFLE AND "
          "RANDOM on sumR/meanR (cost in evaluate()), AND the per-ticker "
          "lift over the labeler's NAIVE baseline is consistent.")

    # ---- Auto-regularization summary (overfit detect + fix) --------------
    if binary and auto_regularize and _default_head:
        n_units = len(results)
        rem = [r['remediated'] for r in results if r.get('remediated')]
        if rem:
            from collections import Counter
            tags = Counter(str(t) for t in rem)
            print(f"\n🛠  AUTO-REGULARIZE: {len(rem)}/{n_units} (fold,seed) units "
                  f"overfit train→val by >{OVERFIT_GAP_R}R → re-fit with a more "
                  f"regularized head (selected on validation):")
            for tag, cnt in tags.most_common():
                print(f"     {cnt}× → {tag}")
        else:
            print(f"\n🛠  AUTO-REGULARIZE: 0/{n_units} units triggered "
                  f"(no head overfit train→val by >{OVERFIT_GAP_R}R) — none needed.")

    # ---- Robustness block (additive; never breaks the run) ---------------
    try:
        _print_robustness(out, pool)
    except Exception as e:                              # pragma: no cover
        print(f"  [robustness] skipped: {e}")

    # ---- Confidence-threshold dashboard (binary labelers only) -----------
    # FFM/XGBoost-pipeline-style: per threshold, report trades / wins / WR
    # (= Prec) / Recall / PF / vs-NAIVE / verdict. The entry-signal
    # calibration test for downstream RL/account managers — pick the
    # threshold where WR is high enough AND trade frequency is meaningful.
    if binary and proba_records:
        all_y = np.concatenate([y for _, _, y, _ in proba_records])
        naive_wr = float(np.mean(all_y == 1)) if len(all_y) else 0.0
        total_pos = int((all_y == 1).sum())
        # risk-head MAE on confirmed signals (only ones we'd take) — the
        # quality of the dynamic-TP prediction.
        rp_have_risk = any(r is not None for _, _, _, r in proba_records)
        if rp_have_risk:
            r_pred_all = np.concatenate(
                [r for _, _, _, r in proba_records if r is not None])
            r_true_all = np.concatenate(
                [np.asarray([k[3] if len(k) > 3 else 0.0 for k in keys],
                            np.float32)
                 for keys, _, _, _ in proba_records])
            mae = float(np.mean(np.abs(r_pred_all - r_true_all)))
            corr = float(np.corrcoef(r_pred_all, r_true_all)[0, 1]) \
                if r_pred_all.std() > 0 and r_true_all.std() > 0 else 0.0
            print(f"\n📐 RISK-HEAD (predicts max_rr_realized per signal)")
            print(f"   MAE: {mae:.2f}R  |  Pearson r(pred, actual): "
                  f"{corr:+.3f}  |  pred mean: {r_pred_all.mean():.2f}R  |  "
                  f"actual mean: {r_true_all.mean():.2f}R")

        # ---- Fixed-TP confidence dashboard (signal-head only) ------------
        print(f"\n🎯 CONFIDENCE THRESHOLDS — REAL @ fixed-TP (pooled)")
        print(f"   {'Thresh':>6}  {'Trades':>6}  {'Wins':>5}  {'WR':>6}  "
              f"{'Recall':>6}  {'PF':>6}  {'vs NAIVE':>8}  Verdict")
        print(f"   {'-' * 66}")
        for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            R_all, tp_total, fp_total = [], 0, 0
            for keys, proba, yte, _r in proba_records:
                preds_t = (proba >= thr).astype(int)
                R_t = labeler.evaluate(keys, preds_t)
                if len(R_t):
                    R_all.append(R_t)
                tp_total += int(((preds_t == 1) & (yte == 1)).sum())
                fp_total += int(((preds_t == 1) & (yte == 0)).sum())
            trades = tp_total + fp_total
            if trades == 0:
                print(f"   {thr:>5.2f}  {0:>6}     —     —       —"
                      f"      —      —      ⚠️ no trades")
                continue
            Rc = np.concatenate(R_all)
            wr = tp_total / trades
            recall = (tp_total / total_pos) if total_pos else 0.0
            wins_R = Rc[Rc > 0]; losses_R = Rc[Rc < 0]
            pf = (wins_R.sum() / abs(losses_R.sum())
                  if len(losses_R) and losses_R.sum() < 0 else float('inf'))
            pf_s = f"{pf:>6.2f}" if pf < 999 else "   inf"
            delta = wr - naive_wr
            verdict = ('✅ EDGE' if wr >= 0.60 and trades >= 20
                       else ('⚠️ THIN' if trades < 20
                             else ('🟡 LIFT' if delta > 0.05 else '❌')))
            print(f"   {thr:>5.2f}  {trades:>6}  {tp_total:>5}  "
                  f"{100 * wr:>5.1f}%  {100 * recall:>5.1f}%  {pf_s}  "
                  f"{100 * delta:>+7.1f}%  {verdict}")

        # ---- Dynamic-TP confidence dashboard (signal + risk head) --------
        # The deployment-equivalent metric: per signal, TP is set from
        # the risk-head prediction. This is what the bot will earn.
        dyn_rows = []
        if rp_have_risk:
            print(f"\n💎 CONFIDENCE THRESHOLDS — REAL @ dynamic-TP "
                  f"(TP = clip(0.8 × R̂, 1.5, 8.0))")
            print(f"   {'Thresh':>6}  {'Trades':>6}  {'WR':>6}  "
                  f"{'AvgR':>6}  {'sumR':>7}  {'PF':>6}  Verdict")
            print(f"   {'-' * 56}")
            for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
                R_all = []
                for keys, proba, _yte, risk in proba_records:
                    if risk is None:
                        continue
                    preds_t = (proba >= thr).astype(int)
                    R_t = labeler.evaluate(keys, preds_t, risk_preds=risk)
                    if len(R_t):
                        R_all.append(R_t)
                if not R_all:
                    print(f"   {thr:>5.2f}  {0:>6}     —      —       —"
                          f"      —     ⚠️ no trades")
                    continue
                Rc = np.concatenate(R_all)
                trades = len(Rc)
                wr = float(np.mean(Rc > 0))
                meanR = float(Rc.mean())
                wins_R = Rc[Rc > 0]; losses_R = Rc[Rc < 0]
                pf = (wins_R.sum() / abs(losses_R.sum())
                      if len(losses_R) and losses_R.sum() < 0
                      else float('inf'))
                pf_s = f"{pf:>6.2f}" if pf < 999 else "   inf"
                row_verdict = ('✅ EDGE' if meanR >= 1.5 and trades >= 20
                               else ('⚠️ THIN' if trades < 20
                                     else ('🟡 LIFT' if meanR > 0.5 else '❌')))
                print(f"   {thr:>5.2f}  {trades:>6}  {100*wr:>5.1f}%  "
                      f"{meanR:+5.2f}R  {Rc.sum():+7.1f}  {pf_s}  {row_verdict}")
                dyn_rows.append((thr, trades, wr, meanR, Rc.sum(), pf))

        print(f"\n   NAIVE-WR baseline (take every signal): "
              f"{100 * naive_wr:.1f}%")
        print(f"   (the threshold tables above are a TEST DIAGNOSTIC — the "
              f"operating point below is selected on VALIDATION, not test.)")

        # ---- OPERATING POINT: threshold SELECTED ON VALIDATION, reported on
        # TEST. This is the standard train/validate/test discipline — test is
        # never used to choose the threshold, so the reported number has no
        # threshold-on-test bias. The fixed-TP sweep above is diagnostic only.
        def _fixedtp(records, thr):
            R_all = []
            for keys, proba in records:
                preds = (proba >= thr).astype(int)
                Rt = labeler.evaluate(keys, preds)
                if len(Rt):
                    R_all.append(Rt)
            if not R_all:
                return None
            Rc = np.concatenate(R_all)
            w = Rc[Rc > 0].sum(); loss = abs(Rc[Rc < 0].sum())
            return dict(n=len(Rc), wr=float(np.mean(Rc > 0)),
                        meanR=float(Rc.mean()), sumR=float(Rc.sum()),
                        pf=(w / loss if loss > 0 else float('inf')))

        test_recs = [(k, p) for k, p, _, _ in proba_records]
        grid = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
        thr_star, best = 0.50, None                         # selected on VAL
        for thr in grid:
            mv = _fixedtp(val_records, thr)
            if mv and mv['n'] >= 30 and (best is None or mv['meanR'] > best):
                best, thr_star = mv['meanR'], thr
        v_at = _fixedtp(val_records, thr_star)
        t_at = _fixedtp(test_recs, thr_star)
        bar = "═" * 60
        print(f"\n{bar}\n🚦 OPERATING POINT — thr {thr_star:.2f} SELECTED ON "
              f"VALIDATION → reported on TEST\n{bar}")
        if v_at:
            pfv = f"{v_at['pf']:.2f}" if v_at['pf'] < 999 else "inf"
            print(f"   VAL  @{thr_star:.2f}:  n={v_at['n']:5d}  "
                  f"WR={100*v_at['wr']:.1f}%  meanR={v_at['meanR']:+.3f}  PF={pfv}")
        if t_at:
            pft = f"{t_at['pf']:.2f}" if t_at['pf'] < 999 else "inf"
            print(f"   TEST @{thr_star:.2f}:  n={t_at['n']:5d}  "
                  f"WR={100*t_at['wr']:.1f}%  meanR={t_at['meanR']:+.3f}  "
                  f"PF={pft}   ← honest OOS")
        gap = None
        if v_at and t_at:
            gap = v_at['meanR'] - t_at['meanR']
            print(f"   VAL→TEST gap: {gap:+.3f}R  "
                  f"{'⚠️ OVERFIT to val (does NOT generalize)' if gap > GEN_GAP_TOL else '✓ generalizes'}")

        # pre-registered verdict on the VAL-locked TEST operating point + the
        # threshold-free control margins (edge). risk-head/dynamic-TP omitted —
        # the live exit is the RL ratchet, not the risk head.
        real_m = (float(np.concatenate(pool['REAL'][0]).mean())
                  if pool['REAL'][0] else 0.0)
        shuf_m = (float(np.concatenate(pool['SHUFFLE'][0]).mean())
                  if pool['SHUFFLE'][0] else 0.0)
        rand_m = (float(np.concatenate(pool['RANDOM'][0]).mean())
                  if pool['RANDOM'][0] else 0.0)
        naive_m = (float(np.concatenate(pool['NAIVE'][0]).mean())
                   if pool.get('NAIVE', ([], []))[0] else 0.0)
        checks, all_pass, verdict = _operating_verdict(
            v_at, t_at, gap, real_m, shuf_m, rand_m, naive_m,
            thr=thr_star, records=out)
        print(f"\n   {'✅ PASS' if all_pass else '❌ FAIL'} — operating point:")
        for ok, msg in checks:
            print(f"     {'✓' if ok else '✗'} {msg}")
    return verdict if return_verdict else out


# Degenerate-shuffle floors (ported from the original XGBoost pipeline's
# phase_d): below either, the shuffled run is too thin for PF to be meaningful,
# so the shuffle line falls back to the meanR margin instead of a PF claim.
SHUF_MIN_TRADES_PF = 50


def _pf(R):
    """Profit factor of a realized-R array (gross win / gross loss)."""
    R = np.asarray(R, float)
    w = R[R > 0].sum()
    loss = abs(R[R < 0].sum())
    return (w / loss) if loss > 0 else float('inf')


def _print_robustness(out, pool):
    """Additive robustness views (no effect on any computed R, model, or the
    pre-registered verdict): per-OOS-month PF consistency, per-ticker PF, and a
    shuffle-survival check with a degenerate-sample guard. Print-only."""
    bar = "─" * 60
    print(f"\n{bar}\n🛡  ROBUSTNESS — per-OOS-month + per-ticker + shuffle "
          f"survival\n{bar}")

    # 1) per-OOS-month (fold) REAL PF gate — consistency across time
    by_fold = defaultdict(list)
    for r in out:
        by_fold[r['fold']].append(np.asarray(r['REAL'], float))
    fold_pf = {f: _pf(np.concatenate(v)) for f, v in by_fold.items()
               if sum(len(a) for a in v)}
    if fold_pf:
        n_mo = len(fold_pf)
        n_pos = sum(1 for pf in fold_pf.values() if pf > 1.0)
        flag = '✓ all months profitable' if n_pos == n_mo else \
               f'✗ {n_mo - n_pos} negative month(s)'
        print(f"  per-OOS-month PF>1: {n_pos}/{n_mo}  {flag}")

    # 2) per-ticker REAL PF + positivity
    realR = (np.concatenate(pool['REAL'][0]) if pool['REAL'][0]
             else np.array([], float))
    tks = pool['REAL'][1]
    if len(realR) and len(realR) == len(tks):
        by_tk = defaultdict(list)
        for r, t in zip(realR, tks):
            by_tk[t].append(r)
        n_pos = sum(1 for rs in by_tk.values() if np.mean(rs) > 0)
        cells = "  ".join(f"{t}:PF{_pf(rs):.2f}"
                          for t, rs in sorted(by_tk.items()))
        flag = '✓' if n_pos == len(by_tk) else '✗'
        print(f"  per-ticker meanR>0: {n_pos}/{len(by_tk)} {flag}   [{cells}]")

    # 3) shuffle survival (with degenerate-sample guard)
    shufR = (np.concatenate(pool['SHUFFLE'][0]) if pool['SHUFFLE'][0]
             else np.array([], float))
    if len(realR) and len(shufR):
        r_mean = float(realR.mean())
        s_mean = float(shufR.mean())
        margin = r_mean - s_mean
        ok = margin >= PASS_LIFT_MARGIN_R
        if len(shufR) < SHUF_MIN_TRADES_PF:
            print(f"  shuffle thin (n={len(shufR)}) — margin only: REAL "
                  f"{r_mean:+.3f} − SHUF {s_mean:+.3f} = {margin:+.3f}R "
                  f"{'✓' if ok else '✗'}")
        else:
            print(f"  REAL PF {_pf(realR):.2f} (meanR {r_mean:+.3f}) vs "
                  f"SHUFFLE PF {_pf(shufR):.2f} (meanR {s_mean:+.3f}) → "
                  f"margin {margin:+.3f}R  "
                  f"{'✓ real clears shuffle' if ok else '✗ margin below bar'}")


def _print_verdict(dyn_rows, pool, risk_mae, risk_corr):
    """Apply pre-registered criteria to the dashboard data and print
    PASS/FAIL with explicit reasons. No human interpretation — just numbers
    against the constants at top of this module."""
    real_meanR = float(np.concatenate(pool['REAL'][0]).mean()) \
        if pool['REAL'][0] else 0.0
    shuffle_meanR = float(np.concatenate(pool['SHUFFLE'][0]).mean()) \
        if pool['SHUFFLE'][0] else 0.0
    random_meanR = float(np.concatenate(pool['RANDOM'][0]).mean()) \
        if pool['RANDOM'][0] else 0.0
    naive_meanR = (float(np.concatenate(pool['NAIVE'][0]).mean())
                   if pool.get('NAIVE', ([], []))[0] else 0.0)

    # Find lowest threshold meeting ALL of (WR, meanR, trades) targets —
    # gives max trade count among passing thresholds.
    pass_row = next((r for r in dyn_rows
                     if r[2] >= PASS_TARGET_WR
                     and r[3] >= PASS_TARGET_MEAN_R
                     and r[1] >= PASS_MIN_TRADES_AGG), None)

    # Per-ticker lift over NAIVE
    real_by_tk = defaultdict(list)
    for r, t in zip(np.concatenate(pool['REAL'][0]) if pool['REAL'][0]
                    else np.array([], float), pool['REAL'][1]):
        real_by_tk[t].append(r)
    naive_by_tk = defaultdict(list)
    if pool.get('NAIVE', ([], []))[0]:
        for r, t in zip(np.concatenate(pool['NAIVE'][0]),
                        pool['NAIVE'][1]):
            naive_by_tk[t].append(r)
    n_lift = sum(1 for tk, rs in real_by_tk.items()
                 if np.mean(rs) > np.mean(naive_by_tk.get(tk, [0.0])))

    risk_ok = (risk_mae is not None and risk_corr is not None
               and risk_mae <= PASS_MAX_RISK_MAE_R
               and risk_corr >= PASS_MIN_RISK_CORR)

    checks = [
        (pass_row is not None,
         f"a threshold meets WR≥{int(PASS_TARGET_WR*100)}% + "
         f"AvgR≥{PASS_TARGET_MEAN_R}R + trades≥{PASS_MIN_TRADES_AGG}"),
        (real_meanR - shuffle_meanR >= PASS_LIFT_MARGIN_R,
         f"REAL meanR beats SHUFFLE by ≥{PASS_LIFT_MARGIN_R}R "
         f"(got {real_meanR - shuffle_meanR:+.2f}R)"),
        (real_meanR - random_meanR >= PASS_LIFT_MARGIN_R,
         f"REAL meanR beats RANDOM by ≥{PASS_LIFT_MARGIN_R}R "
         f"(got {real_meanR - random_meanR:+.2f}R)"),
        (real_meanR - naive_meanR >= PASS_LIFT_MARGIN_R,
         f"REAL meanR beats NAIVE by ≥{PASS_LIFT_MARGIN_R}R "
         f"(got {real_meanR - naive_meanR:+.2f}R)"),
        (risk_ok,
         f"risk-head useful: MAE≤{PASS_MAX_RISK_MAE_R}R + "
         f"corr≥{PASS_MIN_RISK_CORR}" + (
             "" if risk_mae is None
             else f" (got MAE {risk_mae:.2f}, r {risk_corr:+.2f})")),
        (n_lift >= PASS_MIN_TICKERS_LIFT,
         f"≥{PASS_MIN_TICKERS_LIFT} tickers above NAIVE on meanR "
         f"(got {n_lift})"),
    ]
    all_pass = all(ok for ok, _ in checks)

    bar = "═" * 60
    print(f"\n{bar}\n🚦 PRE-REGISTERED VERDICT\n{bar}")
    if all_pass:
        thr, trades, wr, meanR, sumR, pf = pass_row
        pf_s = f"{pf:.2f}" if pf < 999 else "inf"
        print(f"✅ PASS  —  thr={thr:.2f} / WR={100*wr:.1f}% / "
              f"AvgR={meanR:+.2f}R / n={trades} / PF={pf_s}")
    else:
        print("❌ FAIL  —  not all criteria met")
        if dyn_rows:
            best = max(dyn_rows, key=lambda r: r[3])    # max meanR
            thr, trades, wr, meanR, sumR, pf = best
            pf_s = f"{pf:.2f}" if pf < 999 else "inf"
            print(f"   Best meanR @ thr={thr:.2f}: WR={100*wr:.1f}% / "
                  f"AvgR={meanR:+.2f}R / n={trades} / PF={pf_s}")
    for ok, msg in checks:
        print(f"   {'✓' if ok else '✗'} {msg}")
    if not all_pass:
        print("\n   ↳ NEXT: try labeler variants (RR target sweep, "
              "MFE-trail exit, feature subset A/B), then re-run.")
