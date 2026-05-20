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

from .data import walk_forward_folds
from . import backbone
from .head_xgb import XGBHead


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


def run(labeler, head_factory=None, seeds=(0, 1, 2), train_m=3, test_m=1,
        max_folds=None):
    """labeler: a StrategyLabeler. head_factory: nc -> head (default
    XGBHead). max_folds=None -> sweep every available OOS month-pair
    (XGBoost-pipeline convention). Prints REAL/SHUFFLE/RANDOM per
    fold-seed, an aggregate stats block + per-ticker breakdown for
    REAL, AND (binary labelers only) a WR-by-confidence-threshold sweep
    — the entry-signal calibration test for downstream RL/account
    management. Returns the per-(fold,seed) records."""
    head_factory = head_factory or (lambda nc: XGBHead(nc))
    binary = labeler.n_classes == 2
    out = []
    pool = {'REAL': ([], []), 'SHUFFLE': ([], []), 'RANDOM': ([], [])}
    # WR-sweep scratch (binary only): per (fold, seed) save
    # (keys, proba_class1, Yte) so the post-hoc dashboard can compute
    # Prec/Recall/PF/vs-baseline at each confidence threshold.
    proba_records = []
    for fold, tr, te in walk_forward_folds(labeler.calendar(), train_m,
                                           test_m):
        if max_folds is not None and len(out) // max(1, len(seeds)) \
                >= max_folds:
            break
        ts0 = te['timestamp'].min()
        Ctr, Ytr, Ktr = labeler.build(tr['timestamp'].min(),
                                       tr['timestamp'].max(), ts0)
        Cte, Yte, Kte = labeler.build(
            te['timestamp'].min(),
            te['timestamp'].max() + np.timedelta64(1, 'ns'), None)
        if len(Ytr) < 50 or len(Cte) < 50:
            continue                       # unproductive fold
        Ytr = np.asarray(Ytr)
        Xtr = _featurize(labeler, Ctr, Ktr)
        Xte = _featurize(labeler, Cte, Kte)
        print(f"\n== fold {fold} | ntr={len(Ytr)} nte={len(Cte)} | "
              f"feat_dim={Xtr.shape[1]} | classes={labeler.n_classes} ==")
        for seed in seeds:
            head = head_factory(labeler.n_classes).fit(Xtr, Ytr, seed)
            p_real = head.predict(Xte)
            R = labeler.evaluate(Kte, p_real)
            if binary:
                # P(class 1) — confidence the model would TAKE this signal
                proba = head.predict_proba(Xte)[:, 1].astype(np.float32)
                proba_records.append(
                    (Kte, proba, np.asarray(Yte, np.int8)))
            ysh = Ytr.copy()
            np.random.default_rng(seed + 1).shuffle(ysh)
            p_sh = (head_factory(labeler.n_classes)
                    .fit(Xtr, ysh, seed).predict(Xte))
            Rs = labeler.evaluate(Kte, p_sh)
            p_rd = np.random.default_rng(seed + 2).integers(
                0, labeler.n_classes, len(Kte))
            Rr = labeler.evaluate(Kte, p_rd)
            print(f"  seed {seed}  [REAL   ] {_stats(R)}")
            print(f"          [SHUFFLE] {_stats(Rs)}")
            print(f"          [RANDOM ] {_stats(Rr)}")
            out.append({'fold': fold, 'seed': seed,
                        'REAL': R, 'SHUFFLE': Rs, 'RANDOM': Rr})
            pool['REAL'][0].append(R)
            pool['REAL'][1].extend(_taken_tickers(Kte, p_real))
            pool['SHUFFLE'][0].append(Rs)
            pool['SHUFFLE'][1].extend(_taken_tickers(Kte, p_sh))
            pool['RANDOM'][0].append(Rr)
            pool['RANDOM'][1].extend(_taken_tickers(Kte, p_rd))
    print(f"\n== AGGREGATE | folds={len({r['fold'] for r in out})} "
          f"seeds={len(seeds)} | pooled across all (fold,seed) ==")
    for name in ('REAL', 'SHUFFLE', 'RANDOM'):
        R = (np.concatenate(pool[name][0]) if pool[name][0]
             else np.array([], float))
        _agg_stats(name, R, tks=pool[name][1] if name == 'REAL' else None)
    print("\n-> Believe a result only if REAL clearly beats SHUFFLE AND "
          "RANDOM on sumR/meanR (cost in evaluate()), AND the per-ticker "
          "lift over the labeler's NAIVE baseline is consistent.")

    # ---- Confidence-threshold dashboard (binary labelers only) -----------
    # FFM/XGBoost-pipeline-style: per threshold, report trades / wins / WR
    # (= Prec) / Recall / PF / vs-NAIVE / verdict. The entry-signal
    # calibration test for downstream RL/account managers — pick the
    # threshold where WR is high enough AND trade frequency is meaningful.
    if binary and proba_records:
        # NAIVE-WR baseline = take every signal -> WR = base-rate of label==1
        all_y = np.concatenate([y for _, _, y in proba_records])
        naive_wr = float(np.mean(all_y == 1)) if len(all_y) else 0.0

        print(f"\n🎯 CONFIDENCE THRESHOLDS — REAL (pooled all fold,seed)")
        print(f"   {'Thresh':>6}  {'Trades':>6}  {'Wins':>5}  {'WR':>6}  "
              f"{'Recall':>6}  {'PF':>6}  {'vs NAIVE':>8}  Verdict")
        print(f"   {'-' * 66}")
        total_pos = int((all_y == 1).sum())
        for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            R_all, tp_total, fp_total = [], 0, 0
            for keys, proba, yte in proba_records:
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
        print(f"\n   NAIVE-WR baseline (take every signal): "
              f"{100 * naive_wr:.1f}%")
        print(f"   Target for RL entry signal: WR ≥ 60% at meaningful "
              f"trade frequency (don't chase 100% WR by shrinking to ~0 "
              f"trades).")
    return out
