"""Final-model training + held-out OOS — the 'produce' half (mirrors pipeline.produce).

Two modes:
  stream=False (default): featurize the train/val/oos windows into RAM arrays. Fine for
    small/medium sets and the torch-free tests.
  stream=True (the REAL full-data run): featurize to DISK memmaps in chunks (parent RAM
    = one chunk) and train PER-BATCH from the memmaps (worker RAM = one batch), with
    per-channel standardize stats applied per-batch. Lets a transformer train on the
    FULL aligned-pivot set (no 5.6GB array in RAM) — the data it needs to not overfit.

Either way: train on all data < holdout_start (inner val + early-stop), score the
held-out 2026 OOS once with a SHUFFLE control, and (export_onnx) emit <base>.onnx +
<base>_signal.json (input spec, channel names, standardize mu/sd, oos metrics, sha).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .classifier import get_classifier
from .wf import (_pct_threshold, _arm_R, _meanR, _standardize_on_train,
                 OP_PERCENTILE, PASS_LIFT_MARGIN_R)


def _contract(labeler, classifier, ck, C, seq, mu, sd, out, onnx_path, sha,
              holdout_start, n_train, n_oos):
    tfs = sorted({k[1] for k in getattr(labeler, '_b', {})}) or None
    tks = sorted({k[0] for k in getattr(labeler, '_b', {})}) or None
    return {
        'contract_version': '1.0', 'role': 'mantis_signal', 'classifier': classifier,
        'input': {'channels': int(C), 'seq_len': int(seq),
                  'mv_mode': getattr(labeler, 'MV_MODE', None)},
        'channel_names': (labeler.mv_feature_names()
                          if hasattr(labeler, 'mv_feature_names') else None),
        'standardize': ({'mu': np.asarray(mu).tolist(), 'sd': np.asarray(sd).tolist()}
                        if mu is not None else None),
        'mv_contexts_fn': 'strategy.mv_contexts (direction-normalized causal window)',
        'nan_policy': {'posinf': 0, 'neginf': 0, 'nan': 0},
        'ft_config': {k: v for k, v in (ck or {}).items()
                      if k not in ('standardize_mu', 'standardize_sd', 'log_path')},
        'n_classes': 2, 'proba_meaning': 'P(good trend pivot reaches target before stop)',
        'output_fn': 'softmax(logits)[:,1]',
        'train_scope': {'tickers': tks, 'timeframes': tfs, 'holdout_start': holdout_start,
                        'n_train': int(n_train), 'n_oos': int(n_oos)},
        'oos_metrics': {k: out.get(k) for k in
                        ('oos_auc', 'oos_meanR', 'shuffle_meanR', 'edge_shuffle', 'oos_trades')},
        'onnx': (Path(onnx_path).name if sha else None), 'content_sha': sha,
    }


def train_final(labeler, classifier='mantis', clf_kwargs=None, holdout_start='2026-01-01',
                val_frac=0.15, seed=0, max_train=None, stream=False, chunk=2000,
                export_onnx=False, output_path=None, verbose=True):
    clf = get_classifier(classifier, **dict(clf_kwargs or {}))
    hs = pd.Timestamp(holdout_start, tz='UTC')
    cal = labeler.calendar()
    lo, hi = cal['timestamp'].min(), cal['timestamp'].max()
    Ctr, Ytr, Ktr = labeler.build(lo, hs, hs)
    Cte, Yte, Kte = labeler.build(hs, hi + pd.Timedelta('1ns'), None)
    Ytr = np.asarray(Ytr).astype(int); Yte = np.asarray(Yte).astype(int)
    if len(Ytr) < 50 or len(Kte) < 20:
        raise ValueError(f"insufficient data: train={len(Ytr)} oos={len(Kte)}")
    if max_train and len(Ktr) > max_train:
        sub = np.random.default_rng(seed).choice(len(Ktr), max_train, replace=False)
        Ktr = [Ktr[j] for j in sub]; Ytr = Ytr[sub]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(Ktr)); rng.shuffle(idx)
    nv = max(10, int(len(idx) * val_frac))
    va_i, tr_i = idx[:nv], idx[nv:]
    Ktr_tr = [Ktr[j] for j in tr_i]; Ytr_tr = Ytr[tr_i]
    Ktr_va = [Ktr[j] for j in va_i]; Ytr_va = Ytr[va_i]

    ck = dict(clf_kwargs or {})
    mu = sd = None
    if stream:
        from ._memmap import featurize_to_memmap, memmap_standardize_stats
        rundir = (Path(output_path).parent if output_path else Path('.'))
        rundir.mkdir(parents=True, exist_ok=True)
        Xtr = str(rundir / '_Xtr.npy'); Xval = str(rundir / '_Xval.npy'); Xte = str(rundir / '_Xte.npy')
        _, (ntr, C, seq) = featurize_to_memmap(clf, labeler, Ktr_tr, Xtr, chunk)
        featurize_to_memmap(clf, labeler, Ktr_va, Xval, chunk)
        featurize_to_memmap(clf, labeler, Kte, Xte, chunk)
        if clf.needs_standardize:
            mu, sd = memmap_standardize_stats(Xtr)
            ck['standardize_mu'] = mu.tolist(); ck['standardize_sd'] = sd.tolist()
        # features are on disk now; evaluate() reads R from the key tuples (not _b), so
        # drop the labeler's bars to free RAM before the worker trains (stream is the
        # memory-critical full-data path — always free).
        import gc
        for attr in ('_b', '_labels'):
            if hasattr(labeler, attr):
                try:
                    getattr(labeler, attr).clear()
                except Exception:
                    pass
        gc.collect()
        if verbose:
            print("  [mem] freed labeler bars after featurize (memmap holds features)",
                  flush=True)
    else:
        Xtr = clf.featurize(labeler, Ktr_tr)
        Xval = clf.featurize(labeler, Ktr_va)
        Xte = clf.featurize(labeler, Kte)
        C, seq = int(Xtr.shape[1]), int(Xtr.shape[2])
        if clf.needs_standardize:
            Xtr, Xval, Xte, mu, sd = _standardize_on_train(Xtr, Xval, Xte)

    if verbose:
        print(f"=== PRODUCE ({classifier}{' STREAM' if stream else ''}: train < "
              f"{holdout_start}, 2026 OOS) ===")
        print(f"  train={len(tr_i)} val={len(va_i)} oos={len(Kte)} C={C} seq={seq} "
              f"good(train)={Ytr_tr.mean():.3f} good(oos)={Yte.mean():.3f}", flush=True)

    clf_run = get_classifier(classifier, **ck)
    p_val, p_te, ba = clf_run.fit_predict(Xtr, Ytr_tr, Xval, Ytr_va, Xte, seed)
    thr = _pct_threshold(p_val, OP_PERCENTILE)
    R = _arm_R(labeler, Kte, p_te, thr)
    ysh = Ytr_tr.copy(); rng.shuffle(ysh)
    psv, ps, _ = clf_run.fit_predict(Xtr, ysh, Xval, Ytr_va, Xte, seed)
    Rs = _arm_R(labeler, Kte, ps, _pct_threshold(psv, OP_PERCENTILE))

    auc = None
    if len(np.unique(Yte)) == 2:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(Yte, p_te))
    edge = _meanR(R) - _meanR(Rs)
    out = dict(oos_auc=auc, best_val_auc=ba, oos_meanR=_meanR(R), shuffle_meanR=_meanR(Rs),
               edge_shuffle=edge, n_train=len(tr_i), n_oos=len(Kte), oos_trades=int(len(R)),
               beats_shuffle=bool(edge >= PASS_LIFT_MARGIN_R))
    if verbose:
        print(f"  OOS AUC {auc:.4f}" if auc is not None else "  OOS AUC n/a")
        print(f"  OOS meanR REAL {out['oos_meanR']:+.3f} SHUFFLE {out['shuffle_meanR']:+.3f} "
              f"edge {edge:+.3f} (trades={out['oos_trades']})")
        print(f"  -> {'beats SHUFFLE' if out['beats_shuffle'] else 'does NOT beat SHUFFLE'}")

    if export_onnx and output_path:
        base = Path(output_path).with_suffix('')
        onnx_path = str(base) + '.onnx'
        get_classifier(classifier, **dict(ck, export_onnx_path=onnx_path)).fit_predict(
            Xtr, Ytr_tr, Xval, Ytr_va, Xte, seed)
        sha = (hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()
               if Path(onnx_path).exists() else None)
        contract = _contract(labeler, classifier, ck, C, seq, mu, sd, out, onnx_path, sha,
                             holdout_start, len(tr_i), len(Kte))
        cpath = str(base) + '_signal.json'
        Path(cpath).write_text(json.dumps(contract, indent=2))
        out['artifacts'] = {'onnx': onnx_path if sha else None, 'contract': cpath,
                            'content_sha': sha}
        if verbose:
            print(f"  artifacts: onnx={out['artifacts']['onnx']} contract={cpath}")
    return out
