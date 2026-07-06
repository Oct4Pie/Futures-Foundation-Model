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
import os
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
        'contract_version': '1.0', 'role': 'signal', 'classifier': classifier,
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


def _bundle_files(base):
    """Existing ONNX artifact files for a given export base (single-file classifiers write
    <base>.onnx; the frozen bundle writes <base>_encoder.onnx + <base>_signal_head.onnx)."""
    return [p for p in (str(base) + '.onnx', str(base) + '_encoder.onnx',
                        str(base) + '_signal_head.onnx') if Path(p).exists()]


def _emit(out, classifier, ck, eval_lab, mu, sd, C, seq,
          channel_names, tks, tfs, holdout_start, export_onnx, output_path, verbose):
    """Write the deploy contract for the ONNX exported DURING the REAL fit (no refit — a
    third full fit doubled peak RAM and OOM-killed the Colab session at produce scale)."""
    if not (export_onnx and output_path):
        return out
    base = Path(output_path).with_suffix('')
    bundle = _bundle_files(base)
    sha = (hashlib.sha256(b''.join(Path(p).read_bytes() for p in bundle)).hexdigest()
           if bundle else None)
    contract = {
        'contract_version': '1.0', 'role': 'signal', 'classifier': classifier,
        'input': {'channels': int(C), 'seq_len': int(seq),
                  'mv_mode': getattr(eval_lab, 'MV_MODE', None)},
        'channel_names': channel_names,
        'standardize': ({'mu': np.asarray(mu).tolist(), 'sd': np.asarray(sd).tolist()}
                        if mu is not None else None),
        'mv_contexts_fn': 'strategy.mv_contexts (direction-normalized causal window)',
        'nan_policy': {'posinf': 0, 'neginf': 0, 'nan': 0},
        'ft_config': {k: v for k, v in (ck or {}).items()
                      if k not in ('standardize_mu', 'standardize_sd', 'log_path')},
        'n_classes': 2,
        'proba_meaning': ('P(reach target before stop) — Platt-CALIBRATED to the empirical hit rate'
                          if out.get('platt') else 'P(good trend pivot reaches target before stop)'),
        # head.onnx emits RAW proba; the bot applies calibration AFTER it -> cal = sigmoid(A*logit(p)+B).
        # None = serve raw. Carrying it here is what keeps serve-time proba == the calibrated produce metric.
        'calibration': ({'method': 'platt', 'formula': 'sigmoid(A*logit(p_raw)+B)',
                         'A': out['platt'][0], 'B': out['platt'][1]} if out.get('platt') else None),
        'output_fn': ('platt(head_raw_proba)' if out.get('platt') else 'softmax(logits)[:,1]'),
        'train_scope': {'tickers': tks, 'timeframes': tfs, 'holdout_start': holdout_start,
                        'n_train': int(out['n_train']), 'n_oos': int(out['n_oos'])},
        'oos_metrics': {k: out.get(k) for k in
                        ('oos_auc', 'oos_meanR', 'shuffle_meanR', 'edge_shuffle', 'oos_trades')},
        'onnx': ([Path(p).name for p in bundle] if sha else None), 'content_sha': sha,
    }
    cpath = str(base) + '_signal.json'
    Path(cpath).write_text(json.dumps(contract, indent=2))
    out['artifacts'] = {'onnx': (bundle if sha else None), 'contract': cpath, 'content_sha': sha}
    if verbose:
        print(f"  artifacts: onnx={out['artifacts']['onnx']} contract={cpath}")
    return out


def _fit_score(classifier, ck, eval_lab, Xtr, Ytr_tr, Xval, Ytr_va, Xte, Kte, Yte, seed, verbose,
               onnx_path=None):
    """Fit REAL (exporting ONNX during that fit when onnx_path is set) + SHUFFLE control.
    Two fits total — the export must ride the REAL fit; a separate export refit doubles
    peak RAM (fresh standardized copies on top of allocator creep) and OOMs at full scale."""
    import gc
    rng = np.random.default_rng(seed)
    if verbose:
        print(f"  [produce 1/2] fit REAL head{' (+ ONNX export)' if onnx_path else ''}",
              flush=True)
    ck_real = dict(ck, export_onnx_path=onnx_path) if onnx_path else ck
    clf_real = get_classifier(classifier, **ck_real)          # keep the instance: it holds _platt
    p_val, p_te, ba = clf_real.fit_predict(Xtr, Ytr_tr, Xval, Ytr_va, Xte, seed)
    gc.collect()
    thr = _pct_threshold(p_val, OP_PERCENTILE)
    R = _arm_R(eval_lab, Kte, p_te, thr)
    # SHUFFLE control = the produce-side honest ruler. SKIP_SHUFFLE=1 skips it (halves produce time)
    # when the WF ALREADY ran the honest ruler on this config; REAL + calibration are unaffected.
    if os.environ.get('SKIP_SHUFFLE') == '1':
        Rs = None
        if verbose:
            print("  [produce 2/2] SKIPPED (SKIP_SHUFFLE=1; honest ruler already run in WF)", flush=True)
    else:
        ysh = np.asarray(Ytr_tr).copy(); rng.shuffle(ysh)
        if verbose:
            print("  [produce 2/2] fit SHUFFLE control (honest ruler)", flush=True)
        psv, ps, _ = get_classifier(classifier, **ck).fit_predict(Xtr, ysh, Xval, Ytr_va, Xte, seed)
        gc.collect()
        Rs = _arm_R(eval_lab, Kte, ps, _pct_threshold(psv, OP_PERCENTILE))
    auc = None
    if len(np.unique(Yte)) == 2:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(Yte, p_te))
    edge = (_meanR(R) - _meanR(Rs)) if Rs is not None else None
    out = dict(oos_auc=auc, best_val_auc=ba, oos_meanR=_meanR(R),
               shuffle_meanR=(_meanR(Rs) if Rs is not None else None), edge_shuffle=edge,
               n_train=len(Ytr_tr), n_oos=len(Kte), oos_trades=int(len(R)),
               beats_shuffle=(bool(edge >= PASS_LIFT_MARGIN_R) if edge is not None else None),
               platt=getattr(clf_real, '_platt', None))       # Platt (A,B) -> deploy contract
    if verbose:
        print(f"  OOS AUC {auc:.4f}" if auc is not None else "  OOS AUC n/a")
        if edge is not None:
            print(f"  OOS meanR REAL {out['oos_meanR']:+.3f} SHUFFLE {out['shuffle_meanR']:+.3f} "
                  f"edge {edge:+.3f} (trades={out['oos_trades']})  -> "
                  f"{'beats SHUFFLE' if out['beats_shuffle'] else 'does NOT beat SHUFFLE'}")
        else:
            print(f"  OOS meanR REAL {out['oos_meanR']:+.3f} (SHUFFLE skipped; trades={out['oos_trades']})")
    return out


def train_final_streamed(make_labeler, streams, classifier, clf_kwargs=None,
                         holdout_start='2026-01-01', val_frac=0.15, seed=0, chunk=2000,
                         export_onnx=False, output_path=None, verbose=True):
    """Run on ALL data across many (ticker, timeframe) streams with bounded memory:
    load each stream sequentially, featurize its train/val/oos pivots to part memmaps,
    RELEASE its bars, next stream; concat parts into full memmaps; train per-batch.
    Peak RAM = one stream + one batch. This is the full 3/5/15 (or all-tickers) run."""
    import gc
    from ._memmap import featurize_to_memmap, concat_memmaps, memmap_standardize_stats
    clf = get_classifier(classifier, **dict(clf_kwargs or {}))
    hs = pd.Timestamp(holdout_start, tz='UTC')
    rng = np.random.default_rng(seed)
    rundir = (Path(output_path).parent if output_path else Path('.'))
    rundir.mkdir(parents=True, exist_ok=True)

    tr_parts, va_parts, te_parts = [], [], []
    Ytr_tr, Ytr_va, all_Kte, all_Yte = [], [], [], []
    channel_names = None; C = seq = None; eval_lab = None
    for i, (tk, tf) in enumerate(streams):
        lab = make_labeler(tk, tf)                       # loads ONLY this stream's bars
        cal = lab.calendar(); lo, hi = cal['timestamp'].min(), cal['timestamp'].max()
        _, Ytr, Ktr = lab.build(lo, hs, hs)
        _, Yte, Kte = lab.build(hs, hi + pd.Timedelta('1ns'), None)
        Ytr = np.asarray(Ytr).astype(int); Yte = np.asarray(Yte).astype(int)
        if channel_names is None and hasattr(lab, 'mv_feature_names'):
            channel_names = lab.mv_feature_names()
        if len(Ktr) >= 2:
            idx = rng.permutation(len(Ktr)); nv = max(1, int(len(Ktr) * val_frac))
            va_i, tr_i = idx[:nv], idx[nv:]
            p = str(rundir / f'_tr{i}.npy')
            _, sh = featurize_to_memmap(clf, lab, [Ktr[j] for j in tr_i], p, chunk)
            tr_parts.append((p, len(tr_i))); Ytr_tr += list(Ytr[tr_i]); C, seq = sh[1], sh[2]
            p = str(rundir / f'_va{i}.npy')
            featurize_to_memmap(clf, lab, [Ktr[j] for j in va_i], p, chunk)
            va_parts.append((p, len(va_i))); Ytr_va += list(Ytr[va_i])
        if len(Kte):
            p = str(rundir / f'_te{i}.npy')
            featurize_to_memmap(clf, lab, list(Kte), p, chunk)
            te_parts.append((p, len(Kte))); all_Kte += list(Kte); all_Yte += list(Yte)
        if verbose:
            print(f"  [stream {tk}@{tf}] train={len(Ktr)} oos={len(Kte)}", flush=True)
        for attr in ('_b', '_labels'):                   # release bars (evaluate uses keys)
            if hasattr(lab, attr):
                try:
                    getattr(lab, attr).clear()
                except Exception:
                    pass
        eval_lab = lab                                   # keep one (R from key tuples)
        gc.collect()

    Xtr, _ = concat_memmaps(tr_parts, str(rundir / '_Xtr.npy'))
    Xval, _ = concat_memmaps(va_parts, str(rundir / '_Xval.npy'))
    Xte, _ = concat_memmaps(te_parts, str(rundir / '_Xte.npy'))
    Ytr_tr = np.array(Ytr_tr); Ytr_va = np.array(Ytr_va); Yte = np.array(all_Yte)
    ck = dict(clf_kwargs or {}); mu = sd = None
    if clf.needs_standardize:
        mu, sd = memmap_standardize_stats(Xtr)
        ck['standardize_mu'] = mu.tolist(); ck['standardize_sd'] = sd.tolist()
    tks = sorted({s[0] for s in streams}); tfs = sorted({s[1] for s in streams})
    if verbose:
        print(f"=== PRODUCE STREAMED ({classifier}: {len(streams)} streams, 2026 OOS) ===")
        print(f"  train={len(Ytr_tr)} val={len(Ytr_va)} oos={len(all_Kte)} C={C} seq={seq} "
              f"good(train)={Ytr_tr.mean():.3f} good(oos)={Yte.mean():.3f}", flush=True)
    onnx_path = (str(Path(output_path).with_suffix('')) + '.onnx'
                 if (export_onnx and output_path) else None)
    out = _fit_score(classifier, ck, eval_lab, Xtr, Ytr_tr, Xval, Ytr_va, Xte, all_Kte, Yte,
                     seed, verbose, onnx_path=onnx_path)
    return _emit(out, classifier, ck, eval_lab, mu, sd, C, seq,
                 channel_names, tks, tfs, holdout_start, export_onnx, output_path, verbose)


def train_final(labeler, classifier, clf_kwargs=None, holdout_start='2026-01-01',
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

    onnx_path = (str(Path(output_path).with_suffix('')) + '.onnx'
                 if (export_onnx and output_path) else None)
    if verbose:
        print(f"  [produce 1/2] fit REAL head{' (+ ONNX export)' if onnx_path else ''}",
              flush=True)
    ck_real = dict(ck, export_onnx_path=onnx_path) if onnx_path else ck
    p_val, p_te, ba = get_classifier(classifier, **ck_real).fit_predict(
        Xtr, Ytr_tr, Xval, Ytr_va, Xte, seed)
    thr = _pct_threshold(p_val, OP_PERCENTILE)
    R = _arm_R(labeler, Kte, p_te, thr)
    ysh = Ytr_tr.copy(); rng.shuffle(ysh)
    if verbose:
        print("  [produce 2/2] fit SHUFFLE control (honest ruler)", flush=True)
    psv, ps, _ = get_classifier(classifier, **ck).fit_predict(Xtr, ysh, Xval, Ytr_va, Xte, seed)
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
        bundle = _bundle_files(base)                  # exported during the REAL fit (no refit)
        sha = (hashlib.sha256(b''.join(Path(p).read_bytes() for p in bundle)).hexdigest()
               if bundle else None)
        contract = _contract(labeler, classifier, ck, C, seq, mu, sd, out, onnx_path, sha,
                             holdout_start, len(tr_i), len(Kte))
        contract['onnx'] = [Path(p).name for p in bundle] if sha else None
        cpath = str(base) + '_signal.json'
        Path(cpath).write_text(json.dumps(contract, indent=2))
        out['artifacts'] = {'onnx': (bundle if sha else None), 'contract': cpath,
                            'content_sha': sha}
        if verbose:
            print(f"  artifacts: onnx={out['artifacts']['onnx']} contract={cpath}")
    return out
