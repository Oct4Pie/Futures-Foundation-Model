"""MantisFrozenClassifier — FROZEN backbone, only a head trains (the head-only path).

featurize() runs the strategy's multivariate windows through the FROZEN Mantis encoder ONCE
(via the isolated _embed_worker, encoder-only + interpolated to native length) -> a cached
embedding [N, C*hidden]. fit_predict() then trains a cheap HEAD on those embeddings per fold
(torch-free sklearn) — the backbone is never updated. This is the Chronos+XGBoost "embed
once -> head per fold" pattern, but with the frozen masked-SSL Mantis encoder.

backbone_ckpt -> the masked-SSL encoder (the A/B "SSL" arm); None -> vanilla Mantis (the
"vanilla" arm). head='logistic' (linear probe of the frozen rep, default) or 'mlp'.
Registered as 'mantis_frozen'.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..classifier import Classifier, register_classifier

_EMBED_KEYS = ('model_id', 'device', 'batch')


def _embed_cache_path(cfg, labeler, keys):
    """Cross-run cache path for ONE stream's frozen embedding (None = caching off).

    The embedding of a window is deterministic in (backbone_ckpt, the bars, mv_mode,
    seq, the bar indices) — independent of labels/handcraft. So we key on exactly those
    and cache the [N, emb_dim] array; reruns / the verify / head-only iterations reuse it
    and never re-embed. Handcraft is concatenated FRESH after load, so the cache survives
    handcraft changes. EMBED_CACHE=0 disables; EMBED_CACHE_DIR overrides the location."""
    if os.environ.get('EMBED_CACHE', '1') != '1' or not keys:
        return None
    ckpt = cfg.get('backbone_ckpt')
    if ckpt and Path(ckpt).exists():
        p = Path(ckpt)
        ckpt_id = f"{p.name}:{int(p.stat().st_mtime)}:{p.stat().st_size}"
    else:
        ckpt_id = str(ckpt) if ckpt else 'vanilla'
    sid = keys[0][0]                                   # "TK@TF" (one stream per call)
    tk, tf = sid.split('@')
    mv_mode = getattr(labeler, 'MV_MODE', '?')
    seq = int(getattr(labeler, 'MV_SEQ', 0))
    try:
        nbars = int(len(labeler._b[(tk, tf)]['c']))   # data fingerprint (changes -> miss)
    except Exception:
        nbars = -1
    bi = np.asarray([int(k[1]) for k in keys], np.int64)
    di = np.asarray([int(k[2]) for k in keys], np.int64)
    h = hashlib.sha1()
    h.update(f"{ckpt_id}|{sid}|{mv_mode}|{seq}|{nbars}|{len(keys)}".encode())
    h.update(bi.tobytes()); h.update(di.tobytes())
    cache_dir = Path(os.environ.get('EMBED_CACHE_DIR', 'temp/embed_cache'))
    return cache_dir / f"{tk}_{tf}_{h.hexdigest()[:16]}.npy"


def export_head_onnx(clf, n_features, path):
    """Convert the fitted sklearn head (logistic/MLP) to ONNX: input [N, n_features] standardized
    [emb|handcraft] -> probabilities [N, 2]. zipmap off so the proba output is a plain array."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    onx = convert_sklearn(clf, initial_types=[('input', FloatTensorType([None, int(n_features)]))],
                          options={id(clf): {'zipmap': False}}, target_opset=15)
    Path(path).write_bytes(onx.SerializeToString())
    return path


def _export_frozen_bundle(cfg, clf, n_features, Xval_std):
    """Deployable ONNX bundle in the incumbent format: <base>_encoder.onnx (raw OHLCV window ->
    Mantis embedding) + <base>_signal_head.onnx (standardized [emb|handcraft] -> P). The encoder
    runs in the isolated subprocess (parent stays torch-free); the head converts via skl2onnx
    in-process. Head output is parity-checked vs the sklearn head. Bot serves: window ->
    encoder.onnx -> concat handcraft -> standardize (contract mu/sd) -> head.onnx -> P."""
    base = str(cfg['export_onnx_path'])
    if base.endswith('.onnx'):
        base = base[:-5]
    head_path, enc_path = base + '_signal_head.onnx', base + '_encoder.onnx'
    export_head_onnx(clf, n_features, head_path)
    ecfg = dict(_export_encoder=enc_path, ckpt=cfg.get('backbone_ckpt'),
                C=int(cfg.get('raw_C', 5)), seq=int(cfg.get('raw_seq', 64)),
                model_id=cfg.get('model_id', 'paris-noah/Mantis-8M'))
    cmd = [sys.executable, '-u', '-m', 'futures_foundation.finetune.classifiers._embed_worker']
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / 'cfg.json').write_text(json.dumps(ecfg))
        r = subprocess.run(cmd + [str(d)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"encoder onnx export failed:\n{r.stderr[-2000:]}")
    diff = -1.0
    try:                                              # parity: onnx head vs sklearn head
        import onnxruntime as ort
        sess = ort.InferenceSession(head_path, providers=['CPUExecutionProvider'])
        outs = sess.run(None, {'input': np.asarray(Xval_std, np.float32)})
        proba = [o for o in outs if getattr(o, 'ndim', 0) == 2 and o.shape[1] == 2][0]
        diff = float(np.abs(clf.predict_proba(Xval_std)[:, 1] - proba[:, 1]).max())
    except Exception as e:                            # pragma: no cover
        print(f"[onnx] head parity check skipped: {e}", flush=True)
    print(f"[onnx] wrote {enc_path} + {head_path}  head-parity max|diff|={diff:.2e}", flush=True)
    return enc_path, head_path


@register_classifier('mantis_frozen')
class MantisFrozenClassifier(Classifier):
    needs_standardize = True            # harness standardizes the cached embeddings on train
    embed_once = True                   # featurize the whole stream in ONE call (load Mantis once)

    def __init__(self, **cfg):
        self.cfg = cfg

    def featurize(self, labeler, keys):
        cpath = _embed_cache_path(self.cfg, labeler, keys)
        emb = None
        if cpath is not None and cpath.exists():
            cached = np.load(cpath)                        # cross-run HIT -> skip embed entirely
            if len(cached) == len(keys):
                emb = cached
                print(f"[embed-cache] HIT {cpath.name} ({len(emb)}x{emb.shape[1]})", flush=True)
        if emb is None:                                   # MISS -> embed (frozen) then cache
            windows = np.asarray(labeler.mv_contexts(keys), np.float32)    # [N, C, seq]
            ecfg = {k: self.cfg[k] for k in _EMBED_KEYS if k in self.cfg}
            ecfg['ckpt'] = self.cfg.get('backbone_ckpt')                   # SSL ckpt or None
            cmd = [sys.executable, '-u', '-m',
                   'futures_foundation.finetune.classifiers._embed_worker']
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                np.save(d / 'w.npy', windows)
                (d / 'cfg.json').write_text(json.dumps(dict(ecfg, _windows=str(d / 'w.npy'))))
                r = subprocess.run(cmd + [str(d)], capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"embed worker failed:\n{r.stderr[-2000:]}")
                emb = np.load(d / 'emb.npy')               # [N, emb_dim] (frozen OHLCV embedding)
            if cpath is not None:                          # persist for future runs (atomic write)
                cpath.parent.mkdir(parents=True, exist_ok=True)
                tmp = cpath.parent / f"{cpath.stem}.{os.getpid()}.tmp.npy"   # ends .npy
                np.save(tmp, emb); os.replace(tmp, cpath)
                print(f"[embed-cache] WROTE {cpath.name} ({len(emb)}x{emb.shape[1]})", flush=True)
        # concat the strategy's handcraft features (HTF dir / session / structure / ... the
        # market-context the OHLCV window can't express) -> [emb | handcraft], like the old
        # Chronos fractal (embed + handcraft -> head). Off via with_features=False.
        if self.cfg.get('with_features', True) and hasattr(labeler, 'features'):
            feats = np.nan_to_num(np.asarray(labeler.features(keys), np.float32))
            emb = np.concatenate([emb, feats], axis=1)    # [N, emb_dim + F]
        return emb[:, :, None]                            # -> [N, D, 1] for the WF memmap
        # (harness standardizes per-"channel" = per dim, seq=1; fit_predict flattens to [N, D])

    def fit_predict(self, Xtr, ytr, Xval, yval, Xeval, seed=0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import roc_auc_score

        def arr(a):
            x = np.asarray(np.load(a, mmap_mode='r') if isinstance(a, str) else a, np.float32)
            return x.reshape(len(x), -1)                  # [N, emb_dim, 1] -> [N, emb_dim]
        Xtr, Xval, Xeval = arr(Xtr), arr(Xval), arr(Xeval)
        ytr = np.asarray(ytr).astype(int); yval = np.asarray(yval).astype(int)
        if len(np.unique(ytr)) < 2:
            return np.full(len(Xval), .5), np.full(len(Xeval), .5), 0.5
        if self.cfg.get('head', 'logistic') == 'mlp':
            clf = MLPClassifier(hidden_layer_sizes=tuple(self.cfg.get('hidden', (128,))),
                                max_iter=int(self.cfg.get('max_iter', 300)),
                                early_stopping=True, random_state=seed)
        else:
            clf = LogisticRegression(max_iter=1000, C=float(self.cfg.get('C', 1.0)))
        clf.fit(Xtr, ytr)
        if self.cfg.get('export_onnx_path'):          # deployable bundle: encoder + head ONNX
            _export_frozen_bundle(self.cfg, clf, Xtr.shape[1], Xval)
        p_val = clf.predict_proba(Xval)[:, 1]
        p_eval = clf.predict_proba(Xeval)[:, 1]
        auc = roc_auc_score(yval, p_val) if len(np.unique(yval)) == 2 else 0.5
        return p_val, p_eval, float(auc)
