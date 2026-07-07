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

from ...classifier import Classifier, register_classifier

_EMBED_KEYS = ('model_id', 'device', 'batch')

_CKPT_FP_CACHE: dict = {}


def _ckpt_fingerprint(p: Path) -> str:
    """Short CONTENT hash of the checkpoint file — the embed-cache identity. STABLE across
    re-clones: identical weights -> identical fingerprint -> the cache is REUSED. Replaces the
    file MTIME, which changes on every git re-clone / re-download and churns a whole new ~30GB
    cache set for byte-identical embeddings (the Colab-disk / Drive-quota pain). Memoized per
    (path, size) so the 32MB file is hashed once per process, not per stream."""
    st = p.stat()
    key = (str(p), st.st_size)
    fp = _CKPT_FP_CACHE.get(key)
    if fp is None:
        h = hashlib.sha1()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        fp = h.hexdigest()[:16]
        _CKPT_FP_CACHE[key] = fp
    return fp


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
        ckpt_id = f"{p.name}:{_ckpt_fingerprint(p)}:{p.stat().st_size}"   # content hash, not mtime
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


def _bar_cache_path(cfg, labeler, keys):
    """Per-stream BAR-INDEXED cache (ohlcv mode): ONE file per (ticker,tf,ckpt,nbars,seq).
    In ohlcv mode the embedding of a window depends ONLY on its bar index (no direction flip),
    so ANY keyset that needs bar i reuses it — WF full-stream, produce train/val/oos subsets,
    and any head (logistic/MLP) all share one embed. None = caching off / non-ohlcv mode."""
    if os.environ.get('EMBED_CACHE', '1') != '1' or not keys:
        return None
    if getattr(labeler, 'MV_MODE', '?') != 'ohlcv':       # only ohlcv: emb independent of dir
        return None
    ckpt = cfg.get('backbone_ckpt')
    if ckpt and Path(ckpt).exists():
        p = Path(ckpt)
        ckpt_id = f"{p.name}-{_ckpt_fingerprint(p)}-{p.stat().st_size}"   # content hash, not mtime
    else:
        ckpt_id = (str(ckpt).replace('/', '_') if ckpt else 'vanilla')
    tk, tf = keys[0][0].split('@')
    seq = int(getattr(labeler, 'MV_SEQ', 0))
    try:
        nbars = int(len(labeler._b[(tk, tf)]['c']))
    except Exception:
        nbars = -1
    cache_dir = Path(os.environ.get('EMBED_CACHE_DIR', 'temp/embed_cache'))
    return cache_dir / f"bars_{tk}_{tf}_{ckpt_id}_{nbars}_{seq}.npz"


def _load_bar_cache(path):
    if path is None or not path.exists():
        return None, None
    try:
        d = np.load(path)
        return d['idx'], d['emb']
    except Exception:
        return None, None


def _save_bar_cache(path, idx, emb):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.{os.getpid()}.tmp.npz"      # atomic write
    np.savez(tmp, idx=np.asarray(idx, np.int64), emb=np.asarray(emb, np.float32))
    os.replace(tmp, path)


def _fit_with_heartbeat(clf, X, y, every=60):
    """clf.fit with a liveness heartbeat. sklearn solvers (lbfgs on millions of rows) are
    SILENT for the whole fit — on the full-data produce that's hours with no output, which
    reads as hung. A daemon thread prints elapsed time every `every`s while fit() runs, so
    a long fit is visibly alive; fits under `every`s stay quiet (no WF fold spam)."""
    import threading
    import time
    t0 = time.time()
    stop = threading.Event()

    def _beat():
        while not stop.wait(every):
            print(f"    [head] fitting {type(clf).__name__} on {len(X):,}x{X.shape[1]} "
                  f"... {time.time() - t0:,.0f}s elapsed (alive)", flush=True)

    th = threading.Thread(target=_beat, daemon=True)
    th.start()
    try:
        clf.fit(X, y)
    finally:
        stop.set()
        th.join(timeout=1)
    if time.time() - t0 >= every:
        print(f"    [head] fit done in {time.time() - t0:,.0f}s", flush=True)
    return clf


# Calibration is generic (backbone-agnostic) — lives in finetune.calibration. Re-exported here
# for back-compat with existing importers.
from ...calibration import fit_platt, apply_platt   # noqa: E402,F401


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
    cmd = [sys.executable, '-u', '-m', 'futures_foundation.finetune.classifiers.mantis._embed_worker']
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
        self._platt = None                  # Platt (A,B) after calibrate; None = raw proba

    def _embed(self, labeler, keys):
        """Frozen-encoder embedding of `keys` via the isolated subprocess worker -> [N, emb]."""
        windows = np.asarray(labeler.mv_contexts(keys), np.float32)        # [N, C, seq]
        ecfg = {k: self.cfg[k] for k in _EMBED_KEYS if k in self.cfg}
        ecfg['ckpt'] = self.cfg.get('backbone_ckpt')                       # SSL ckpt or None
        cmd = [sys.executable, '-u', '-m',
               'futures_foundation.finetune.classifiers.mantis._embed_worker']
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.save(d / 'w.npy', windows)
            (d / 'cfg.json').write_text(json.dumps(dict(ecfg, _windows=str(d / 'w.npy'))))
            r = subprocess.run(cmd + [str(d)], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"embed worker failed:\n{r.stderr[-2000:]}")
            return np.load(d / 'emb.npy')                  # [N, emb_dim]

    def _with_features(self, labeler, keys, emb):
        # concat the strategy's handcraft features (HTF dir / session / structure / ...) ->
        # [emb | handcraft], like the old Chronos fractal. Off via with_features=False.
        if self.cfg.get('with_features', True) and hasattr(labeler, 'features'):
            feats = np.nan_to_num(np.asarray(labeler.features(keys), np.float32))
            emb = np.concatenate([emb, feats], axis=1)     # [N, emb_dim + F]
        return emb[:, :, None]                             # -> [N, D, 1] for the memmap

    def featurize(self, labeler, keys):
        # 1) LEGACY per-keyset cache FIRST — preserves existing WF caching byte-for-byte.
        legacy = _embed_cache_path(self.cfg, labeler, keys)
        if legacy is not None and legacy.exists():
            cached = np.load(legacy)
            if len(cached) == len(keys):
                print(f"[embed-cache] HIT {legacy.name} ({len(cached)})", flush=True)
                return self._with_features(labeler, keys, cached)
        # 2) BAR-INDEXED cache — lets produce's train/val/oos subsets (and any head) share ONE
        #    embed. Embedding depends only on the bar index (ohlcv), so a subset slices the
        #    per-stream cache; missing bars are embedded once and appended.
        bpath = _bar_cache_path(self.cfg, labeler, keys)
        if bpath is not None:
            bar_idx = np.asarray([int(k[1]) for k in keys], np.int64)
            cidx, cemb = _load_bar_cache(bpath)
            if cemb is not None:
                pos = {int(b): r for r, b in enumerate(cidx)}
                sel = np.array([pos.get(int(b), -1) for b in bar_idx])
                if (sel >= 0).all():                       # full HIT -> slice
                    print(f"[embed-cache] HIT bars {bpath.name} ({len(sel)})", flush=True)
                    return self._with_features(labeler, keys, cemb[sel])
                miss = sel < 0                             # partial -> embed missing, append
                memb = self._embed(labeler, [keys[i] for i in np.where(miss)[0]])
                cidx = np.concatenate([cidx, bar_idx[miss]])
                cemb = np.concatenate([cemb, memb])
                _save_bar_cache(bpath, cidx, cemb)
                pos = {int(b): r for r, b in enumerate(cidx)}
                print(f"[embed-cache] +{int(miss.sum())} bars -> {bpath.name} ({len(cidx)})",
                      flush=True)
                return self._with_features(labeler, keys, cemb[[pos[int(b)] for b in bar_idx]])
            emb = self._embed(labeler, keys)              # first time -> seed the bar cache
            _save_bar_cache(bpath, bar_idx, emb)
            print(f"[embed-cache] WROTE bars {bpath.name} ({len(emb)})", flush=True)
            return self._with_features(labeler, keys, emb)
        # 3) caching off / non-ohlcv -> embed directly
        return self._with_features(labeler, keys, self._embed(labeler, keys))

    def fit_predict(self, Xtr, ytr, Xval, yval, Xeval, seed=0, keys_tr=None, keys_val=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import roc_auc_score
        # Train-stat standardize (streamed harness passes mu/sd in cfg; the in-RAM harness
        # pre-standardizes the arrays and passes NO stats — never double-applied). CRITICAL for
        # both fit quality (raw 1300-dim embeddings stall lbfgs at the iter cap) and the ONNX
        # serve contract: the bot standardizes with the contract mu/sd before the head, so the
        # head MUST be fit on standardized features or serving is a train/serve mismatch.
        mu, sd = self.cfg.get('standardize_mu'), self.cfg.get('standardize_sd')
        mu = None if mu is None else np.asarray(mu, np.float32).reshape(1, -1, 1)
        # harness stats already carry a +1e-6 epsilon; clamp anyway so hand-fed stats
        # (contract round-trip, manual cfg) can never divide by zero
        sd = None if sd is None else np.maximum(np.asarray(sd, np.float32), 1e-6).reshape(1, -1, 1)

        def arr(a):
            x = np.asarray(np.load(a, mmap_mode='r') if isinstance(a, str) else a, np.float32)
            if mu is not None:
                x = x - mu                                # new array (x may be a read-only mmap)
                x /= sd
            return x.reshape(len(x), -1)                  # [N, emb_dim, 1] -> [N, emb_dim]
        Xtr, Xval, Xeval = arr(Xtr), arr(Xval), arr(Xeval)
        ytr = np.asarray(ytr).astype(int); yval = np.asarray(yval).astype(int)
        if len(np.unique(ytr)) < 2:
            return np.full(len(Xval), .5), np.full(len(Xeval), .5), 0.5
        # DISTRIBUTIONAL reach-ladder head (rank='expected_reach'): instead of one P(3R) head,
        # fit a per-target survival head P(reach >= Xr) over reach_targets from the strategy keys
        # (each key carries per-target realized R), and RANK pivots by expected forward-R (the
        # area under the calibrated survival curve). The ship METRIC is unchanged — WR@3R — since
        # yval stays the 3R label and expected-R ranks 3R-reachers to the top; the ladder just adds
        # the "how far will it run" signal (P(>=6R/8R) big-runner detection) to the ranking. Needs
        # keys; falls back to the single head if they aren't threaded. Head type = MLP (same as the
        # signal head) so every rung is an MLPClassifier.
        if self.cfg.get('rank') == 'expected_reach' and keys_tr is not None:
            from ...risk_head import RiskHead, TARGETS
            if self.cfg.get('export_onnx_path'):
                # The ladder head is 5 per-rung MLPs + the survival->expected_reach reduction; that
                # ONNX graph isn't wired yet. Fail LOUD instead of silently exporting nothing (this
                # branch returns before the single-head export block below).
                raise NotImplementedError(
                    "ONNX export for the distributional reach-ladder head is not implemented yet "
                    "(rank='expected_reach'). Run produce with EXPORT_ONNX=0 to get the 2026 OOS "
                    "metrics; build the ladder ONNX bundle (5 rungs + expected-R reduction) before "
                    "deploying it to the bot.")
            targets = tuple(self.cfg.get('reach_targets', TARGETS))
            rh = RiskHead(targets=targets, head=self.cfg.get('head', 'mlp'),
                          calibrate=bool(self.cfg.get('calibrate', True)),
                          hidden=tuple(self.cfg.get('hidden', (128,))),
                          max_iter=int(self.cfg.get('max_iter', 300)),
                          mlp_batch=int(self.cfg.get('mlp_batch', 4096)),
                          mlp_alpha=float(self.cfg.get('mlp_alpha', 1e-4)),
                          C=float(self.cfg.get('C', 1.0)))
            rh.fit(Xtr, keys_tr, Xval, keys_val, seed=seed)
            self._risk_head = rh
            p_val = rh.predict_stats(Xval)['exp_reach']
            p_eval = rh.predict_stats(Xeval)['exp_reach']
            auc = roc_auc_score(yval, p_val) if len(np.unique(yval)) == 2 else 0.5
            return p_val, p_eval, float(auc)
        if self.cfg.get('head', 'logistic') == 'mlp':
            # batch_size: sklearn's default is min(200, n) — absurdly small at produce scale (2.78M
            # rows -> ~14k minibatches/epoch). A bigger batch = ~same solution (early-stopping guards
            # it) but 10-40x faster. alpha = L2 regularization; RAISE it to guard overfitting. Both
            # env-overridable so the WF can re-validate the setting before produce trusts it.
            clf = MLPClassifier(hidden_layer_sizes=tuple(self.cfg.get('hidden', (128,))),
                                max_iter=int(self.cfg.get('max_iter', 300)),
                                batch_size=int(self.cfg.get('mlp_batch', 4096)),
                                alpha=float(self.cfg.get('mlp_alpha', 1e-4)),
                                early_stopping=True, random_state=seed)
        else:
            clf = LogisticRegression(max_iter=int(self.cfg.get('max_iter', 1000)),
                                     C=float(self.cfg.get('C', 1.0)))
        _fit_with_heartbeat(clf, Xtr, ytr)
        if self.cfg.get('export_onnx_path'):          # deployable bundle: encoder + head ONNX
            # parity-check on a slice — running onnxruntime + predict_proba over the FULL val
            # set (~500k x 1311 at produce scale) stacks GBs at peak RAM for no extra signal
            _export_frozen_bundle(self.cfg, clf, Xtr.shape[1], Xval[:2048])
        # CALIBRATION (Platt) — OFF by default (cfg 'calibrate'). Fit on the VAL set, which the clf
        # never trained on (leak-free), so the proba tracks the empirical hit rate: P=0.5 => a true
        # ~50% signal, and the proba is a trustworthy regime-confidence across tiers (proba-sizing).
        # MLP proba is typically over-confident; logistic ~self-calibrated. AUC unchanged (Platt is
        # monotonic). Eval is calibrated OUT-OF-SAMPLE (Platt fit on val, applied to eval).
        raw_val = clf.predict_proba(Xval)[:, 1]
        self._platt = fit_platt(raw_val, yval) if self.cfg.get('calibrate') else None
        p_val = apply_platt(raw_val, self._platt)
        p_eval = apply_platt(clf.predict_proba(Xeval)[:, 1], self._platt)
        auc = roc_auc_score(yval, raw_val) if len(np.unique(yval)) == 2 else 0.5   # ranking-invariant
        return p_val, p_eval, float(auc)
