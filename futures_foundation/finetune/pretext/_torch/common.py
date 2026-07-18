"""Shared TORCH layer for the SSL pretext trainers.

Window helpers (encode / standardize / control-corrupt / gather), the frozen-embedding + ONNX
primitives, and a BaseTrainer that owns the SHARED training loop (epoch loop, AMP, grad-clip,
cosine LR, early-stop on val loss, best-ENCODER-state snapshot). Each pretext trainer
(mask / forecast / contrastive) subclasses BaseTrainer and provides only build_net / make_batch /
compute_loss / val_eval — no copied loops. torch imports live under this subpackage only (loaded
lazily), so the orchestrator + task registry stay torch-free.
"""
import os
import random
from contextlib import contextmanager

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import load_mantis


DEPLOYMENT_BUNDLE_SCHEMA = 'ffm_mantis_embedding_bundle_v1'
TRAINING_STATE_SCHEMA = 'ffm_ssl_training_state_v1'
EMBEDDING_CONTRACT = 'mantis_channel_independent_concat_v1'
PREPROCESSING_CONTRACT = 'per_window_per_channel_zscore_v1'
SHARED_PRICE_PREPROCESSING_CONTRACT = 'per_window_shared_ohlc_zscore_v1'
LOG_PRICE_PREPROCESSING_CONTRACT = 'per_window_log_price_rel_volume_zscore_v1'
SUPPORTED_PREPROCESSING_CONTRACTS = {
    PREPROCESSING_CONTRACT, SHARED_PRICE_PREPROCESSING_CONTRACT,
    LOG_PRICE_PREPROCESSING_CONTRACT}


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _enc(encoder, x1):
    """Encode one channel [B,1,L] -> [B, hidden], interpolating the window to Mantis's native
    seq_len (512) first so it ALWAYS sees its pretrained patch size."""
    L = int(getattr(encoder, 'seq_len', 512))
    if x1.shape[-1] != L:
        x1 = F.interpolate(x1, size=L, mode='linear', align_corners=False)
    return encoder(x1)


def normalization_stats(x, preprocessing=PREPROCESSING_CONTRACT):
    """Past-window-only affine statistics for a versioned OHLCV preprocessing contract."""
    if preprocessing == PREPROCESSING_CONTRACT:
        return x.mean(dim=2, keepdim=True), x.std(dim=2, keepdim=True).clamp_min(1e-6)
    if preprocessing == SHARED_PRICE_PREPROCESSING_CONTRACT:
        if x.ndim != 3 or x.shape[1] < 4:
            raise ValueError('shared OHLC preprocessing requires [B,C>=4,T] input')
        price = x[:, :4, :]
        pm = price.mean(dim=(1, 2), keepdim=True).expand(-1, 4, -1)
        ps = price.std(dim=(1, 2), keepdim=True).clamp_min(1e-6).expand(-1, 4, -1)
        if x.shape[1] == 4:
            return pm, ps
        other = x[:, 4:, :]
        om = other.mean(dim=2, keepdim=True)
        osd = other.std(dim=2, keepdim=True).clamp_min(1e-6)
        return torch.cat((pm, om), dim=1), torch.cat((ps, osd), dim=1)
    if preprocessing == LOG_PRICE_PREPROCESSING_CONTRACT:
        raise ValueError('log-price preprocessing is nonlinear; use preprocess_windows')
    raise ValueError(f'unsupported preprocessing contract: {preprocessing}')


def preprocess_windows(x, preprocessing=PREPROCESSING_CONTRACT, clamp=None):
    if preprocessing == LOG_PRICE_PREPROCESSING_CONTRACT:
        if x.ndim != 3 or x.shape[1] < 4:
            raise ValueError('log-price preprocessing requires [B,C>=4,T] input')
        # One causal reference for every OHLC channel preserves candle geometry. Unlike a
        # per-window z-score, log(price / first close) retains return/volatility amplitude while
        # remaining invariant to the contract's absolute price level.
        base = x[:, 3:4, :1].clamp_min(1e-9)
        price = torch.log(x[:, :4, :].clamp_min(1e-9) / base)
        if x.shape[1] == 4:
            out = price
        else:
            volume = torch.log1p(x[:, 4:, :].clamp_min(0.0))
            volume = ((volume - volume.mean(dim=2, keepdim=True)) /
                      volume.std(dim=2, keepdim=True).clamp_min(1e-6))
            out = torch.cat((price, volume), dim=1)
    else:
        m, s = normalization_stats(x, preprocessing)
        out = (x - m) / s
    return out if clamp is None else out.clamp(-float(clamp), float(clamp))


def preprocess_context_and_future(context, future, preprocessing=PREPROCESSING_CONTRACT,
                                  clamp=None):
    """Apply one past-only transform to a context and its future target.

    The future never contributes statistics. This is the Stage-3 counterpart of
    ``preprocess_windows`` and prevents a nonlinear preprocessing contract from silently falling
    back to the old affine z-score path.
    """
    preprocessing = resolve_preprocessing(preprocessing)
    if preprocessing != LOG_PRICE_PREPROCESSING_CONTRACT:
        m, s = normalization_stats(context, preprocessing)
        ctx, fut = (context - m) / s, (future - m) / s
    else:
        if context.ndim != 3 or context.shape[1] < 4 or future.shape[1] != context.shape[1]:
            raise ValueError('log-price context/future preprocessing requires aligned [B,C>=4,T]')
        base = context[:, 3:4, :1].clamp_min(1e-9)
        cp = torch.log(context[:, :4, :].clamp_min(1e-9) / base)
        fp = torch.log(future[:, :4, :].clamp_min(1e-9) / base)
        if context.shape[1] == 4:
            ctx, fut = cp, fp
        else:
            cv = torch.log1p(context[:, 4:, :].clamp_min(0.0))
            fv = torch.log1p(future[:, 4:, :].clamp_min(0.0))
            vm = cv.mean(dim=2, keepdim=True)
            vs = cv.std(dim=2, keepdim=True).clamp_min(1e-6)
            ctx = torch.cat((cp, (cv - vm) / vs), dim=1)
            fut = torch.cat((fp, (fv - vm) / vs), dim=1)
    if clamp is not None:
        limit = float(clamp)
        ctx, fut = ctx.clamp(-limit, limit), fut.clamp(-limit, limit)
    return ctx, fut


def _standardize(x):
    """Backward-compatible alias for the original per-channel contract."""
    return preprocess_windows(x, PREPROCESSING_CONTRACT)


def encode_independent(encoder, x):
    """Canonical multivariate embedding used by BOTH SSL and deployment.

    Mantis is a univariate encoder. Fold channels into the batch, encode each original channel
    independently, then concatenate in channel order. Keeping this primitive shared prevents a
    train-only channel adapter from silently changing the function exported downstream.
    Input must already follow the declared preprocessing contract.
    """
    B, C, L = x.shape
    emb = _enc(encoder, x.reshape(B * C, 1, L))
    return emb.reshape(B, C * emb.shape[-1])


def make_deployment_bundle(encoder_state, *, model_id, model_version, channels,
                           train_context_lengths=None, preprocessing=PREPROCESSING_CONTRACT):
    """Versioned inference artifact. It contains no optimizer/objective state."""
    return {
        'schema_version': DEPLOYMENT_BUNDLE_SCHEMA,
        'encoder_state': encoder_state,
        'embedding_contract': {
            'name': EMBEDDING_CONTRACT,
            'preprocessing': preprocessing,
            'channel_mode': 'independent_concat',
            'channels': int(channels),
            'native_length': 512,
            'resize': 'linear_align_corners_false',
            'train_context_lengths': ([int(x) for x in train_context_lengths]
                                      if train_context_lengths is not None else None),
        },
        'model': {'id': model_id, 'version': model_version},
    }


def _checkpoint_object(path_or_obj):
    return (torch.load(path_or_obj, map_location='cpu')
            if isinstance(path_or_obj, (str, os.PathLike)) else path_or_obj)


def checkpoint_preprocessing(path_or_obj):
    obj = _checkpoint_object(path_or_obj)
    if isinstance(obj, dict) and obj.get('schema_version') == DEPLOYMENT_BUNDLE_SCHEMA:
        return (obj.get('embedding_contract') or {}).get('preprocessing')
    return None


def resolve_preprocessing(preprocessing=None, checkpoint=None):
    bundled = checkpoint_preprocessing(checkpoint) if checkpoint is not None else None
    if bundled not in (None, *SUPPORTED_PREPROCESSING_CONTRACTS):
        raise ValueError(f'unsupported bundled preprocessing contract: {bundled}')
    if preprocessing is not None and preprocessing not in SUPPORTED_PREPROCESSING_CONTRACTS:
        raise ValueError(f'unsupported preprocessing contract: {preprocessing}')
    if bundled is not None and preprocessing is not None and bundled != preprocessing:
        raise ValueError('requested preprocessing conflicts with deployment bundle contract')
    return bundled or preprocessing or PREPROCESSING_CONTRACT


def encoder_state_from_checkpoint(path_or_obj, *, model_id=None, model_version=None,
                                  expected_channels=None):
    """Load a legacy encoder state_dict or a versioned deployment bundle.

    Training-state checkpoints are intentionally rejected: deployment and exact-resume artifacts
    are separate contracts and must never be accepted interchangeably.
    """
    obj = _checkpoint_object(path_or_obj)
    if not isinstance(obj, dict):
        raise ValueError('checkpoint must be an encoder state_dict or deployment bundle')
    schema = obj.get('schema_version')
    if schema == TRAINING_STATE_SCHEMA:
        raise ValueError('training-state checkpoint cannot be used for deployment; use .bundle.pt')
    if schema == DEPLOYMENT_BUNDLE_SCHEMA:
        contract = obj.get('embedding_contract') or {}
        if contract.get('name') != EMBEDDING_CONTRACT:
            raise ValueError(f"unsupported embedding contract: {contract.get('name')}")
        if contract.get('preprocessing') not in SUPPORTED_PREPROCESSING_CONTRACTS:
            raise ValueError(f"unsupported preprocessing contract: {contract.get('preprocessing')}")
        if expected_channels is not None and int(contract.get('channels', -1)) != int(expected_channels):
            raise ValueError('deployment bundle channel count does not match requested input')
        model = obj.get('model') or {}
        if model_id is not None and model.get('id') not in (None, model_id):
            raise ValueError('deployment bundle model_id mismatch')
        if model_version is not None and model.get('version') not in (None, model_version):
            raise ValueError('deployment bundle model_version mismatch')
        state = obj.get('encoder_state')
        if not isinstance(state, dict):
            raise ValueError('deployment bundle is missing encoder_state')
        return state
    if schema is not None:
        raise ValueError(f'unsupported checkpoint schema: {schema}')
    # Legacy encoder-only state_dict: retain read compatibility, but new runs also emit a bundle.
    return obj


def load_encoder_checkpoint(encoder, path_or_obj, **expected):
    encoder.load_state_dict(encoder_state_from_checkpoint(path_or_obj, **expected))
    return encoder


def _time_shuffle(x):
    """Permute the time axis independently per sample -> destroys temporal order, keeps the exact
    value set. Used for the SHUFFLE control."""
    B, C, T = x.shape
    perm = torch.argsort(torch.rand(B, T, device=x.device), 1)
    return torch.gather(x, 2, perm[:, None, :].expand(B, C, T))


def _apply_control(x, control):
    """Corrupt ONLY the model INPUT per the apples-to-apples control: 'shuffle' scrambles the time
    axis, 'random' replaces with noise, else (real) passes through. The target/trend-key is always
    computed from the REAL context by the caller -> real must beat shuffle/random. Shared by every
    pretext trainer so the control logic lives in one place."""
    if control == 'shuffle':
        return _time_shuffle(x)
    if control == 'random':
        return torch.randn_like(x)
    return x


def _gather_batch(big, starts, b_idx, length):
    """big [T, C] -> windows [B, C, length] for the start positions starts[b_idx]."""
    s = starts[b_idx]                                    # [B]
    rows = s[:, None] + torch.arange(length, device=big.device)[None, :]   # [B, length]
    return big[rows].permute(0, 2, 1).contiguous()       # [B, C, length]


# ----------------------------------------------------------------- frozen embedding (probe / cache)
@torch.no_grad()
def embed_encoder(big, starts, seq, *, ckpt=None, model_id='paris-noah/Mantis-8M',
                  model_version=None,
                  device=None, batch=512, max_windows=20000, seed=0, preprocessing=None):
    """Frozen ENCODER-ONLY embeddings of clean (per-window z-scored) windows — the quantity that
    transfers downstream via backbone_ckpt. Each OHLCV channel is encoded independently and
    concatenated -> [M, C*hidden]. ckpt=None -> vanilla Mantis (probe baseline)."""
    dev = device or ('cuda' if torch.cuda.is_available()
                     else 'mps' if torch.backends.mps.is_available() else 'cpu')
    enc = load_mantis(model_id, model_version=model_version, device='cpu')
    artifact = _checkpoint_object(ckpt) if ckpt else None
    preprocessing = resolve_preprocessing(preprocessing, artifact)
    if ckpt:
        load_encoder_checkpoint(enc, artifact, model_id=model_id, model_version=model_version,
                                expected_channels=int(np.asarray(big).shape[1]))
    enc = enc.to(dev).eval()
    big_t = torch.as_tensor(np.asarray(big, np.float32), device=dev)
    s = np.asarray(starts, np.int64)
    if len(s) > max_windows:
        s = np.sort(np.random.default_rng(seed).choice(s, max_windows, replace=False))
    s_t = torch.as_tensor(s, device=dev)
    out = []
    for b in range(0, len(s_t), batch):
        win = _gather_batch(big_t, s_t, torch.arange(b, min(b + batch, len(s_t)), device=dev), seq)
        win = preprocess_windows(win, preprocessing)     # [B, C, seq]
        emb = encode_independent(enc, win)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32), s


@torch.no_grad()
def embed_windows(windows, *, ckpt=None, model_id='paris-noah/Mantis-8M', model_version=None,
                  device=None, batch=512, preprocessing=None):
    """Frozen ENCODER-ONLY embeddings of pre-extracted windows [N, C, seq] -> [N, C*hidden]. The
    head-only/cached downstream primitive: backbone frozen, embed ONCE, then a cheap head trains
    on the cache."""
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    dev = device or ('cuda' if torch.cuda.is_available()
                     else 'mps' if torch.backends.mps.is_available() else 'cpu')
    enc = load_mantis(model_id, model_version=model_version, device='cpu')
    artifact = _checkpoint_object(ckpt) if ckpt else None
    preprocessing = resolve_preprocessing(preprocessing, artifact)
    if ckpt:
        load_encoder_checkpoint(enc, artifact, model_id=model_id, model_version=model_version,
                                expected_channels=int(np.asarray(windows).shape[1]))
    enc = enc.to(dev).eval()
    X = torch.as_tensor(np.asarray(windows, np.float32))
    out = []
    for b in range(0, len(X), batch):
        w = preprocess_windows(X[b:b + batch].to(dev), preprocessing)
        emb = encode_independent(enc, w)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


class _EncoderONNX(nn.Module):
    """ONNX-exportable wrapper reproducing embed_windows EXACTLY: per-window standardize ->
    per-channel interpolate to native length -> encode -> concat. Raw window [B,C,seq] in,
    embedding [B, C*hidden] out (standardize baked in so the bot feeds RAW OHLCV).

    Channels are folded into the BATCH ([B,C,seq] -> [B*C,1,seq]) so the encoder appears ONCE
    in the traced graph. The per-channel python loop traced C copies of the transformer into
    the ONNX (5x nodes, defeated ORT fusion — profiled at 22% Transpose / 57% shape-plumbing).
    reshape(B, C*hidden) on the [B*C, hidden] output reproduces cat(dim=-1) block order exactly."""

    def __init__(self, encoder, C, preprocessing=PREPROCESSING_CONTRACT):
        super().__init__()
        self.encoder = encoder
        self.C = int(C)
        self.preprocessing = resolve_preprocessing(preprocessing)

    def forward(self, w):                                     # [B, C, seq] raw OHLCV
        w = preprocess_windows(w, self.preprocessing)
        return encode_independent(self.encoder, w)


def _ort_optimize_graph(path):
    """Offline onnxruntime graph optimization: constant-fold the tracer's shape-plumbing
    (Shape/Gather/Unsqueeze/Constant), eliminate redundant Transposes, fuse LayerNorm/attention.
    Saves the optimized graph over `path`. EXTENDED level emits some ORT-specific fused ops —
    fine for our serve path (the bot runs onnxruntime), and it's the level that kills the
    Transpose overhead. Best-effort: on any failure the un-optimized (still correct) file stands."""
    try:
        import tempfile

        import onnxruntime as ort
        fd, tmp = tempfile.mkstemp(suffix='.onnx', dir=os.path.dirname(os.path.abspath(path)))
        os.close(fd)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        so.optimized_model_filepath = tmp
        ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        os.replace(tmp, path)
        print(f"[onnx] ORT-optimized encoder graph saved -> {path}", flush=True)
    except Exception as e:                                    # pragma: no cover
        print(f"[onnx] ORT offline optimization skipped: {e}", flush=True)


def export_encoder_onnx(path, *, ckpt=None, C=5, seq=64,
                        model_id='paris-noah/Mantis-8M', model_version=None, device='cpu',
                        preprocessing=None):
    """Export the frozen encoder (standardize+interp+encode) to ONNX: raw window [B,C,seq] ->
    embedding [B, C*hidden]. Matches embed_windows numerically (parity-tested).

    The graph holds ONE encoder (channels batched, see _EncoderONNX) and is post-processed by
    onnxruntime offline optimization (constant folding + transpose elimination + fusion)."""
    enc = load_mantis(model_id, model_version=model_version, device='cpu')
    artifact = _checkpoint_object(ckpt) if ckpt else None
    preprocessing = resolve_preprocessing(preprocessing, artifact)
    if ckpt:
        load_encoder_checkpoint(enc, artifact, model_id=model_id, model_version=model_version,
                                expected_channels=int(C))
    enc = enc.to(device).eval()
    m = _EncoderONNX(enc, C, preprocessing=preprocessing).to(device).eval()
    dummy = torch.randn(2, int(C), int(seq), device=device)   # >1 row so std is well-defined
    # Mantis calls torch.diff internally (aten::diff has no ONNX symbolic) -> swap for an
    # equivalent slice-subtract during export so the traced graph is exportable. Restored after.
    _orig_diff = torch.diff

    def _diff_traceable(x, n=1, dim=-1, *, axis=None, prepend=None, append=None):
        d = axis if axis is not None else dim
        for _ in range(int(n)):
            x = x.narrow(d, 1, x.size(d) - 1) - x.narrow(d, 0, x.size(d) - 1)
        return x
    torch.diff = _diff_traceable
    try:
        torch.onnx.export(m, dummy, path, input_names=['window'], output_names=['embedding'],
                          dynamic_axes={'window': {0: 'batch'}, 'embedding': {0: 'batch'}},
                          do_constant_folding=True,
                          opset_version=17, dynamo=False)      # legacy tracer (dynamo chokes on Mantis)
    finally:
        torch.diff = _orig_diff
    _ort_optimize_graph(path)
    return path


def _atomic_save(obj, path):
    """Crash-safe save: write to a temp file then os.replace (atomic) -> a Colab disconnect can
    never leave a half-written checkpoint."""
    tmp = str(path) + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_meta(path, best_val, epoch):
    import json
    tmp = str(path) + '.meta.json.tmp'
    with open(tmp, 'w') as f:
        json.dump({'best_val': float(best_val), 'epoch': int(epoch)}, f)
    os.replace(tmp, str(path) + '.meta.json')


def _read_meta_best(path):
    import json
    mp = str(path) + '.meta.json'
    if os.path.exists(mp):
        try:
            return float(json.load(open(mp)).get('best_val', 1e18))
        except Exception:
            return 1e18
    return 1e18


def _freeze_encoder(encoder, n_layers):
    """Anti-forgetting: freeze the input tokenizer + the first n_layers transformer blocks of a
    Mantis encoder when REFINING a warm-started encoder, so the bulk of learned structure can't
    drift. Later blocks and the objective head stay trainable, so the embedding can still adapt.
    n_layers<=0 -> no freeze. Robust to V1 (vit_unit) / V2 (transf_unit) paths."""
    if not n_layers or int(n_layers) <= 0:
        return 0
    n = int(n_layers)
    tok = getattr(encoder, 'tokgen_unit', None)              # input patch/scalar tokenizer (general)
    if tok is not None:
        for p in tok.parameters():
            p.requires_grad = False
    unit = getattr(encoder, 'vit_unit', None) or getattr(encoder, 'transf_unit', None)
    tr = getattr(unit, 'transformer', None) if unit is not None else None
    layers = getattr(tr, 'layers', None) if tr is not None else None
    frozen = 0
    if layers is not None:
        for blk in list(layers)[:n]:
            for p in blk.parameters():
                p.requires_grad = False
            frozen += 1
    return frozen


# ================================================================= BASE TRAINER (shared loop)
class BaseTrainer:
    """Shared SSL training loop for every pretext. Subclass and implement:
      * build_net()            -> set self.net (+ warm-start from backbone_ckpt)
      * make_batch(starts)     -> a batch object for compute_loss / accumulated over steps
      * compute_loss(batch)    -> scalar loss (called inside the AMP context)
      * val_eval()             -> (val_loss: float, extra: dict incl 'std' for the collapse guard)
    fit() owns: the epoch/step loop, AMP (cuda) + grad-clip, cosine LR, early-stop on val_loss,
    and the best-ENCODER-state snapshot. Subclasses may override make_optimizer / log_line."""

    def __init__(self, big, train_starts, val_starts, *, epochs=60, steps_per_epoch=200, batch=512,
                 lr=1e-4, weight_decay=0.05, patience=8, device=None, seed=0, grad_clip=None,
                 amp=True, amp_dtype='fp16', verbose=True, control='real',
                 ckpt_path=None, resume=False, freeze_encoder_layers=0, std_guard=0.0,
                 train_group_bounds=None, val_group_bounds=None, stop_after_epoch=None,
                 val_batches=None):
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        self.ckpt_path, self.resume = ckpt_path, resume    # progressive best-save + resume (real run only)
        self.seed = int(seed)
        self.stop_after_epoch = stop_after_epoch           # deterministic interruption hook for tests
        self.val_batches = None if val_batches is None else int(val_batches)
        self.freeze_encoder_layers = freeze_encoder_layers  # anti-forgetting: freeze first N enc layers
        self.std_guard = float(std_guard or 0.0)           # >0: HALT when emb_std exceeds it (drift guard)
        self.dev = device or ('cuda' if torch.cuda.is_available()
                              else 'mps' if torch.backends.mps.is_available() else 'cpu')
        torch.manual_seed(seed)
        self.gen = torch.Generator(device=self.dev); self.gen.manual_seed(seed)
        self.big_t = torch.as_tensor(np.asarray(big, np.float32), device=self.dev)
        self.tr = torch.as_tensor(np.asarray(train_starts, np.int64), device=self.dev)
        self.va = torch.as_tensor(np.asarray(val_starts, np.int64), device=self.dev)
        self.tr_groups = self._validate_group_bounds(train_group_bounds, len(self.tr), 'train')
        self.va_groups = self._validate_group_bounds(val_group_bounds, len(self.va), 'validation')
        self.epochs, self.steps_per_epoch, self.batch = epochs, steps_per_epoch, batch
        self.lr, self.weight_decay, self.patience = lr, weight_decay, patience
        self.grad_clip, self.verbose, self.control = grad_clip, verbose, control
        self.use_amp = (self.dev == 'cuda') and amp                # contrastive runs fp32 (amp=False)
        _adt = torch.float16 if str(amp_dtype).lower() in ('fp16', 'float16') else torch.bfloat16
        self.amp_ctx = (lambda: torch.autocast('cuda', dtype=_adt)) if self.use_amp else (lambda: _nullctx())
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.net = None

    def _validate_group_bounds(self, bounds, n, name):
        """Validate contiguous [lo,hi) stream slices used for macro-balanced sampling."""
        if bounds is None:
            return None
        arr = np.asarray(bounds, np.int64).reshape(-1, 2)
        if len(arr) == 0 or arr[0, 0] != 0 or arr[-1, 1] != n:
            raise ValueError(f"{name} group bounds must cover [0,{n}) exactly")
        if np.any(arr[:, 0] >= arr[:, 1]) or np.any(arr[1:, 0] != arr[:-1, 1]):
            raise ValueError(f"{name} group bounds must be non-empty and contiguous")
        return torch.as_tensor(arr, dtype=torch.long, device=self.dev)

    def _groups_for(self, starts):
        return self.tr_groups if starts is self.tr else self.va_groups if starts is self.va else None

    def sample_start_indices(self, starts, *, generator=None, return_groups=False, count=None):
        """Sample streams uniformly, then windows uniformly inside each selected stream.

        Without group metadata this preserves the legacy flat-window sampler.  Equal stream
        probability prevents dense 1-minute data from drowning out 30/60-minute examples.
        """
        gen = generator or self.gen
        count = self.batch if count is None else int(count)
        bounds = self._groups_for(starts)
        if bounds is None:
            idx = torch.randint(0, len(starts), (count,), device=self.dev, generator=gen)
            groups = torch.zeros(count, dtype=torch.long, device=self.dev)
        else:
            groups = torch.randint(0, len(bounds), (count,), device=self.dev, generator=gen)
            lo, hi = bounds[groups, 0], bounds[groups, 1]
            u = torch.rand(count, device=self.dev, generator=gen)
            idx = lo + torch.floor(u * (hi - lo).float()).long()
        return (idx, groups) if return_groups else idx

    # ---- hooks (subclass implements) ----
    def build_net(self):
        raise NotImplementedError

    def make_batch(self, starts):
        raise NotImplementedError

    def compute_loss(self, batch):
        raise NotImplementedError

    def val_eval(self):
        raise NotImplementedError

    def make_optimizer(self):
        return torch.optim.AdamW([p for p in self.net.parameters() if p.requires_grad],
                                 lr=self.lr, weight_decay=self.weight_decay)

    def log_line(self, ep, tr_loss, vloss, extra, improved):
        if self.verbose:
            print(f"  ep{ep:>3} train={tr_loss:.4f} val={vloss:.4f} "
                  f"emb_std={extra.get('std', 0.0):.4f}{'  *' if improved else ''}", flush=True)

    def _encoder(self):
        net = self.net
        return net.encoder if not hasattr(net, '_orig_mod') else net._orig_mod.encoder

    def _model(self):
        return self.net if not hasattr(self.net, '_orig_mod') else self.net._orig_mod

    def _params(self):
        return [p for p in self.net.parameters() if p.requires_grad]

    @contextmanager
    def fixed_validation_rng(self, seed=20260704):
        """Make stochastic validation identical at every epoch without changing training RNG.

        Validation for the mask/forecast/discriminative tasks samples windows (and sometimes
        corruptions) on demand.  Re-seeding those draws makes checkpoint selection compare the
        same validation experiment each epoch.  All RNG states are restored afterwards so this
        does not alter the subsequent training trajectory.
        """
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        gen_state = self.gen.get_state()
        old_nprng = getattr(self, '_nprng', None)
        torch.manual_seed(int(seed))
        self.gen.manual_seed(int(seed))
        if old_nprng is not None:
            self._nprng = np.random.default_rng(int(seed))
        try:
            yield
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            self.gen.set_state(gen_state)
            if old_nprng is not None:
                self._nprng = old_nprng

    def _resume_signature(self):
        return {'epochs': int(self.epochs), 'steps_per_epoch': int(self.steps_per_epoch),
                'batch': int(self.batch), 'lr': float(self.lr),
                'weight_decay': float(self.weight_decay), 'patience': int(self.patience),
                'freeze_encoder_layers': int(self.freeze_encoder_layers or 0),
                'control': self.control,
                'preprocessing': getattr(self, 'preprocessing', PREPROCESSING_CONTRACT),
                'val_batches': self.val_batches}

    def _rng_state(self):
        state = {'torch_cpu': torch.random.get_rng_state(), 'trainer_generator': self.gen.get_state(),
                 'python': random.getstate(), 'numpy_global': np.random.get_state()}
        if torch.cuda.is_available():
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
        if hasattr(self, '_nprng'):
            state['numpy_generator'] = self._nprng.bit_generator.state
        return state

    def _restore_rng_state(self, state):
        torch.random.set_rng_state(state['torch_cpu'])
        self.gen.set_state(state['trainer_generator'])
        random.setstate(state['python'])
        np.random.set_state(state['numpy_global'])
        if 'torch_cuda' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['torch_cuda'])
        if 'numpy_generator' in state and hasattr(self, '_nprng'):
            self._nprng.bit_generator.state = state['numpy_generator']

    def _training_state_path(self):
        return None if not self.ckpt_path else str(self.ckpt_path) + '.train.pt'

    def fit(self):
        """Run the shared loop and keep two deliberately separate artifacts.

        ``ckpt_path`` is the best encoder-only deployment weight file (legacy-compatible).
        ``ckpt_path.train.pt`` is the latest full epoch-boundary training state and is the ONLY
        artifact accepted by ``resume=True``. It restores the objective head, optimizer,
        scheduler, AMP scaler, epoch, early-stop counters, history, and RNG streams.
        """
        from ...native_training_routes import block_unadmitted_optimizer
        block_unadmitted_optimizer(
            "futures_foundation.finetune.pretext._torch.common.BaseTrainer.fit"
        )
        self.build_net()
        save_ok = bool(self.ckpt_path) and self.control == 'real'   # controls never touch artifacts
        if self.resume and not save_ok:
            raise ValueError('exact resume requires a real run with ckpt_path')
        nfz = _freeze_encoder(self._encoder(), self.freeze_encoder_layers)
        if nfz and self.verbose:
            ntr = sum(p.requires_grad for p in self.net.parameters())
            print(f"  [freeze] tokenizer + first {nfz} encoder layers frozen ({ntr} trainable tensors)",
                  flush=True)
        opt = self.make_optimizer()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        best, best_state, best_epoch = 1e18, None, -1
        bad, history, start_epoch = 0, [], 0
        train_path = self._training_state_path()
        if self.resume:
            if not os.path.exists(train_path):
                raise FileNotFoundError(
                    f'exact resume state missing: {train_path}; encoder-only checkpoints are '
                    'warm starts, not resumable training state')
            # This is a locally produced full training artifact containing Python/NumPy RNG
            # tuples in addition to tensors. It is never accepted through the deployment loader.
            saved = torch.load(train_path, map_location='cpu', weights_only=False)
            if saved.get('schema_version') != TRAINING_STATE_SCHEMA:
                raise ValueError('unsupported SSL training-state checkpoint')
            if saved.get('trainer_signature') != self._resume_signature():
                raise ValueError('resume configuration differs from the saved training trajectory')
            self._model().load_state_dict(saved['model_state'])
            opt.load_state_dict(saved['optimizer_state'])
            sched.load_state_dict(saved['scheduler_state'])
            self.scaler.load_state_dict(saved['scaler_state'])
            best = float(saved['best_val'])
            best_epoch = int(saved['best_epoch'])
            best_state = saved['best_encoder_state']
            bad = int(saved['bad_epochs'])
            history = list(saved['history'])
            start_epoch = int(saved['epoch']) + 1
            self._restore_rng_state(saved['rng_state'])
            # Repair a stale/missing deployment weight if a crash occurred between atomic saves.
            _atomic_save(best_state, self.ckpt_path)
            _write_meta(self.ckpt_path, best, best_epoch)
            if self.verbose:
                print(f"  [resume] restored full state after epoch {start_epoch - 1} "
                      f"(best_val={best:.4f})", flush=True)
        for ep in range(start_epoch, self.epochs):
            self.net.train(); tr_tot = 0.0
            for _ in range(self.steps_per_epoch):
                opt.zero_grad(set_to_none=True)
                with self.amp_ctx():
                    loss = self.compute_loss(self.make_batch(self.tr))
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip:
                        self.scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(self._params(), self.grad_clip)
                    self.scaler.step(opt); self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self._params(), self.grad_clip)
                    opt.step()
                tr_tot += float(loss.detach())
            sched.step()
            if self.dev == 'cuda':
                torch.cuda.empty_cache()
            vloss, extra = self.val_eval()
            history.append({'epoch': ep, 'train_loss': tr_tot / self.steps_per_epoch,
                            'val_loss': vloss, **extra})
            if self.std_guard and extra.get('std', 0.0) > self.std_guard:
                # DRIFT GUARD: embedding std past the ceiling = the representation is drifting off
                # the data (the unanchored-discrimination failure mode). HALT NOW and do NOT save
                # this epoch — val often keeps micro-improving while drift bakes in, so waiting for
                # early-stop would keep crowning drifted epochs as "best".
                if self.verbose:
                    print(f"  [std-guard] emb_std {extra['std']:.3f} > {self.std_guard:.2f} at "
                          f"ep{ep} — HALTED (best checkpoint kept from before the breach)",
                          flush=True)
                break
            improved = vloss < best - 1e-5
            if improved:
                best, bad = vloss, 0
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in self._encoder().state_dict().items()}
            else:
                bad += 1
            if save_ok:
                full_state = {
                    'schema_version': TRAINING_STATE_SCHEMA,
                    'epoch': int(ep), 'best_epoch': int(best_epoch), 'best_val': float(best),
                    'bad_epochs': int(bad), 'history': history,
                    'trainer_signature': self._resume_signature(),
                    'model_state': {k: v.detach().cpu().clone()
                                    for k, v in self._model().state_dict().items()},
                    'best_encoder_state': best_state,
                    'optimizer_state': opt.state_dict(), 'scheduler_state': sched.state_dict(),
                    'scaler_state': self.scaler.state_dict(), 'rng_state': self._rng_state(),
                }
                # Full state first. If interrupted between saves, resume repairs best weights.
                _atomic_save(full_state, train_path)
                if improved:
                    _atomic_save(best_state, self.ckpt_path)
                    _write_meta(self.ckpt_path, best, best_epoch)
            self.log_line(ep, tr_tot / self.steps_per_epoch, vloss, extra, improved)
            if self.stop_after_epoch is not None and ep >= int(self.stop_after_epoch):
                break
            if bad >= self.patience:
                break
        return best_state, history
