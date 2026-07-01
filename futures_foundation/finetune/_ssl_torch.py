"""Backward-compat SHIM — the torch SSL layer now lives under `pretext/_torch/`.

The per-pretext trainers (mask / forecast / contrastive) each got their own module under a shared
BaseTrainer, and the window/embedding/ONNX helpers moved to `pretext/_torch/common.py`. This module
re-exports everything so existing references keep working (ssl_probe.embed_encoder, classifiers'
embed_windows / export_encoder_onnx, tests' train_ssl_*, ContrastiveTrendNet, _trend_key, ...).

New code should import from `futures_foundation.finetune.pretext._torch` directly. Torch loads only
when THIS module (or that subpackage) is imported — which is always lazy — so the orchestrator and
pretext task registry stay torch-free.
"""
from .pretext._torch.common import (                      # noqa: F401  window/embedding/ONNX + base
    _nullctx, _enc, _standardize, _time_shuffle, _apply_control, _gather_batch,
    embed_encoder, embed_windows, _EncoderONNX, export_encoder_onnx, BaseTrainer)
from .pretext._torch.mask import MaskNetwork, train_ssl_mask                             # noqa: F401
from .pretext._torch.forecast import MultiHorizonForecastNet, train_ssl_forecast         # noqa: F401
from .pretext._torch.contrastive import (                 # noqa: F401
    ContrastiveTrendNet, train_ssl_contrastive, _trend_key, _multi_positive_infonce,
    _random_crop_resize)
