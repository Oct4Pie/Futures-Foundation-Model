"""Backward-compat SHIM — the torch SSL layer now lives under `pretext/_torch/`.

The per-pretext trainers (mask / forecast / contrastive) each got their own module under a shared
BaseTrainer, and the window/embedding/ONNX helpers moved to `pretext/_torch/common.py`. This module
re-exports everything so existing references keep working (ssl_probe.embed_encoder, classifiers'
embed_windows / export_encoder_onnx, tests' train_ssl_*, ContrastiveTrendNet, _future_key, ...).

New code should import from `futures_foundation.finetune.pretext._torch` directly. Torch loads only
when THIS module (or that subpackage) is imported — which is always lazy — so the orchestrator and
pretext task registry stay torch-free.
"""
from .pretext._torch.common import (                      # noqa: F401  window/embedding/ONNX + base
    _nullctx, _enc, _standardize, normalization_stats, preprocess_windows,
    preprocess_context_and_future,
    resolve_preprocessing, _time_shuffle, _apply_control, _gather_batch,
    embed_encoder, embed_windows, _EncoderONNX, export_encoder_onnx, BaseTrainer)
from .pretext._torch.mask import MaskNetwork, train_ssl_mask                             # noqa: F401
from .pretext._torch.structure_mask import (                                             # noqa: F401
    StructuralMaskNetwork, structural_targets, train_ssl_structure_mask)
from .pretext._torch.forecast import MultiHorizonForecastNet, train_ssl_forecast         # noqa: F401
from .pretext._torch.path import (                                                        # noqa: F401
    PathNetwork, wall_clock_steps, path_targets, path_loss, train_ssl_path)
from .pretext._torch.forecast_dist import (                # noqa: F401  stage-2.5 distributional refine
    train_ssl_forecast_dist, get_dist_objective, CandleQuantile, CandleBins)
from .pretext._torch.contrastive import (                 # noqa: F401  stage-2 temporal regime geometry
    ContrastiveTrendNet, train_ssl_contrastive, regime_gate, _random_crop_resize)
from .pretext._torch.nextleg import NextLegNet, train_ssl_nextleg              # noqa: F401
from .pretext._torch.electra import (                     # noqa: F401  stage-4 turn-electra (replaced-TURN)
    ElectraNetwork, train_ssl_electra, clamp_valid_ohlc_t)
