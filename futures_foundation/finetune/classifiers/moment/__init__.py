"""MOMENT backbone (stub) — proves the plugin/DI drop-in: same Classifier interface as Mantis,
imported lazily via the manifest, consumed by the generic harness/calibration/risk_head unchanged.
Not yet implemented (see frozen.py). MOMENT = the moment-timeseries-foundation-model transformer.
"""
from .frozen import MomentFrozenClassifier      # noqa: F401  registers 'moment_frozen'

# This backbone's default foundation checkpoint (placeholder until MOMENT is wired in).
BASE_CKPT = 'checkpoints/moment_base.pt'
