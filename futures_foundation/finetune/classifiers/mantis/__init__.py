"""Mantis backbone — concrete Classifier implementations (torch-bearing, loaded lazily via
`finetune.classifier.get_classifier`). Importing this package self-registers the backbone's
classifiers ('mantis', 'mantis_frozen') and exposes its default SSL foundation checkpoint.
"""
from .classifier import MantisClassifier          # noqa: F401  registers 'mantis'
from .frozen import MantisFrozenClassifier         # noqa: F401  registers 'mantis_frozen'

# This backbone's default SSL foundation (the checkpoint a new strategy finetunes on top of).
# Single source of truth for the Mantis base — generic code reads it via the DI accessor, never
# by importing this constant.
BASE_CKPT = 'checkpoints/mantis_ssl_nextleg.pt'   # stage-2.6 next-leg — PROMOTED 2026-07-16 (beat ctr_seq2seq on all 3 gates: 6/6 scorecard, 10/10 dry-run op points, mini 3/4)
