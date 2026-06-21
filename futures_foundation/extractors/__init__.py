"""Swappable feature-extractor interface.

The pipeline depends ONLY on this package's interface (FeatureExtractor +
evaluate_with_extractor). Each concrete backbone lives in its own subpackage and
implements FeatureExtractor; swap by name via get_extractor(). Chronos is the
only backbone today (MOMENT was evaluated and dropped — no lift); new models drop
in as `extractors/<name>/` + a registry entry.
"""
from .base import FeatureExtractor, evaluate_with_extractor, _windows
from .chronos import ChronosExtractor

_REGISTRY = {'chronos': ChronosExtractor}


def get_extractor(name: str = 'chronos', **kw) -> FeatureExtractor:
    """Resolve an extractor by name (only 'chronos' today)."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown extractor {name!r}; have {list(_REGISTRY)}")
    return _REGISTRY[name](**kw)


__all__ = ['FeatureExtractor', 'ChronosExtractor', 'get_extractor',
           'evaluate_with_extractor']
