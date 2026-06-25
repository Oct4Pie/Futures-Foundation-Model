"""Unit test for the embed-worker device resolution (torch-free)."""
from futures_foundation.extractors.chronos._worker import _resolve_embed_device


def test_default_cpu_stays_cpu():
    assert _resolve_embed_device('cpu', mps_available=False) == 'cpu'
    assert _resolve_embed_device('cpu', mps_available=True) == 'cpu'


def test_mps_honored_only_when_available():
    assert _resolve_embed_device('mps', mps_available=True) == 'mps'


def test_mps_falls_back_to_cpu_when_unavailable():
    # parity-safe: never silently run on a device that isn't there
    assert _resolve_embed_device('mps', mps_available=False) == 'cpu'


def test_other_device_passthrough():
    assert _resolve_embed_device('cuda', mps_available=False) == 'cuda'
