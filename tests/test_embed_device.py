"""Unit test for the embed-worker device resolution (torch-free)."""
from futures_foundation.extractors.chronos._worker import _resolve_embed_device


def test_default_cpu_stays_cpu():
    assert _resolve_embed_device('cpu', mps_available=False) == 'cpu'
    assert _resolve_embed_device('cpu', mps_available=True, cuda_available=True) == 'cpu'


def test_mps_honored_only_when_available():
    assert _resolve_embed_device('mps', mps_available=True) == 'mps'
    assert _resolve_embed_device('mps', mps_available=False) == 'cpu'


def test_cuda_honored_only_when_available():
    assert _resolve_embed_device('cuda', mps_available=False, cuda_available=True) == 'cuda'
    assert _resolve_embed_device('cuda', mps_available=False, cuda_available=False) == 'cpu'


def test_auto_prefers_cuda_then_mps_then_cpu():
    assert _resolve_embed_device('auto', mps_available=True, cuda_available=True) == 'cuda'
    assert _resolve_embed_device('auto', mps_available=True, cuda_available=False) == 'mps'
    assert _resolve_embed_device('auto', mps_available=False, cuda_available=False) == 'cpu'
