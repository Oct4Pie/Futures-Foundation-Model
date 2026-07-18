import numpy as np
import pytest

from futures_foundation.finetune.chronos_family import (
    CANDIDATES, QUANTILE_LEVELS, benchmark_signature, evaluate_close_forecasts,
    persistence_quantiles, resolve_candidates,
)
from scripts.benchmark_chronos_family import _normalize_quantiles, _parse_checkpoint_map


def _windows(n=12, context=16, horizon=4):
    rng = np.random.default_rng(4)
    base = np.linspace(90, 110, n)
    contexts = np.empty((n, context, 5), np.float32)
    futures = np.empty((n, horizon, 5), np.float32)
    for row in range(n):
        past = base[row] * np.exp(np.linspace(-0.01, 0, context))
        future = base[row] * np.exp(np.cumsum(rng.normal(0.001, 0.004, horizon)))
        contexts[row] = np.stack(
            [past, past + .2, past - .2, past, np.full(context, 100)], axis=1,
        )
        futures[row] = np.stack(
            [future, future + .2, future - .2, future, np.full(horizon, 100)], axis=1,
        )
    return {
        "context": contexts, "future": futures,
        "ticker": np.array(["ES"] * (n // 2) + ["NQ"] * (n - n // 2)),
        "timeframe": np.array(["1min"] * n),
    }


def test_candidates_are_pinned_capacity_bounded_and_capability_labeled():
    selected = resolve_candidates(("chronos_v1", "chronos_bolt", "chronos_v2"))
    assert [candidate.key for candidate in selected] == list(CANDIDATES)
    assert all(len(candidate.revision) == 40 for candidate in selected)
    assert max(candidate.parameters for candidate in selected) < 30_000_000
    assert CANDIDATES["chronos_v1"].public_embedding_api
    assert CANDIDATES["chronos_v2"].native_multivariate
    with pytest.raises(ValueError, match="unknown"):
        resolve_candidates(("chronos_v3",))
    with pytest.raises(ValueError, match="unique"):
        resolve_candidates(("chronos_v1", "chronos_v1"))


def test_signature_binds_model_data_and_inference_config():
    base = {"model": "v1", "window": "abc", "samples": 20}
    assert benchmark_signature(base) == benchmark_signature(dict(base))
    assert benchmark_signature(base) != benchmark_signature({**base, "samples": 21})


def test_perfect_close_quantiles_score_perfect_point_forecast():
    windows = _windows()
    actual = windows["future"][:, :, 3]
    quantiles = np.repeat(actual[:, :, None], len(QUANTILE_LEVELS), axis=2)
    result = evaluate_close_forecasts(windows, quantiles)
    assert result["overall"]["path_log_return_mse"] == pytest.approx(0.0, abs=1e-14)
    assert result["overall"]["path_skill_vs_persistence"] == pytest.approx(1.0)
    assert result["overall"]["fwd_absmove_r2"] == pytest.approx(1.0)
    assert result["overall"]["vol_r2"] == pytest.approx(1.0)
    assert result["macro_stream"]["stream_count"] == 2
    assert result["diagnostic_gate"]["all_pass"]


def test_persistence_is_exact_zero_skill_baseline():
    windows = _windows()
    quantiles = persistence_quantiles(windows)
    result = evaluate_close_forecasts(windows, quantiles)
    assert result["overall"]["path_skill_vs_persistence"] == pytest.approx(0.0)
    assert not result["diagnostic_gate"]["macro_path_skill_positive"]


def test_official_pipeline_outputs_normalize_without_family_shape_leak():
    torch = pytest.importorskip("torch")
    base = torch.ones(2, 4, 3)
    normalized = _normalize_quantiles(
        CANDIDATES["chronos_bolt"], base, 2, 4, 3,
    )
    assert normalized.shape == (2, 4, 3)
    joint = [torch.stack([torch.full((4, 3), channel) for channel in range(5)])
             for _ in range(2)]
    normalized = _normalize_quantiles(
        CANDIDATES["chronos_v2"], joint, 2, 4, 3, close_channel=3,
    )
    np.testing.assert_array_equal(normalized, 3.0)
    with pytest.raises(ValueError, match="one tensor per input"):
        _normalize_quantiles(CANDIDATES["chronos_v2"], joint[:1], 2, 4, 3)


def test_trained_checkpoint_map_is_explicit_and_unique():
    assert _parse_checkpoint_map("chronos_v1=/a.pt,chronos_v2=/b.pt") == {
        "chronos_v1": "/a.pt", "chronos_v2": "/b.pt",
    }
    with pytest.raises(ValueError, match="candidate=path"):
        _parse_checkpoint_map("/a.pt")
    with pytest.raises(ValueError, match="unknown"):
        _parse_checkpoint_map("moment=/a.pt")
    with pytest.raises(ValueError, match="unique"):
        _parse_checkpoint_map("chronos_v1=/a.pt,chronos_v1=/b.pt")
