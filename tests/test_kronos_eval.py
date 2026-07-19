import numpy as np
import pandas as pd
import pytest

from futures_foundation.finetune.kronos_eval import (
    build_forecast_windows, evaluate_predictions, validate_eval_period, window_fingerprint,
)
from scripts.benchmark_kronos import _context_frame


def _write_stream(path):
    first = pd.date_range("2024-06-28", periods=80, freq="1min", tz="UTC")
    second = pd.date_range("2024-07-01", periods=120, freq="1min", tz="UTC")
    ts = first.append(second)
    close = np.linspace(100, 120, len(ts))
    contract = np.array(["M24"] * len(first) + ["U24"] * len(second))
    pd.DataFrame({
        "datetime": ts, "open": close - 0.1, "high": close + 0.3,
        "low": close - 0.3, "close": close, "volume": 100,
        "contract_id": contract,
    }).to_csv(path, index=False)


def test_kronos_period_rejects_published_pretraining_interval():
    with pytest.raises(ValueError, match="pretraining data extends through June 2024"):
        validate_eval_period("2024-06-01", "2025-01-01")
    start, end = validate_eval_period("2024-07-01", "2025-01-01")
    assert start < end and start.month == 7


def test_windows_are_causal_roll_safe_and_content_bound(tmp_path):
    _write_stream(tmp_path / "ES_1min.csv")
    windows = build_forecast_windows(
        tmp_path, ("ES",), ("1min",), context=16, horizon=4,
        eval_start="2024-07-01", eval_end="2024-07-02", max_per_stream=8,
        separation_bars=4, seed=7, chunksize=31, verbose=False,
    )
    assert windows["context"].shape == (8, 16, 5)
    assert windows["future"].shape == (8, 4, 5)
    assert np.all(windows["context_time_ns"][:, -1] < windows["future_time_ns"][:, 0])
    assert pd.to_datetime(windows["future_time_ns"].min(), utc=True) >= pd.Timestamp(
        "2024-07-01", tz="UTC")
    # Adjacent future labels are separated by the full horizon.
    ordered = np.sort(windows["future_time_ns"][:, 0])
    assert np.all(np.diff(ordered) >= pd.Timedelta("4min").value)
    assert window_fingerprint(windows) == window_fingerprint(windows)
    changed = dict(windows)
    changed["future"] = windows["future"].copy()
    changed["future"][0, 0, 3] += 1
    assert window_fingerprint(changed) != window_fingerprint(windows)


def test_windows_reject_stream_without_contract_identity(tmp_path):
    path = tmp_path / "ES_1min.csv"
    _write_stream(path)
    frame = pd.read_csv(path).drop(columns=["contract_id"])
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="contract_id"):
        build_forecast_windows(
            tmp_path, ("ES",), ("1min",), context=16, horizon=4,
            eval_start="2024-07-01", eval_end="2024-07-02", max_per_stream=8,
            separation_bars=4, seed=7, chunksize=31, verbose=False,
        )


def test_perfect_forecast_beats_persistence_and_has_valid_structure():
    n, context, horizon = 12, 16, 4
    rng = np.random.default_rng(4)
    base = np.linspace(90, 110, n)
    contexts = np.empty((n, context, 5), np.float32)
    futures = np.empty((n, horizon, 5), np.float32)
    for i in range(n):
        c = base[i] * np.exp(np.linspace(-0.01, 0, context))
        f = base[i] * np.exp(np.cumsum(rng.normal(0.001, 0.004, horizon)))
        contexts[i] = np.stack([c, c + .2, c - .2, c, np.full(context, 100)], axis=1)
        futures[i] = np.stack([f, f + .2, f - .2, f, np.full(horizon, 100)], axis=1)
    windows = {
        "context": contexts, "future": futures,
        "ticker": np.array(["ES"] * n), "timeframe": np.array(["1min"] * n),
    }
    pred = np.concatenate([futures, np.ones((n, horizon, 1), np.float32)], axis=2)
    metrics = evaluate_predictions(windows, pred)["overall"]
    assert metrics["path_log_return_mse"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["path_skill_vs_persistence"] == pytest.approx(1.0)
    assert metrics["fwd_absmove_r2"] == pytest.approx(1.0)
    assert metrics["vol_r2"] == pytest.approx(1.0)
    assert metrics["price_series_ic"] == pytest.approx(1.0)
    assert metrics["price_series_rank_ic"] == pytest.approx(1.0)
    assert metrics["valid_candle_fraction"] == 1.0


def test_kronos_input_mode_drops_volume_without_mutating_context():
    context = np.arange(40, dtype=np.float32).reshape(8, 5)
    original = context.copy()
    assert list(_context_frame(context, "ohlcv").columns) == [
        "open", "high", "low", "close", "volume"]
    assert list(_context_frame(context, "ohlc").columns) == [
        "open", "high", "low", "close"]
    np.testing.assert_array_equal(context, original)
    with pytest.raises(ValueError, match="unsupported"):
        _context_frame(context, "bad")
