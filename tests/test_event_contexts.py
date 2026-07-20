import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from futures_foundation.execution_economics import load_execution_economics
from futures_foundation.finetune.event_contexts import (
    EventContextConfig,
    TAG_NAMES,
    causal_baseline_features,
    context_shard_fingerprint,
    detect_context_tags,
    event_policy_labels,
    load_context_shard,
    materialize_context_stream,
    save_context_shard,
)
from futures_foundation.finetune.event_contexts import _context_edge_is_valid
from futures_foundation.finetune.event_contexts import _detect_compression_breakout
from futures_foundation.finetune.event_contexts import _detect_pullback_continuation
from futures_foundation.finetune.event_contexts import _single_policy_path
from futures_foundation.finetune.path_labels import BARRIER_AMBIGUOUS
from futures_foundation.finetune.path_labels import PathLabelConfig
from futures_foundation.pipeline._primitives import compute_atr


def _economics():
    return load_execution_economics(
        Path(__file__).resolve().parents[1] / "config/execution_costs.yaml",
        evaluation_start="2024-01-01T00:00:00Z",
        evaluation_end="2025-01-01T00:00:00Z",
        required_roots=("ES",),
    )


def _frame(n=1100, seed=4):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.25, n))
    open_ = np.r_[close[0], close[:-1]]
    wick = rng.uniform(0.05, 0.25, n)
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "open": open_,
        "high": np.maximum(open_, close) + wick,
        "low": np.minimum(open_, close) - wick,
        "close": close,
        "volume": rng.integers(1, 1000, n),
        "contract_id": np.full(n, "ESH4"),
        "source_row_idx": np.arange(10_000, 10_000 + n),
    })


def _config(frame):
    return EventContextConfig(
        eval_start=str(frame["datetime"].iloc[0]),
        eval_end=str(frame["datetime"].iloc[-1] + pd.Timedelta(minutes=1)),
        context_bars=256,
        atr_period=20,
        path=PathLabelConfig(
            horizons_minutes=(10, 20), targets_r=(1.0, 2.0), atr_period=20,
            context_minutes=10, barrier_chunk_rows=64,
        ),
    )


def test_baseline_features_are_prefix_invariant_and_context_bounded():
    frame = _frame()
    full, names = causal_baseline_features(frame)
    prefix, prefix_names = causal_baseline_features(frame.iloc[:800])
    assert names == prefix_names and len(names) == full.shape[1]
    np.testing.assert_allclose(full[:800], prefix, equal_nan=True)
    assert np.isnan(full[254]).any()
    assert np.isfinite(full[255:]).all()


def test_baseline_features_reset_all_state_at_contract_boundaries():
    frame = _frame(n=900)
    frame.loc[450:, "contract_id"] = "ESM4"
    original, names = causal_baseline_features(frame)
    changed = frame.copy()
    changed.loc[:449, ["open", "high", "low", "close"]] += 10_000.0
    perturbed, changed_names = causal_baseline_features(changed)
    assert names == changed_names
    np.testing.assert_allclose(original[450:], perturbed[450:], equal_nan=True)
    long_feature = names.index("net_change_context_scale_256bar")
    assert np.isnan(original[450:705, long_feature]).all()
    assert np.isfinite(original[705, long_feature])


def test_baseline_features_do_not_read_before_the_declared_256_bar_context():
    frame = _frame(n=900)
    original, names = causal_baseline_features(frame)
    decision = 600
    context_start = decision - 255
    changed = frame.copy()
    changed.loc[:context_start - 1, ["open", "high", "low", "close", "volume"]] *= 7.0
    perturbed, changed_names = causal_baseline_features(changed)
    assert names == changed_names
    np.testing.assert_allclose(original[decision], perturbed[decision], rtol=0.0, atol=1e-6)


def test_baseline_features_are_finite_for_negative_prices_without_log_semantics():
    frame = _frame(n=900)
    frame[["open", "high", "low", "close"]] -= 250.0
    features, names = causal_baseline_features(frame)
    assert np.isfinite(features[255:]).all()
    assert not any("log_return" in name for name in names)
    assert "net_change_context_scale_256bar" in names


def test_monotone_context_has_unit_trend_efficiency():
    frame = _frame(n=400)
    close = np.arange(400, dtype=float) - 200.0
    frame["open"] = np.r_[close[0], close[:-1]]
    frame["close"] = close
    frame["high"] = np.maximum(frame["open"], frame["close"]) + 0.25
    frame["low"] = np.minimum(frame["open"], frame["close"]) - 0.25
    features, names = causal_baseline_features(frame)
    column = names.index("trend_eff_256bar")
    assert features[255, column] == pytest.approx(1.0)


@pytest.mark.parametrize("value", [np.nan, "", "   "])
def test_event_context_rejects_missing_or_blank_contract_ids(value):
    frame = _frame(n=800)
    frame.loc[400, "contract_id"] = value
    with pytest.raises(ValueError, match="contract_id"):
        materialize_context_stream(
            frame, ticker="ES", timeframe="1min", config=_config(frame),
            execution_economics=_economics(),
        )


def test_all_trigger_tags_are_prefix_invariant_without_trade_suppression():
    frame = _frame(n=1400)
    full = detect_context_tags(frame, timeframe="1min")
    prefix = detect_context_tags(frame.iloc[:900], timeframe="1min")
    cutoff = 897  # centered fractal requires its right-side confirmation/entry edge
    for key in ("tags", "tag_direction", "tag_origin_source_idx", "tag_htf_agreement"):
        np.testing.assert_array_equal(full[key][:cutoff], prefix[key][:cutoff])
    np.testing.assert_array_equal(full["htf_direction"][:cutoff], prefix["htf_direction"][:cutoff])
    assert full["tags"].any(axis=0).all()


def test_new_event_detectors_use_only_the_decision_prefix():
    frame = _frame(n=1500, seed=9)
    full = detect_context_tags(frame, timeframe="1min")
    for end in (700, 1000, 1300):
        prefix = detect_context_tags(frame.iloc[:end], timeframe="1min")
        for name in ("pullback_continuation", "compression_breakout"):
            column = TAG_NAMES.index(name)
            np.testing.assert_array_equal(full["tags"][:end, column], prefix["tags"][:, column])
            np.testing.assert_array_equal(
                full["tag_direction"][:end, column], prefix["tag_direction"][:, column],
            )
            np.testing.assert_array_equal(
                full["tag_origin_source_idx"][:end, column],
                prefix["tag_origin_source_idx"][:, column],
            )


def test_new_event_detectors_fire_only_after_causal_confirmation():
    config = EventContextConfig()

    close = 100.0 + np.arange(140) * 0.1
    close[110] = close[109] - 1.0
    close[111] = close[109] + 0.3
    close[112:] = close[111] + np.arange(1, len(close) - 111) * 0.1
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.25
    low = np.minimum(open_, close) - 0.25
    atr = compute_atr(high, low, close, config.atr_period)
    pullbacks = _detect_pullback_continuation(open_, high, low, close, atr, config)
    assert pullbacks == [{"confirm": 111, "direction": 1, "origin": 110}]
    assert not _detect_pullback_continuation(
        open_[:111], high[:111], low[:111], close[:111], atr[:111], config,
    )

    close = np.full(80, 100.0)
    open_ = close.copy()
    high, low = close + 0.1, close - 0.1
    open_[60], close[60], high[60], low[60] = 100.0, 101.0, 101.1, 99.9
    atr = compute_atr(high, low, close, config.atr_period)
    breakouts = _detect_compression_breakout(open_, high, low, close, atr, config)
    assert breakouts == [{"confirm": 60, "direction": 1, "origin": 40}]
    assert not _detect_compression_breakout(
        open_[:60], high[:60], low[:60], close[:60], atr[:60], config,
    )


def test_materializer_deduplicates_contexts_and_rejects_rolls_and_gaps():
    frame = _frame(n=1400)
    frame.loc[620:, "contract_id"] = "ESM4"
    frame.loc[1000:, "datetime"] += pd.Timedelta(minutes=1)
    arrays, metadata = materialize_context_stream(
        frame, ticker="ES", timeframe="1min", config=_config(frame),
        execution_economics=_economics(),
    )
    decisions = arrays["decision_source_idx"]
    assert len(decisions) == len(np.unique(decisions)) == metadata["rows"]
    assert arrays["features"].shape[0] == len(decisions)
    assert np.isfinite(arrays["features"]).all()
    assert np.isfinite(arrays["sample_weight"]).all()
    assert abs(float(arrays["sample_weight"].mean()) - 1.0) < 1e-6
    assert metadata["detectors"]["atr_zigzag"] == "prefix_invariant_v2"
    policy_key = np.column_stack((
        arrays["policy_event_context_row"], arrays["policy_event_tag_index"],
    ))
    assert len(policy_key) == len(np.unique(policy_key, axis=0)) == metadata["policy_events"]
    assert arrays["policy_valid"].shape == (len(policy_key), 2)
    assert arrays["policy_gross_r"].shape[:2] == (len(policy_key), 2)

    source_to_local = {int(value): i for i, value in enumerate(frame["source_row_idx"])}
    for start_source, decision_source, label_end in zip(
        arrays["context_start_source_idx"], arrays["decision_source_idx"],
        arrays["label_end_time_ns"][:, -1],
    ):
        start = source_to_local[int(start_source)]
        decision = source_to_local[int(decision_source)]
        end = int(np.searchsorted(pd.DatetimeIndex(frame["datetime"]).asi8, label_end))
        assert frame["contract_id"].iloc[start] == frame["contract_id"].iloc[decision]
        assert frame["contract_id"].iloc[decision] == frame["contract_id"].iloc[end]
        context_delta = pd.DatetimeIndex(frame["datetime"].iloc[start:decision + 1]).asi8
        assert (np.diff(context_delta) == 60 * 1_000_000_000).all()


def test_context_edges_fail_closed_without_verified_session_capability():
    scheduled = pd.DatetimeIndex([
        pd.Timestamp("2024-01-02 15:59", tz="America/Chicago"),
        pd.Timestamp("2024-01-02 17:00", tz="America/Chicago"),
    ])
    assert _context_edge_is_valid(scheduled, 60 * 1_000_000_000).tolist() == [False]
    arbitrary = pd.DatetimeIndex([
        pd.Timestamp("2024-01-02 10:00", tz="America/Chicago"),
        pd.Timestamp("2024-01-02 10:02", tz="America/Chicago"),
    ])
    assert _context_edge_is_valid(arbitrary, 60 * 1_000_000_000).tolist() == [False]
    weekend = pd.DatetimeIndex([
        pd.Timestamp("2024-01-05 16:00", tz="America/Chicago"),
        pd.Timestamp("2024-01-07 17:00", tz="America/Chicago"),
    ])
    assert _context_edge_is_valid(weekend, 60 * 1_000_000_000).tolist() == [False]
    holiday_sized_arbitrary_gap = pd.DatetimeIndex([
        pd.Timestamp("2024-01-02 10:00", tz="America/Chicago"),
        pd.Timestamp("2024-01-04 10:00", tz="America/Chicago"),
    ])
    assert _context_edge_is_valid(
        holiday_sized_arbitrary_gap, 60 * 1_000_000_000,
    ).tolist() == [False]


def test_executable_policy_preserves_ambiguity_and_scores_adverse_first():
    h = np.array([100.0, 103.0, 100.0])
    l = np.array([100.0, 98.0, 100.0])
    c = np.array([100.0, 101.0, 100.0])
    state, realized, reached, exits = _single_policy_path(
        h, l, c, decision=0, direction=1, entry=100.0, risk=1.0, steps=1,
        targets=np.array([2.0, 3.0]),
    )
    np.testing.assert_array_equal(state, [BARRIER_AMBIGUOUS, BARRIER_AMBIGUOUS])
    np.testing.assert_allclose(realized, [-1.0, -1.0])
    assert not reached.any()
    np.testing.assert_array_equal(exits, [1, 1])


def test_vectorized_event_policies_match_scalar_reference():
    frame = _frame(n=40)
    selected = np.array([10, 20], dtype=np.int64)
    tags = np.zeros((2, len(TAG_NAMES)), dtype=bool)
    tags[0, 0] = True
    tags[1, 1] = True
    directions = np.zeros((2, len(TAG_NAMES)), dtype=np.int8)
    directions[0, 0] = 1
    directions[1, 1] = -1
    origins = np.full((2, len(TAG_NAMES)), -1, dtype=np.int64)
    origins[0, 0] = int(frame["source_row_idx"].iloc[8])
    origins[1, 1] = int(frame["source_row_idx"].iloc[18])
    cfg = EventContextConfig(
        eval_start="2024-01-01", eval_end="2024-01-02", context_bars=256,
        path=PathLabelConfig(
            horizons_minutes=(2, 4), targets_r=(1.0, 2.0), context_minutes=2,
            barrier_chunk_rows=1,
        ),
    )
    output = event_policy_labels(
        frame, ticker="ES", selected=selected, selected_tags=tags,
        selected_tag_direction=directions, selected_tag_origin_source_idx=origins,
        causal_scale=np.ones(len(frame)), horizons_minutes=np.array([2, 4]),
        targets_r=np.array([1.0, 2.0]), timeframe_minutes=1, config=cfg,
        execution_economics=_economics(),
    )
    h, l, c = (frame[name].to_numpy(float) for name in ("high", "low", "close"))
    for event_i, decision in enumerate(selected):
        direction = int(output["policy_event_direction"][event_i])
        for mode_i in range(2):
            if not output["policy_valid"][event_i, mode_i]:
                continue
            risk = float(output["policy_risk_price"][event_i, mode_i])
            for horizon_i, steps in enumerate((2, 4)):
                expected = _single_policy_path(
                    h, l, c, decision=int(decision), direction=direction,
                    entry=float(frame["open"].iloc[decision + 1]), risk=risk, steps=steps,
                    targets=np.array([1.0, 2.0]),
                )
                np.testing.assert_array_equal(
                    output["policy_barrier_state"][event_i, mode_i, horizon_i], expected[0]
                )
                np.testing.assert_allclose(
                    output["policy_gross_r"][event_i, mode_i, horizon_i], expected[1],
                )
                np.testing.assert_array_equal(
                    output["policy_reached"][event_i, mode_i, horizon_i], expected[2]
                )


def test_context_shard_roundtrip_and_hash_guard(tmp_path):
    frame = _frame(n=800)
    arrays, metadata = materialize_context_stream(
        frame, ticker="ES", timeframe="1min", config=_config(frame),
        execution_economics=_economics(),
    )
    path = tmp_path / "ES_1min.npz"
    manifest = save_context_shard(
        path, arrays, metadata, source={"path": "source.csv", "sha256": "abc"},
    )
    loaded, loaded_manifest = load_context_shard(path)
    assert loaded_manifest["content_fingerprint"] == manifest["content_fingerprint"]
    assert set(loaded) == set(arrays)
    assert not any(value.flags.writeable for value in loaded.values())
    np.testing.assert_array_equal(loaded["decision_source_idx"], arrays["decision_source_idx"])

    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_context_shard(path)


def test_current_context_save_rejects_arbitrary_self_consistent_arrays(tmp_path):
    with pytest.raises(ValueError, match="array keys mismatch"):
        save_context_shard(
            tmp_path / "fake-current.npz", {"x": np.asarray([1])}, {"rows": 1},
        )


def test_current_context_load_rejects_semantic_tamper_after_hash_repair(tmp_path):
    frame = _frame(n=800)
    arrays, metadata = materialize_context_stream(
        frame, ticker="ES", timeframe="1min", config=_config(frame),
        execution_economics=_economics(),
    )
    path = tmp_path / "ES_1min.npz"
    save_context_shard(path, arrays, metadata)
    tampered = {name: value.copy() for name, value in arrays.items()}
    tampered["label_end_time_ns"][0, 0] += 60 * 1_000_000_000
    np.savez_compressed(path, **tampered)
    manifest_path = Path(str(path) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["artifact"]["bytes"] = path.stat().st_size
    manifest["content_fingerprint"] = context_shard_fingerprint(
        tampered, manifest["metadata"],
    )
    manifest_path.write_text(json.dumps(manifest, allow_nan=False))
    with pytest.raises(ValueError, match="label endpoints"):
        load_context_shard(path)


def test_legacy_context_schema_requires_explicit_opt_in(tmp_path):
    path = tmp_path / "legacy.npz"
    arrays = {"x": np.asarray([1])}
    metadata = {"rows": 1}
    np.savez_compressed(path, **arrays)
    manifest = {
        "schema_version": "ffm_event_context_shard_v2",
        "status": "complete",
        "content_fingerprint": context_shard_fingerprint(arrays, metadata),
        "artifact": {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        },
        "source": {},
        "metadata": metadata,
    }
    manifest_path = Path(str(path) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported"):
        load_context_shard(path)
    loaded, _ = load_context_shard(path, allow_legacy=True)
    assert loaded["x"].item() == 1
    assert loaded["x"].flags.writeable is False
