import copy

import numpy as np
import pytest

from futures_foundation.tick_path_labels import (
    INVALID_DECISION_OUTSIDE_COVERAGE,
    INVALID_HORIZON_OUTSIDE_SESSION,
    INVALID_STALE_ENDPOINT,
    INVALID_STALE_ENTRY,
    PATH_ADVERSE_FIRST,
    PATH_FAVORABLE_FIRST,
    PATH_NEITHER,
    OrderedTickPathIndex,
    TickPathLabelConfig,
    build_tick_path_labels,
    decision_manifest_sha256,
    load_tick_label_bundle,
    touches_by_horizon,
    write_tick_label_bundle,
)


SECOND = 1_000_000_000
HASHES = {
    "export_receipt_sha256": "1" * 64,
    "source_shard_sha256": "2" * 64,
    "source_file_table_sha256": "3" * 64,
    "corpus_contract_sha256": "4" * 64,
    "environment_receipt_sha256": "5" * 64,
    "instrument_spec_sha256": "6" * 64,
}


def _ticks(
    prices,
    *,
    tick_size=0.25,
    ts=None,
    seq=None,
    bid=None,
    ask=None,
    quote_valid=None,
    root="ES",
    contract_id="ESH25",
    session_start=-10,
    session_end=100,
    coverage_start=0,
    coverage_end=None,
):
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    ts = np.arange(n, dtype=np.int64) * SECOND if ts is None else np.asarray(ts, np.int64)
    seq = np.arange(n, dtype=np.uint64) if seq is None else np.asarray(seq)
    bid = prices - tick_size if bid is None else np.asarray(bid, dtype=np.float64)
    ask = prices + tick_size if ask is None else np.asarray(ask, dtype=np.float64)
    quote_valid = (
        np.ones(n, dtype=bool) if quote_valid is None else np.asarray(quote_valid, dtype=bool)
    )
    return {
        "timestamp_utc_ns": ts,
        "event_seq": seq,
        "price": prices,
        "bid": bid,
        "ask": ask,
        "quote_valid": quote_valid,
        "root": root,
        "contract_id": contract_id,
        "session_day": "2025-01-02",
        "session_start_utc_ns": np.int64(session_start * SECOND),
        "session_end_utc_ns": np.int64(session_end * SECOND),
        "coverage_start_utc_ns": np.int64(coverage_start * SECOND),
        "coverage_end_utc_ns": np.int64(
            (session_end - 1 if coverage_end is None else coverage_end) * SECOND
        ),
        "source_file_index": np.zeros(n, dtype=np.int64),
        "source_row_ordinal": np.arange(n, dtype=np.int64),
        "tick_size": tick_size,
        "tick_value": 12.5,
        **HASHES,
    }


def _cfg(horizon=4, tolerance=1, entry_tolerance=1, targets=(1.0, 2.0)):
    return TickPathLabelConfig(
        horizons_seconds=(horizon,),
        targets_r=targets,
        entry_tolerance_seconds=entry_tolerance,
        endpoint_tolerance_seconds=tolerance,
    )


def _labels(
    ticks,
    *,
    tick_size=0.25,
    decision_ts=(0,),
    decision_seq=(0,),
    risks=(2,),
    known_ts=None,
    known_seq=None,
    config=None,
    backend="indexed",
    manifest=None,
):
    config = config or _cfg()
    index = OrderedTickPathIndex(ticks, tick_size=tick_size, config=config)
    decision_ts = np.asarray(decision_ts, dtype=np.int64)
    decision_seq = np.asarray(decision_seq, dtype=np.uint64)
    risks = np.asarray(risks, dtype=np.int64)
    known_ts = decision_ts if known_ts is None else np.asarray(known_ts, dtype=np.int64)
    known_seq = decision_seq if known_seq is None else np.asarray(known_seq, dtype=np.uint64)
    expected = decision_manifest_sha256(
        index, decision_ts, decision_seq, risks, known_ts, known_seq
    )
    return build_tick_path_labels(
        index,
        decision_time_utc_ns=decision_ts,
        decision_event_seq=decision_seq,
        risk_ticks=risks,
        risk_known_time_utc_ns=known_ts,
        risk_known_event_seq=known_seq,
        decision_manifest_sha256_value=expected if manifest is None else manifest,
        config=config,
        backend=backend,
    )


def test_same_timestamp_sequence_excludes_signal_event():
    ticks = _ticks(
        [100.0, 100.25, 100.5, 101.0, 101.5],
        ts=[0, SECOND, SECOND, 2 * SECOND, 3 * SECOND],
        seq=[0, 10, 11, 12, 13],
    )
    labels = _labels(
        ticks,
        decision_ts=(SECOND,),
        decision_seq=(10,),
        risks=(2,),
        config=_cfg(horizon=3),
    )
    assert labels["valid"].item()
    assert labels["entry_index"].item() == 2
    assert labels["entry_time_utc_ns"].item() == SECOND


def test_missing_or_duplicate_event_sequence_fails_closed():
    ticks = _ticks([100, 100.25, 100.5])
    del ticks["event_seq"]
    with pytest.raises(ValueError, match="missing fields"):
        _labels(ticks)
    ticks = _ticks([100, 100.25, 100.5], ts=[0, SECOND, SECOND], seq=[0, 1, 1])
    with pytest.raises(ValueError, match="strictly increasing"):
        _labels(ticks)


def test_observed_and_marketable_at_trade_paths_are_separate_and_spread_aware():
    ticks = _ticks(
        [100.0, 100.25, 100.75, 101.25, 101.0],
        bid=[99.75, 100.25, 100.5, 101.0, 100.75],
        ask=[100.25, 100.5, 101.0, 101.5, 101.25],
    )
    labels = _labels(ticks)
    long, one_r = 0, 0
    assert labels["observed_barrier_state"][0, 0, long, one_r] == PATH_FAVORABLE_FIRST
    assert (
        labels["marketable_at_trade_barrier_state"][0, 0, long, one_r]
        == PATH_FAVORABLE_FIRST
    )
    assert labels["marketable_at_trade_gross_r"][0, 0, long, one_r] == pytest.approx(1.0)
    favorable = labels["marketable_at_trade_favorable_first_index_max_horizon"][
        0, long, one_r
    ]
    assert labels["marketable_at_trade_favorable_first_event_seq_max_horizon"][
        0, long, one_r
    ] == ticks[
        "event_seq"
    ][favorable]
    assert labels["marketable_at_trade_favorable_first_source_row_ordinal_max_horizon"][
        0, long, one_r
    ] == ticks[
        "source_row_ordinal"
    ][favorable]
    assert not labels["marketable_at_trade_is_fill_proof"]
    assert not labels["fees_included"] and not labels["added_slippage_included"]


def test_gap_through_stop_uses_observed_marketable_quote():
    ticks = _ticks(
        [100.0, 100.0, 98.5, 98.5, 98.5],
        bid=[99.75, 99.75, 98.25, 98.25, 98.25],
        ask=[100.25, 100.25, 98.75, 98.75, 98.75],
    )
    labels = _labels(ticks, risks=(4,))
    assert labels["marketable_at_trade_barrier_state"][0, 0, 0, 0] == PATH_ADVERSE_FIRST
    assert labels["marketable_at_trade_gross_r"][0, 0, 0, 0] == pytest.approx(-2.0)


def test_stale_endpoint_and_session_crossing_are_invalid():
    ticks = _ticks([100, 100.25, 100.5], ts=[0, SECOND, 2 * SECOND], session_end=10)
    stale = _labels(ticks, risks=(1,), config=_cfg(horizon=8, tolerance=1))
    assert not stale["valid"].item()
    assert stale["invalid_reason"].item() == INVALID_STALE_ENDPOINT
    outside_ticks = _ticks(
        [100, 100.25, 100.5, 100.75], ts=[0, SECOND, 2 * SECOND, 3 * SECOND], session_end=10
    )
    outside = _labels(
        outside_ticks,
        decision_ts=(2 * SECOND,),
        decision_seq=(2,),
        risks=(1,),
        config=_cfg(horizon=8, tolerance=10),
    )
    assert not outside["valid"].item()
    assert outside["invalid_reason"].item() == INVALID_HORIZON_OUTSIDE_SESSION


def test_prefix_invariance_and_semantic_fingerprint():
    prefix = _ticks([100, 100.25, 100.5, 100.75, 101.0], session_end=20)
    full = _ticks([100, 100.25, 100.5, 100.75, 101.0, 90.0, 110.0], session_end=20)
    a = _labels(prefix, config=_cfg(horizon=4))
    b = _labels(full, config=_cfg(horizon=4))
    for name in (
        "valid",
        "entry_index",
        "terminal_index",
        "observed_mfe_r",
        "observed_mae_r",
        "observed_barrier_state",
        "marketable_at_trade_barrier_state",
        "marketable_at_trade_gross_r",
    ):
        np.testing.assert_array_equal(a[name], b[name])
    assert a["semantic_fingerprint_sha256"] == b["semantic_fingerprint_sha256"]


def test_tick_alignment_decision_order_and_integral_risk_fail_closed():
    with pytest.raises(ValueError, match="not aligned"):
        _labels(_ticks([100.0, 100.1, 100.5]))
    ticks = _ticks([100.0, 100.25, 100.5, 100.75, 101.0])
    with pytest.raises(ValueError, match="decision keys"):
        _labels(
            ticks,
            decision_ts=(SECOND, 0),
            decision_seq=(1, 0),
            risks=(1, 1),
        )
    index = OrderedTickPathIndex(ticks, tick_size=0.25, config=_cfg())
    with pytest.raises(ValueError, match="integer array"):
        build_tick_path_labels(
            index,
            decision_time_utc_ns=np.asarray([0], np.int64),
            decision_event_seq=np.asarray([0], np.uint64),
            risk_ticks=np.asarray([1.5]),
            risk_known_time_utc_ns=np.asarray([0], np.int64),
            risk_known_event_seq=np.asarray([0], np.uint64),
            decision_manifest_sha256_value="0" * 64,
            config=_cfg(),
        )


def test_source_lineage_and_coverage_fail_closed():
    ticks = _ticks([100.0, 100.25, 100.5, 100.75, 101.0], coverage_end=4)
    labels = _labels(ticks, risks=(1,), config=_cfg(horizon=5))
    assert not labels["valid"].item()
    ticks = _ticks([100.0, 100.25, 100.5, 100.75, 101.0])
    ticks["source_row_ordinal"][2] = ticks["source_row_ordinal"][1]
    with pytest.raises(ValueError, match="lineage keys"):
        _labels(ticks)


@pytest.mark.parametrize(
    ("tick_size", "start"),
    [(0.1, 100.0), (0.00005, 1.0)],
)
def test_exact_barrier_touches_use_integer_ticks(tick_size, start):
    risk = 3
    prices = [start, start, start + risk * tick_size, start + risk * tick_size]
    ticks = _ticks(prices, tick_size=tick_size)
    labels = _labels(
        ticks,
        tick_size=tick_size,
        risks=(risk,),
        config=_cfg(horizon=3, targets=(1.0,)),
    )
    assert labels["observed_barrier_state"][0, 0, 0, 0] == PATH_FAVORABLE_FIRST
    assert labels["target_distance_ticks"].item() == risk


def test_negative_cl_prices_are_valid_tick_prices():
    ticks = _ticks(
        [-5.0, -5.0, -4.99, -4.98, -4.97],
        tick_size=0.01,
        root="CL",
        contract_id="CLK20",
    )
    labels = _labels(ticks, tick_size=0.01, risks=(2,))
    assert labels["valid"].item()
    assert labels["observed_barrier_state"][0, 0, 0, 0] == PATH_FAVORABLE_FIRST


def test_reference_and_indexed_backends_match_on_randomized_paths():
    rng = np.random.default_rng(9137)
    tick_walk = 400 + np.cumsum(rng.integers(-3, 4, size=240, dtype=np.int64))
    prices = tick_walk * 0.25
    ticks = _ticks(prices, tick_size=0.25, session_end=300)
    decisions = np.arange(0, 180, 15, dtype=np.int64) * SECOND
    sequences = np.arange(0, 180, 15, dtype=np.uint64)
    risks = rng.integers(1, 12, size=len(decisions), dtype=np.int64)
    config = TickPathLabelConfig(
        horizons_seconds=(20, 40),
        targets_r=(0.5, 1.0, 2.5),
        entry_tolerance_seconds=2,
        endpoint_tolerance_seconds=2,
    )
    indexed = _labels(
        ticks,
        decision_ts=decisions,
        decision_seq=sequences,
        risks=risks,
        config=config,
        backend="indexed",
    )
    reference = _labels(
        ticks,
        decision_ts=decisions,
        decision_seq=sequences,
        risks=risks,
        config=config,
        backend="reference",
    )
    assert indexed["semantic_fingerprint_sha256"] == reference["semantic_fingerprint_sha256"]
    for name, value in indexed.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, reference[name])


def test_shard_metadata_must_be_strict_scalar_and_valid():
    ticks = _ticks([100, 100.25, 100.5, 100.75, 101])
    ticks["contract_id"] = np.asarray(["ESH25"] * 5)
    with pytest.raises(ValueError, match="shard-level"):
        _labels(ticks)
    ticks = _ticks([100, 100.25, 100.5, 100.75, 101])
    ticks["root"] = None
    with pytest.raises(ValueError, match="string metadata"):
        _labels(ticks)
    ticks = _ticks([100, 100.25, 100.5, 100.75, 101])
    ticks["session_day"] = "2025-99-99"
    with pytest.raises(ValueError, match="valid ISO date"):
        _labels(ticks)


def test_decision_before_coverage_and_stale_entry_fail_closed():
    ticks = _ticks(
        [100, 100.25, 100.5, 100.75, 101],
        ts=np.arange(10, 15, dtype=np.int64) * SECOND,
        seq=np.arange(10, 15, dtype=np.uint64),
        session_start=-20,
        coverage_start=10,
    )
    outside = _labels(
        ticks,
        decision_ts=(-SECOND,),
        decision_seq=(0,),
        risks=(1,),
        config=_cfg(horizon=20, entry_tolerance=20),
    )
    assert outside["invalid_reason"].item() == INVALID_DECISION_OUTSIDE_COVERAGE

    ticks["coverage_start_utc_ns"] = np.int64(0)
    stale = _labels(
        ticks,
        decision_ts=(0,),
        decision_seq=(0,),
        risks=(1,),
        config=_cfg(horizon=20, entry_tolerance=1),
    )
    assert stale["invalid_reason"].item() == INVALID_STALE_ENTRY


def test_invalid_quote_masks_only_quote_proxy_not_observed_trade_path():
    ticks = _ticks(
        [100, 100, 100.5, 100.75, 101],
        quote_valid=[True, True, False, True, True],
    )
    ticks["bid"][2] = np.nan
    ticks["ask"][2] = np.nan
    labels = _labels(ticks)
    assert labels["valid"].item()
    assert labels["observed_barrier_state"][0, 0, 0, 0] == PATH_FAVORABLE_FIRST
    assert not labels["marketable_at_trade_valid"].item()
    assert labels["marketable_at_trade_barrier_state"][0, 0, 0, 0] == -1
    assert np.isnan(labels["marketable_at_trade_gross_r"][0, 0, 0, 0])


def test_risk_causality_and_manifest_binding_fail_closed():
    ticks = _ticks([100, 100.25, 100.5, 100.75, 101])
    with pytest.raises(ValueError, match="known no later"):
        _labels(ticks, known_ts=(SECOND,), known_seq=(1,))
    with pytest.raises(ValueError, match="manifest hash"):
        _labels(ticks, manifest="f" * 64)


@pytest.mark.parametrize(
    "config",
    [
        TickPathLabelConfig(horizons_seconds=(1.5,)),
        TickPathLabelConfig(targets_r=(float("nan"),)),
        TickPathLabelConfig(targets_r=(float("inf"),)),
        TickPathLabelConfig(entry_tolerance_seconds=1.5),
        TickPathLabelConfig(price_alignment_atol_ticks=float("nan")),
    ],
)
def test_configuration_rejects_fractional_and_nonfinite_values(config):
    with pytest.raises(ValueError):
        config.validate()


def test_alignment_tolerance_is_honored():
    prices = np.asarray([100.0, 100.2501, 100.5, 100.75, 101.0])
    ticks = _ticks(prices)
    permissive = TickPathLabelConfig(
        horizons_seconds=(4,),
        targets_r=(1.0,),
        price_alignment_atol_ticks=0.001,
    )
    assert _labels(ticks, config=permissive)["valid"].item()
    strict = TickPathLabelConfig(
        horizons_seconds=(4,),
        targets_r=(1.0,),
        price_alignment_atol_ticks=0.0001,
    )
    with pytest.raises(ValueError, match="not aligned"):
        _labels(ticks, config=strict)


def test_declared_endpoint_is_the_only_purge_authority():
    ticks = _ticks([100, 100.25, 100.5, 100.75, 101])
    labels = _labels(ticks, config=_cfg(horizon=4))
    assert labels["terminal_time_utc_ns"].item() == 4 * SECOND
    assert labels["declared_label_end_utc_ns"].item() == 4 * SECOND
    np.testing.assert_array_equal(
        labels["purge_time_utc_ns"], labels["declared_label_end_utc_ns"]
    )


def test_entry_record_is_not_reused_as_an_exit_observation():
    ticks = _ticks(
        [100.0, 99.5, 100.0, 100.0, 100.0],
        bid=[99.75, 99.0, 100.0, 100.0, 100.0],
        ask=[100.25, 100.0, 100.5, 100.5, 100.5],
    )
    labels = _labels(ticks, risks=(2,), config=_cfg(targets=(1.0,)))
    assert labels["marketable_at_trade_barrier_state"][0, 0, 0, 0] == PATH_NEITHER


def test_artifact_fingerprint_binds_provenance_but_semantics_do_not():
    ticks = _ticks([100, 100.25, 100.5, 100.75, 101])
    a = _labels(ticks)
    changed = copy.deepcopy(ticks)
    changed["source_shard_sha256"] = "a" * 64
    b = _labels(changed)
    assert a["semantic_fingerprint_sha256"] == b["semantic_fingerprint_sha256"]
    assert a["artifact_fingerprint_sha256"] != b["artifact_fingerprint_sha256"]


def test_input_byte_order_does_not_change_fingerprints():
    ticks = _ticks([100, 100.25, 100.5, 100.75, 101])
    native = _labels(ticks)
    swapped = copy.deepcopy(ticks)
    for name in ("timestamp_utc_ns", "event_seq", "source_file_index", "source_row_ordinal"):
        swapped[name] = swapped[name].astype(swapped[name].dtype.newbyteorder(">"))
    other = _labels(swapped)
    assert native["semantic_fingerprint_sha256"] == other["semantic_fingerprint_sha256"]
    assert native["artifact_fingerprint_sha256"] == other["artifact_fingerprint_sha256"]


def test_endpoint_at_half_open_session_end_is_invalid():
    ticks = _ticks([100, 100.25, 100.5, 100.75, 101], session_end=4, coverage_end=4)
    labels = _labels(ticks, config=_cfg(horizon=4, tolerance=1))
    assert not labels["valid"].item()
    assert labels["invalid_reason"].item() == INVALID_HORIZON_OUTSIDE_SESSION


def test_continuous_contract_and_wrong_instrument_spec_fail_closed():
    ticks = _ticks([100, 100.25, 100.5, 100.75, 101])
    ticks["contract_id"] = "ES_CONT"
    with pytest.raises(ValueError, match="unspliced dated contract"):
        _labels(ticks)

    ticks = _ticks([100, 100.25, 100.5, 100.75, 101])
    with pytest.raises(ValueError, match="hash-bound shard instrument"):
        _labels(ticks, tick_size=0.5)


def test_touch_indices_and_lineage_are_masked_to_each_horizon():
    ticks = _ticks([100, 100, 100, 100, 100.5, 100.5], session_end=20)
    config = TickPathLabelConfig(
        horizons_seconds=(2, 5),
        targets_r=(1.0,),
        entry_tolerance_seconds=1,
        endpoint_tolerance_seconds=1,
    )
    labels = _labels(ticks, risks=(2,), config=config)
    assert labels["observed_barrier_state"][0, 0, 0, 0] == PATH_NEITHER
    by_horizon = touches_by_horizon(
        labels["observed_favorable_first_index_max_horizon"], labels["terminal_index"]
    )
    assert by_horizon[0, 0, 0, 0] == -1
    assert labels["observed_barrier_state"][0, 1, 0, 0] == PATH_FAVORABLE_FIRST
    assert by_horizon[0, 1, 0, 0] == 4


def test_reusable_index_rejects_alignment_policy_drift():
    ticks = _ticks([100.0, 100.2501, 100.5, 100.75, 101.0])
    loose = TickPathLabelConfig(
        horizons_seconds=(4,), targets_r=(1.0,), price_alignment_atol_ticks=0.001
    )
    strict = TickPathLabelConfig(
        horizons_seconds=(4,), targets_r=(1.0,), price_alignment_atol_ticks=0.0001
    )
    index = OrderedTickPathIndex(ticks, tick_size=0.25, config=loose)
    decision_ts = np.asarray([0], np.int64)
    decision_seq = np.asarray([0], np.uint64)
    risks = np.asarray([2], np.int64)
    manifest = decision_manifest_sha256(
        index, decision_ts, decision_seq, risks, decision_ts, decision_seq
    )
    with pytest.raises(ValueError, match="alignment policy"):
        build_tick_path_labels(
            index,
            decision_time_utc_ns=decision_ts,
            decision_event_seq=decision_seq,
            risk_ticks=risks,
            risk_known_time_utc_ns=decision_ts,
            risk_known_event_seq=decision_seq,
            decision_manifest_sha256_value=manifest,
            config=strict,
        )


def test_fractional_target_rounding_uses_decimal_ceiling():
    tick_size = 0.1
    start = 100.0
    prices = [start, start, start + 55 * tick_size, start + 55 * tick_size]
    ticks = _ticks(prices, tick_size=tick_size)
    labels = _labels(
        ticks,
        tick_size=tick_size,
        risks=(50,),
        config=_cfg(horizon=3, targets=(1.1,)),
    )
    assert labels["target_distance_ticks"].item() == 55
    assert labels["observed_barrier_state"][0, 0, 0, 0] == PATH_FAVORABLE_FIRST


def test_index_owns_input_arrays_and_rejects_tick_index_overflow():
    ticks = _ticks([100, 100.25, 100.5, 100.75, 101])
    index = OrderedTickPathIndex(ticks, tick_size=0.25, config=_cfg())
    ticks["quote_valid"][:] = False
    assert index.quote_valid.all()
    with pytest.raises(ValueError, match="read-only"):
        index.quote_valid[0] = False
    with pytest.raises(ValueError, match="read-only"):
        index.trade_index.maximum[1] = 0

    overflow = _ticks([0.0, float(2 ** 63), 0.0, 0.0, 0.0], tick_size=1.0)
    with pytest.raises(ValueError, match="exceed int64"):
        _labels(overflow, tick_size=1.0)


def test_int64_ordering_checks_do_not_overflow():
    ticks = _ticks([100, 100.25, 100.5, 100.75, 101])
    ticks["timestamp_utc_ns"] = np.asarray(
        [1, np.iinfo(np.int64).max - 2, np.iinfo(np.int64).min + 2, -1, 2],
        dtype=np.int64,
    )
    ticks["session_start_utc_ns"] = np.int64(np.iinfo(np.int64).min)
    ticks["session_end_utc_ns"] = np.int64(np.iinfo(np.int64).max)
    ticks["coverage_start_utc_ns"] = np.int64(np.iinfo(np.int64).min)
    ticks["coverage_end_utc_ns"] = np.int64(np.iinfo(np.int64).max)
    with pytest.raises(ValueError, match="nondecreasing"):
        _labels(ticks)


def test_canonical_bundle_round_trip_and_tamper_detection(tmp_path):
    labels = _labels(_ticks([100, 100.25, 100.5, 100.75, 101]))
    bundle = write_tick_label_bundle(labels, tmp_path / "labels")
    loaded = load_tick_label_bundle(bundle)
    assert loaded["semantic_fingerprint_sha256"] == labels["semantic_fingerprint_sha256"]
    assert loaded["artifact_fingerprint_sha256"] == labels["artifact_fingerprint_sha256"]
    for name, value in labels.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(loaded[name], value)
            assert not loaded[name].flags.writeable

    path = bundle / "valid.npy"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="array hash mismatch"):
        load_tick_label_bundle(bundle)
