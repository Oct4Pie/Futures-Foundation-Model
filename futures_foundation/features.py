"""
Feature derivation from raw OHLCV data.

All features are instrument-agnostic via ATR normalization or z-scores,
ensuring the backbone learns transferable patterns across instruments.

Feature Groups (67 continuous model features; candle_type uses its own embedding):
    1. Bar anatomy (8) — body, wicks, range normalized by ATR
    2. Returns & momentum (8) — multi-horizon returns + acceleration
    3. Volume dynamics (6) — relative volume, delta proxy
    4. Volatility measures (6) — ATR z-score, range ratios, realized vol
    5. Session-relative context (5) — distance from session OHLC + VWAP
    6. Market structure (9) — swing distances, range position multi-lookback
    7. CRT sweep state (10) — 1H/4H prior-candle liquidity sweep events
    8. Candle psychology (5) — engulf count, momentum speed, wick rejection, dir consistency, bar size vs session
    9. HTF price context (6) — position within ongoing 1H/4H candle, ATR-normalized return from HTF open, cross-TF alignment, 1H structure
   10. Volume absorption & order flow (4) — cumulative signed volume, absorption signal, momentum alignment
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

try:
    from .candle_psychology import add_candle_features as _add_candle_features
except ImportError:
    import importlib.util as _iu, pathlib as _pl
    _cp_path = str(_pl.Path(__file__).parent / "candle_psychology.py")
    _cp_spec = _iu.spec_from_file_location("_candle_psychology", _cp_path)
    _cp_mod = _iu.module_from_spec(_cp_spec)
    _cp_spec.loader.exec_module(_cp_mod)
    _add_candle_features = _cp_mod.add_candle_features


# =============================================================================
# Constants
# =============================================================================

INSTRUMENT_MAP = {
    "ES": 0, "NQ": 1, "RTY": 2, "YM": 3,
    "GC": 4, "SI": 5, "CL": 6,
    "ZB": 7, "ZN": 8,
}

SESSION_MAP = {"pre_market": 0, "london": 1, "ny_am": 2, "ny_pm": 3}

# Nominal session lengths in hours, keyed by the session id assigned in
# _compute_session_features. Used to normalize sess_bars_elapsed *causally*
# (elapsed-so-far / expected length) — the realized session length is a
# future quantity and must never be used at inference time.
SESSION_NOMINAL_HOURS = {0: 8.0, 1: 5.0, 2: 4.0, 3: 4.0}


# =============================================================================
# Main Feature Derivation
# =============================================================================


def derive_features(
    df: pd.DataFrame,
    instrument: str,
    atr_period: int = 14,
    vol_lookbacks: Tuple[int, ...] = (5, 10, 20),
    structure_lookback: int = 10,
) -> pd.DataFrame:
    """
    Derive the full feature set from raw OHLCV data.

    Args:
        df: DataFrame with columns [datetime, open, high, low, close, volume]
        instrument: Instrument symbol (e.g., "ES", "NQ")
        atr_period: Period for ATR calculation
        vol_lookbacks: Lookback periods for volume ratios
        structure_lookback: Lookback for swing point detection

    Returns:
        DataFrame with 42 derived features + metadata columns
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Bar cadence in minutes (the timeframe). Median of datetime diffs is
    # stable across any history window, so this is deterministic and identical
    # batch vs streaming.
    bar_minutes = max(1, int(df["datetime"].diff().dt.total_seconds().median() / 60))

    atr = _compute_atr(df, atr_period)
    atr_safe = atr.replace(0, np.nan)

    # bar_range can be 0 for zero-movement bars (common overnight/low liquidity).
    # Use a safe version for division but don't let it propagate NaN everywhere.
    bar_range = df["high"] - df["low"]
    bar_range_safe = bar_range.replace(0, np.nan)

    features = pd.DataFrame(index=df.index)

    # --- Group 1: Bar Anatomy (8 features) ---
    body = df["close"] - df["open"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    features["bar_range_atr"] = bar_range / atr_safe
    features["bar_body_atr"] = body / atr_safe
    features["bar_upper_wick_atr"] = upper_wick / atr_safe
    features["bar_lower_wick_atr"] = lower_wick / atr_safe
    features["bar_body_pct"] = (body.abs() / bar_range_safe).fillna(0)
    features["bar_upper_wick_pct"] = (upper_wick / bar_range_safe).fillna(0)
    features["bar_lower_wick_pct"] = (lower_wick / bar_range_safe).fillna(0)
    features["bar_direction"] = np.sign(body)

    # --- Group 2: Returns & Momentum (8 features) ---
    features["ret_close_1"] = df["close"].pct_change(1)
    features["ret_close_3"] = df["close"].pct_change(3)
    features["ret_close_5"] = df["close"].pct_change(5)
    features["ret_open_close"] = (df["close"] - df["open"]) / df["open"]
    for lb in [5, 10, 20]:
        features[f"ret_momentum_{lb}"] = features["ret_close_1"].rolling(lb).sum()
    features["ret_acceleration"] = (
        features["ret_momentum_5"] - features["ret_momentum_5"].shift(5)
    )

    # --- Group 3: Volume Dynamics (6 features) ---
    for lb in vol_lookbacks:
        vol_ma = df["volume"].rolling(lb).mean()
        features[f"vol_ratio_{lb}"] = df["volume"] / vol_ma.replace(0, np.nan)
    features["vol_change"] = df["volume"].pct_change(1)
    features["vol_close_position"] = ((df["close"] - df["low"]) / bar_range_safe).fillna(0.5)
    features["vol_delta_proxy"] = (features["vol_close_position"] - 0.5) * df["volume"]

    # --- Group 4: Volatility Measures (6 features) ---
    features["vty_atr_raw"] = atr  # For label generation; can drop before model input
    features["vty_atr_zscore"] = _zscore(atr, 50)
    features["vty_range_ratio_5"] = bar_range / bar_range.rolling(5).mean().replace(0, np.nan)
    features["vty_range_ratio_20"] = bar_range / bar_range.rolling(20).mean().replace(0, np.nan)
    features["vty_atr_of_atr"] = atr.rolling(14).std() / atr.rolling(14).mean().replace(0, np.nan)
    features["vty_realized_10"] = features["ret_close_1"].rolling(10).std()
    features["vty_realized_20"] = features["ret_close_1"].rolling(20).std()

    # --- Group 5: Session-Relative Context (5 features) ---
    session_ids, session_info = _compute_session_features(df, bar_minutes)
    features["sess_id"] = session_ids
    features["sess_time_of_day"] = session_info["time_of_day"]
    features["sess_bars_elapsed"] = session_info["bars_elapsed"]
    features["sess_dist_from_open"] = session_info["dist_from_open"] / atr_safe
    features["sess_dist_from_high"] = session_info["dist_from_high"] / atr_safe
    features["sess_dist_from_low"] = session_info["dist_from_low"] / atr_safe

    vwap = _compute_session_vwap(df)
    features["sess_dist_from_vwap"] = (df["close"] - vwap) / atr_safe

    # --- Group 6: Market Structure (9 features) ---
    swing_highs, swing_lows = _detect_swings(df, lookback=structure_lookback)
    structure = _classify_structure(swing_highs, swing_lows, len(df))

    features["str_swing_high_dist"] = (
        (df["close"] - swing_highs.ffill()) / atr_safe
    )
    features["str_swing_low_dist"] = (
        (df["close"] - swing_lows.ffill()) / atr_safe
    )
    features["str_structure_state"] = structure

    for lb in [10, 20]:
        rolling_high = df["high"].rolling(lb).max()
        rolling_low = df["low"].rolling(lb).min()
        features[f"str_dist_from_high_{lb}"] = (df["close"] - rolling_high) / atr_safe
        features[f"str_dist_from_low_{lb}"] = (df["close"] - rolling_low) / atr_safe
        rng = (rolling_high - rolling_low).replace(0, np.nan)
        features[f"str_range_position_{lb}"] = ((df["close"] - rolling_low) / rng).fillna(0.5)

    # --- Group 7: CRT Sweep State (10 features) ---
    # Timeframe-agnostic expiry windows from the bar cadence computed above.
    expiry_1h = max(1, round(60 / bar_minutes))
    expiry_4h = max(1, round(240 / bar_minutes))

    df_1h = _resample_ohlcv(df, "60min")
    df_4h = _resample_ohlcv(df, "240min")
    atr_1h = _compute_atr(df_1h, 14)
    atr_4h = _compute_atr(df_4h, 14)

    bull_1h, bear_1h, bmag_1h, brmag_1h = _detect_crt_sweeps(df_1h, atr_1h)
    bull_4h, bear_4h, bmag_4h, brmag_4h = _detect_crt_sweeps(df_4h, atr_4h)

    base_ns = df["datetime"].values.astype(np.int64)
    htf_1h_ns = df_1h.index.values.astype(np.int64)
    htf_4h_ns = df_4h.index.values.astype(np.int64)

    (b1h_act, br1h_act, b1h_age, br1h_age,
     b1h_mag, br1h_mag) = _align_sweeps_to_base(
        base_ns, htf_1h_ns, bull_1h, bear_1h, bmag_1h, brmag_1h, expiry_1h)

    (b4h_act, br4h_act, b4h_age, br4h_age,
     b4h_mag, br4h_mag) = _align_sweeps_to_base(
        base_ns, htf_4h_ns, bull_4h, bear_4h, bmag_4h, brmag_4h, expiry_4h)

    features["swp_1h_bull_active"] = b1h_act
    features["swp_1h_bear_active"] = br1h_act
    features["swp_1h_age_norm"] = np.minimum(b1h_age, br1h_age)
    features["swp_1h_magnitude"] = np.maximum(b1h_mag, br1h_mag)
    features["swp_4h_bull_active"] = b4h_act
    features["swp_4h_bear_active"] = br4h_act
    features["swp_4h_age_norm"] = np.minimum(b4h_age, br4h_age)
    features["swp_4h_magnitude"] = np.maximum(b4h_mag, br4h_mag)

    sweep_dir_1h = b1h_act - br1h_act
    sweep_dir_4h = b4h_act - br4h_act
    features["swp_tf_alignment"] = np.sign(sweep_dir_1h + sweep_dir_4h).astype(np.float32)
    features["swp_dominant_dir"] = features["swp_tf_alignment"].copy()

    # --- 1H Structure State (metadata for label generation) ---
    # Causal: uses only closed 1H bars. Majority direction over last 3 completed 1H bars.
    # +1 = at least 2 of 3 recent 1H bars closed higher (bullish structure)
    # -1 = at least 2 of 3 recent 1H bars closed lower (bearish structure)
    # 0 / NaN = mixed or insufficient history
    _1h_dir = np.sign(df_1h["close"].diff())
    _1h_maj = _1h_dir.shift(1) + _1h_dir.shift(2) + _1h_dir.shift(3)
    _1h_struct_htf = pd.Series(0, index=df_1h.index, dtype="Int64")
    _1h_struct_htf[_1h_maj >= 2] = 1
    _1h_struct_htf[_1h_maj <= -2] = -1
    _1h_struct_htf[_1h_dir.shift(3).isna()] = pd.NA
    _1h_period = df["datetime"].dt.ceil("60min")
    features["_1h_structure"] = _1h_period.map(_1h_struct_htf).astype("Int64")
    # Also expose as a continuous model feature (-1 / 0 / +1 → float32)
    features["htf_1h_structure"] = features["_1h_structure"].astype(float).fillna(0.0).astype(np.float32)

    # --- Daily Structure State (feature 68) ---
    # Causal: majority close direction over last 3 completed daily bars.
    # +1 = bullish daily structure, -1 = bearish, 0 = choppy/mixed.
    df_1d = _resample_ohlcv(df, "1D")
    _1d_dir = np.sign(df_1d["close"].diff())
    _1d_maj = _1d_dir.shift(1) + _1d_dir.shift(2) + _1d_dir.shift(3)
    _1d_struct_htf = pd.Series(0, index=df_1d.index, dtype="Int64")
    _1d_struct_htf[_1d_maj >= 2] = 1
    _1d_struct_htf[_1d_maj <= -2] = -1
    _1d_struct_htf[_1d_dir.shift(3).isna()] = pd.NA
    _1d_period = df["datetime"].dt.ceil("1D")
    features["_daily_structure"] = _1d_period.map(_1d_struct_htf).astype("Int64")
    features["htf_daily_structure"] = features["_daily_structure"].astype(float).fillna(0.0).astype(np.float32)

    # --- Group 8: Candle Psychology (6 features) ---
    df_cp = _add_candle_features(df)
    for _col in ("candle_type", "engulf_count", "momentum_speed_ratio",
                 "wick_rejection", "dir_consistency"):
        features[_col] = df_cp[_col].values

    # Current bar range relative to running session average (resets at session open).
    # Captures whether this bar is unusually large or small for the current session.
    _sess_keys = df["datetime"].dt.date.astype(str) + "_" + session_ids.astype(str)
    _sess_avg_range = bar_range.groupby(_sess_keys).expanding().mean().reset_index(level=0, drop=True)
    features["bar_size_vs_session"] = (bar_range / _sess_avg_range.replace(0, np.nan)).fillna(1.0).astype(np.float32)

    # --- Group 9: HTF Price Context (5 features) ---
    # Where is the current bar within the *ongoing* 1H and 4H candle?
    # These answer "early or late in an HTF move?" — information the sweep
    # features don't carry. Using the ongoing candle (not last completed)
    # matches what's observable in real-time at each bar.
    htf1h_open, htf1h_high, htf1h_low = _compute_htf_context(df, 60)
    htf4h_open, htf4h_high, htf4h_low = _compute_htf_context(df, 240)

    htf1h_range = (htf1h_high - htf1h_low).replace(0, np.nan)
    htf4h_range = (htf4h_high - htf4h_low).replace(0, np.nan)

    features["htf_1h_close_pos"] = (
        ((df["close"] - htf1h_low) / htf1h_range).fillna(0.5).clip(0.0, 1.0).astype(np.float32)
    )
    features["htf_1h_ret"] = (
        ((df["close"] - htf1h_open) / atr_safe).fillna(0.0).astype(np.float32)
    )
    features["htf_4h_close_pos"] = (
        ((df["close"] - htf4h_low) / htf4h_range).fillna(0.5).clip(0.0, 1.0).astype(np.float32)
    )
    features["htf_4h_ret"] = (
        ((df["close"] - htf4h_open) / atr_safe).fillna(0.0).astype(np.float32)
    )
    features["htf_tf_alignment"] = (
        np.sign(features["htf_1h_ret"]) * np.sign(features["htf_4h_ret"])
    ).astype(np.float32)

    # --- Group 10: Volume Absorption & Order Flow (4 features) ---
    # Cumulative signed volume: rolling weighted sum of per-bar buying/selling pressure,
    # normalized by rolling volume total → instrument-agnostic, range [-0.5, 0.5].
    # Positive = buyers have dominated; negative = sellers have dominated.
    _vol5_sum = df["volume"].rolling(5).sum().replace(0, np.nan)
    _vol20_sum = df["volume"].rolling(20).sum().replace(0, np.nan)
    features["vol_cum_signed_5"] = (
        (features["vol_delta_proxy"].rolling(5).sum() / _vol5_sum)
        .fillna(0.0).clip(-0.5, 0.5).astype(np.float32)
    )
    features["vol_cum_signed_20"] = (
        (features["vol_delta_proxy"].rolling(20).sum() / _vol20_sum)
        .fillna(0.0).clip(-0.5, 0.5).astype(np.float32)
    )
    # Absorption: elevated volume on a small-body bar = effort with no directional result.
    # High value = buyers or sellers absorbing the opposing side; signals potential exhaustion.
    features["vol_absorption"] = (
        (features["vol_ratio_5"] * (1.0 - features["bar_body_pct"].abs()))
        .clip(0.0, 5.0).fillna(1.0).astype(np.float32)
    )
    # Momentum alignment: is volume confirming the price trend or diverging from it?
    # Positive = trending direction + above-average volume (conviction, likely continues).
    # Negative = trending direction + below-average volume (weak move, likely fades).
    features["vol_momentum_align"] = (
        (np.sign(features["ret_momentum_5"].fillna(0)) * (features["vol_ratio_5"].fillna(1.0) - 1.0))
        .clip(-3.0, 3.0).astype(np.float32)
    )

    # --- Temporal (for embeddings, not model features) ---
    features["tmp_day_of_week"] = df["datetime"].dt.dayofweek
    features["tmp_hour"] = df["datetime"].dt.hour

    # --- Metadata ---
    features["_datetime"] = df["datetime"]
    features["_instrument"] = instrument
    features["_instrument_id"] = INSTRUMENT_MAP.get(instrument, 0)
    features["_close"] = df["close"]
    features["_high"] = df["high"]
    features["_low"] = df["low"]
    features["_volume"] = df["volume"]

    return features


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def _zscore(series: pd.Series, window: int = 50) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def _compute_session_features(df: pd.DataFrame, bar_minutes: int):
    hours = df["datetime"].dt.hour
    minutes = df["datetime"].dt.minute
    time_decimal = hours + minutes / 60.0

    session_ids = pd.Series(0, index=df.index)
    session_ids[(time_decimal >= 3) & (time_decimal < 8)] = 1
    session_ids[(time_decimal >= 8) & (time_decimal < 12)] = 2
    session_ids[(time_decimal >= 12) & (time_decimal < 16)] = 3

    time_of_day = time_decimal / 24.0
    dates = df["datetime"].dt.date
    session_keys = dates.astype(str) + "_" + session_ids.astype(str)

    session_open = df.groupby(session_keys)["open"].transform("first")
    session_high = df.groupby(session_keys)["high"].transform("cummax")
    session_low = df.groupby(session_keys)["low"].transform("cummin")
    bars_elapsed = df.groupby(session_keys).cumcount()
    # Causal normalization: divide elapsed-so-far by the *nominal* (expected)
    # session length for that session type, NOT the realized max (which needs
    # the whole future session). Clipped to [0, 1] so a session that runs
    # longer than nominal saturates instead of leaking length information.
    nominal_bars = (
        session_ids.map(SESSION_NOMINAL_HOURS).astype(float) * 60.0
        / max(1, bar_minutes)
    ).clip(lower=1.0)

    session_info = pd.DataFrame({
        "time_of_day": time_of_day,
        "bars_elapsed": (bars_elapsed / nominal_bars).clip(0.0, 1.0),
        "dist_from_open": df["close"] - session_open,
        "dist_from_high": df["close"] - session_high,
        "dist_from_low": df["close"] - session_low,
    }, index=df.index)

    return session_ids, session_info


def _compute_session_vwap(df: pd.DataFrame) -> pd.Series:
    dates = df["datetime"].dt.date
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).groupby(dates).cumsum()
    cum_vol = df["volume"].groupby(dates).cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


def _detect_swings(df: pd.DataFrame, lookback: int = 10):
    """
    Causal swing-pivot detection.

    A pivot is geometrically defined by a centered window: bar ``p`` is a
    swing high iff its high is the max of ``[p-lookback, p+lookback]``. That
    centered test cannot be evaluated until bar ``p+lookback`` exists, so the
    pivot is only *confirmable* ``lookback`` bars after it occurs.

    We therefore detect the pivot with the centered window (correct geometry)
    but **publish its price at the confirmation bar** (``shift(lookback)``).
    At bar ``i`` the returned series is non-NaN only for pivots whose full
    confirmation window ``[i-2*lookback, i]`` lies entirely in the past, so
    every downstream consumer (``.ffill()``, ``_classify_structure``) is
    causal. Centered rolling over a fixed window is purely local, so the
    confirmed value is byte-identical between batch and streaming
    (train == serve).
    """
    window = 2 * lookback + 1
    rolling_max = df["high"].rolling(window, center=True).max()
    rolling_min = df["low"].rolling(window, center=True).min()
    swing_highs = df["high"].where(df["high"] == rolling_max).shift(lookback)
    swing_lows = df["low"].where(df["low"] == rolling_min).shift(lookback)
    return swing_highs, swing_lows


def _classify_structure(swing_highs, swing_lows, length):
    structure = pd.Series(0, index=range(length))
    sh, sl = swing_highs.dropna(), swing_lows.dropna()

    if len(sh) >= 2:
        sh_diff = sh.diff()
        hh = (sh_diff > 0).astype(float).reindex(range(length), method="ffill").fillna(0.0).astype(bool)
        lh = (sh_diff < 0).astype(float).reindex(range(length), method="ffill").fillna(0.0).astype(bool)
    else:
        hh = lh = pd.Series(False, index=range(length))

    if len(sl) >= 2:
        sl_diff = sl.diff()
        hl = (sl_diff > 0).astype(float).reindex(range(length), method="ffill").fillna(0.0).astype(bool)
        ll = (sl_diff < 0).astype(float).reindex(range(length), method="ffill").fillna(0.0).astype(bool)
    else:
        hl = ll = pd.Series(False, index=range(length))

    structure[hh & hl] = 1
    structure[lh & ll] = -1
    return structure


def _compute_htf_context(
    df: pd.DataFrame,
    minutes: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    For each base bar, return the open, cumulative-high, and cumulative-low of
    the *ongoing* HTF candle that contains that bar.

    Uses dt.ceil(rule) to group base bars into HTF periods, matching the
    closed="right", label="right" convention used by _resample_ohlcv.
    The first bar of each period sets the HTF open; subsequent bars expand
    the running high/low — no lookahead into the future of that candle.

    Returns (htf_open, htf_high_so_far, htf_low_so_far), all pandas Series
    aligned to df.index.
    """
    rule = f"{minutes}min"
    period = df["datetime"].dt.ceil(rule)

    htf_open = df.groupby(period)["open"].transform("first")
    htf_high = df.groupby(period)["high"].cummax()
    htf_low = df.groupby(period)["low"].cummin()

    return htf_open, htf_high, htf_low


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample base OHLCV (datetime column) to a higher timeframe with a DatetimeIndex."""
    df_idx = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    return df_idx.resample(rule, closed="right", label="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])


def _detect_crt_sweeps(
    df_htf: pd.DataFrame, atr: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect CRT prior-candle sweeps on a higher-timeframe OHLCV DataFrame.

    Bull sweep: low[i] < low[i-1] AND close[i] > low[i-1]
    Bear sweep: high[i] > high[i-1] AND close[i] < high[i-1]

    Returns bull_sweep, bear_sweep, bull_mag, bear_mag — all aligned to df_htf index.
    Magnitude is ATR-normalized wick penetration clipped to [0, 3].
    """
    high = df_htf["high"].values
    low = df_htf["low"].values
    close = df_htf["close"].values
    atr_vals = np.nan_to_num(atr.values, nan=1e-6)
    n = len(df_htf)

    bull_sweep = np.zeros(n, dtype=bool)
    bear_sweep = np.zeros(n, dtype=bool)
    bull_mag = np.zeros(n, dtype=np.float32)
    bear_mag = np.zeros(n, dtype=np.float32)

    for i in range(1, n):
        atr_i = max(float(atr_vals[i]), 1e-6)
        if low[i] < low[i - 1] and close[i] > low[i - 1]:
            bull_sweep[i] = True
            bull_mag[i] = float(np.clip((low[i - 1] - low[i]) / atr_i, 0, 3))
        if high[i] > high[i - 1] and close[i] < high[i - 1]:
            bear_sweep[i] = True
            bear_mag[i] = float(np.clip((high[i] - high[i - 1]) / atr_i, 0, 3))

    return bull_sweep, bear_sweep, bull_mag, bear_mag


def _align_sweeps_to_base(
    base_ns: np.ndarray,
    htf_ns: np.ndarray,
    bull_mask: np.ndarray,
    bear_mask: np.ndarray,
    bull_mag_htf: np.ndarray,
    bear_mag_htf: np.ndarray,
    expiry_bars: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward-fill HTF CRT sweep state onto the base timeframe.

    Uses a vectorized countdown tent: each sweep event spreads a decaying
    countdown [expiry_bars, expiry_bars-1, ..., 1] forward from its first
    active base bar. Where two sweeps overlap the higher (more recent) value
    wins. Loop is over sweep events (~hundreds), not over base bars.

    Returns (bull_active, bear_active, bull_age_norm, bear_age_norm,
             bull_magnitude, bear_magnitude) as float32 arrays.
    """
    n_base = len(base_ns)

    def _spread(mask: np.ndarray, mag_htf: np.ndarray):
        countdown = np.zeros(n_base, dtype=np.float32)
        magnitude = np.zeros(n_base, dtype=np.float32)
        for j in np.where(mask)[0]:
            # First base bar strictly after this HTF bar's close timestamp
            p = int(np.searchsorted(base_ns, htf_ns[j], side="right"))
            if p >= n_base:
                continue
            end = min(p + expiry_bars, n_base)
            length = end - p
            new_counts = np.arange(expiry_bars, expiry_bars - length, -1, dtype=np.float32)
            better = new_counts > countdown[p:end]
            countdown[p:end] = np.where(better, new_counts, countdown[p:end])
            magnitude[p:end] = np.where(better, mag_htf[j], magnitude[p:end])
        active = (countdown > 0).astype(np.float32)
        # age_norm: 0.0 = fresh (just fired), 1.0 = expired / no sweep
        age_norm = np.where(countdown > 0, 1.0 - countdown / expiry_bars, 1.0).astype(np.float32)
        return active, age_norm, magnitude

    bull_active, bull_age, bull_mag = _spread(bull_mask, bull_mag_htf)
    bear_active, bear_age, bear_mag = _spread(bear_mask, bear_mag_htf)
    return bull_active, bear_active, bull_age, bear_age, bull_mag, bear_mag


# =============================================================================
# Feature Column Definitions
# =============================================================================


def get_model_feature_columns() -> list:
    """Returns feature columns that go into the model (excludes metadata)."""
    return [
        "bar_range_atr", "bar_body_atr", "bar_upper_wick_atr", "bar_lower_wick_atr",
        "bar_body_pct", "bar_upper_wick_pct", "bar_lower_wick_pct", "bar_direction",
        "ret_close_1", "ret_close_3", "ret_close_5", "ret_open_close",
        "ret_momentum_5", "ret_momentum_10", "ret_momentum_20", "ret_acceleration",
        "vol_ratio_5", "vol_ratio_10", "vol_ratio_20",
        "vol_change", "vol_close_position", "vol_delta_proxy",
        "vty_atr_zscore", "vty_range_ratio_5", "vty_range_ratio_20",
        "vty_atr_of_atr", "vty_realized_10", "vty_realized_20",
        "sess_bars_elapsed", "sess_dist_from_open", "sess_dist_from_high",
        "sess_dist_from_low", "sess_dist_from_vwap",
        "str_swing_high_dist", "str_swing_low_dist", "str_structure_state",
        "str_dist_from_high_10", "str_dist_from_low_10", "str_range_position_10",
        "str_dist_from_high_20", "str_dist_from_low_20", "str_range_position_20",
        "swp_1h_bull_active", "swp_1h_bear_active", "swp_1h_age_norm", "swp_1h_magnitude",
        "swp_4h_bull_active", "swp_4h_bear_active", "swp_4h_age_norm", "swp_4h_magnitude",
        "swp_tf_alignment", "swp_dominant_dir",
        # Group 8: Candle Psychology (candle_type excluded — uses its own model embedding)
        "engulf_count", "momentum_speed_ratio",
        "wick_rejection", "dir_consistency", "bar_size_vs_session",
        # Group 9: HTF Price Context
        "htf_1h_close_pos", "htf_1h_ret",
        "htf_4h_close_pos", "htf_4h_ret",
        "htf_tf_alignment",
        "htf_1h_structure",
        "htf_daily_structure",
        # Group 10: Volume Absorption & Order Flow
        "vol_cum_signed_5", "vol_cum_signed_20",
        "vol_absorption", "vol_momentum_align",
    ]