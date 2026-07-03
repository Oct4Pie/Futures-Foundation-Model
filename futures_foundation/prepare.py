"""
Data preparation — raw OHLCV → cached feature/label parquets.

`prepare_data` survived the FFM-transformer retirement (tag
`ffm-transformer-final`): it lived in `futures_foundation.pretrain.trainer`
but is fully torch-free and is the canonical builder of the
`{INST}_features.parquet` caches consumed by the Chronos+XGBoost incumbent
and the quarterly-retrain runbook.
The torch pretraining loop it shipped with was deleted with the package.
"""

import json
import time
from pathlib import Path
from typing import Dict

import pandas as pd

from .features import derive_features, get_model_feature_columns, INSTRUMENT_MAP
from .labels import generate_all_labels, print_label_distribution


def prepare_data(
    raw_dir: str,
    output_dir: str,
    force: bool = False,
    atr_period: int = 14,
) -> Dict:
    """Derive 68 features + 4 pretraining labels from raw OHLCV files.

    Scans raw_dir for *.csv and *.parquet files. For each instrument, derives
    features and labels and saves them as parquet pairs to output_dir.
    Skips instruments already prepared unless force=True.

    atr_period: ATR lookback period. Default 14 (5min baseline). Use 20 for 3min
        data to match the same ~60-min lookback (20 × 3min = 60min ≈ 14 × 5min).

    Returns a summary dict keyed by instrument name.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_files = sorted(list(raw_dir.glob('*.csv')) + list(raw_dir.glob('*.parquet')))
    if not data_files:
        raise FileNotFoundError(f'No CSV or parquet files found in {raw_dir}')

    print(f"\n{'='*60}")
    print(f'  PREPARE DATA')
    print(f'  Input:  {raw_dir} ({len(data_files)} files)')
    print(f'  Output: {out_dir}')
    print(f"{'='*60}")

    summary = {}
    total_start = time.time()

    for data_path in data_files:
        instrument = data_path.stem.split('_')[0].upper()

        if instrument not in INSTRUMENT_MAP:
            print(f'\n  ⚠ Skipping {data_path.name} — "{instrument}" not in INSTRUMENT_MAP')
            continue

        feat_path  = out_dir / f'{instrument}_features.parquet'
        label_path = out_dir / f'{instrument}_labels.parquet'

        if feat_path.exists() and label_path.exists() and not force:
            print(f'  ⚡ {instrument} — cached (use force=True to reprocess)')
            summary[instrument] = {'cached': True}
            continue

        print(f'\n{"─"*60}')
        print(f'  {instrument} — {data_path.name}')
        print(f'{"─"*60}')
        t0 = time.time()

        df = (pd.read_parquet(data_path) if data_path.suffix == '.parquet'
              else pd.read_csv(data_path))
        df.columns = df.columns.str.strip().str.lower()
        if 'date' in df.columns and 'datetime' not in df.columns:
            df = df.rename(columns={'date': 'datetime'})

        required = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
        missing = required - set(df.columns)
        if missing:
            print(f'  ❌ Missing columns: {missing} — skipping')
            continue

        print(f'  Loaded {len(df):,} bars  {df["datetime"].iloc[0]} → {df["datetime"].iloc[-1]}')

        features_df = derive_features(df, instrument=instrument, atr_period=atr_period)
        labels_df   = generate_all_labels(features_df)
        print_label_distribution(labels_df)

        feature_cols = get_model_feature_columns()
        valid_count  = features_df[feature_cols].notna().all(axis=1).sum()

        features_df.to_parquet(feat_path,  index=False)
        labels_df.to_parquet(label_path, index=False)

        elapsed = time.time() - t0
        print(f'  ✓ {feat_path.name} + {label_path.name}  ({elapsed:.1f}s)')

        summary[instrument] = {
            'raw_bars':   len(df),
            'valid_bars': int(valid_count),
            'date_start': str(df['datetime'].iloc[0]),
            'date_end':   str(df['datetime'].iloc[-1]),
        }

    config_path = out_dir / 'prep_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'num_features':    len(get_model_feature_columns()),
            'feature_columns': get_model_feature_columns(),
            'atr_period':      atr_period,
            'instruments':     summary,
        }, f, indent=2)

    total_elapsed = time.time() - total_start
    processed = {k: v for k, v in summary.items() if not v.get('cached')}
    total_bars = sum(v.get('raw_bars', 0) for v in processed.values())

    print(f"\n{'='*60}")
    print(f'  ✅ PREPARE DATA COMPLETE')
    print(f'  Processed: {len(processed)} instruments  |  {total_bars:,} bars  ({total_elapsed:.1f}s)')
    print(f'  Output: {out_dir}')
    print(f"{'='*60}")
    return summary
