#!/usr/bin/env python3
"""Build the one immutable validation-window artifact used by every forecast arm."""
from __future__ import annotations

import argparse
from pathlib import Path

from futures_foundation.finetune.foundation_eval import save_window_artifact
from futures_foundation.finetune.kronos_eval import build_forecast_windows
from futures_foundation.finetune.tournament import OOS_START, VALIDATION_START


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/ssl_corpus_v2_6tf")
    parser.add_argument("--output", default="output/foundation_tournament/shared_validation/windows.npz")
    parser.add_argument("--tickers", default="ES,NQ,RTY,YM,GC,SI,CL,ZB,ZN")
    parser.add_argument("--timeframes", default="1min,3min,5min,15min,30min,60min")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--max-per-stream", type=int, default=200)
    parser.add_argument("--separation-bars", type=int, default=16)
    parser.add_argument("--seed", type=int, default=6400)
    parser.add_argument("--csv-chunksize", type=int, default=250_000)
    args = parser.parse_args()
    tickers = tuple(value for value in args.tickers.split(",") if value)
    timeframes = tuple(value for value in args.timeframes.split(",") if value)
    windows = build_forecast_windows(
        args.data_dir, tickers, timeframes, context=args.context, horizon=args.horizon,
        eval_start=VALIDATION_START, eval_end=OOS_START,
        max_per_stream=args.max_per_stream, separation_bars=args.separation_bars,
        seed=args.seed, chunksize=args.csv_chunksize,
    )
    manifest = save_window_artifact(
        Path(args.output), windows,
        config={
            "data_dir": str(Path(args.data_dir).resolve()), "tickers": list(tickers),
            "timeframes": list(timeframes), "context": args.context,
            "horizon": args.horizon, "max_per_stream": args.max_per_stream,
            "separation_bars": args.separation_bars, "seed": args.seed,
        },
    )
    print(manifest)


if __name__ == "__main__":
    main()
