#!/usr/bin/env python3
"""Build the mmap-friendly, OOS-free train+validation cache once for Optuna workers."""
from __future__ import annotations

import argparse
import json

from futures_foundation.finetune.ssl_data import TFS_ALL, TICKERS_9
from futures_foundation.finetune.tournament_data import build_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data/ssl_corpus_v2_6tf")
    parser.add_argument("--cache-dir", default="output/foundation_tournament/data_cache")
    parser.add_argument("--tickers", default=",".join(TICKERS_9))
    parser.add_argument("--timeframes", default=",".join(TFS_ALL))
    args = parser.parse_args()
    report = build_cache(
        args.source_dir, args.cache_dir,
        tuple(x for x in args.tickers.split(",") if x),
        tuple(x for x in args.timeframes.split(",") if x),
    )
    print(json.dumps({"streams": len(report["entries"]),
                      "interval": report["interval"]}, indent=2))


if __name__ == "__main__":
    main()
