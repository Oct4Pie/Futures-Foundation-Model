"""POC-scale local domain-adapt of chronos-t5-tiny on our futures data.

Conditional/lazy: gluonts is imported ONLY inside prep_arrow() — it is not
a hard dependency of the chronos framework (mirrors how backbone lazy-loads
torch). Only invoked on the explicit fine-tune path.

Flow: our 3-min bars -> GluonTS Arrow -> official scripts/training/train.py
(vendored, Apache-2.0) -> fine-tuned T5 checkpoint dir (printed). The
backbone is then pointed at that checkpoint for the frozen-embed CRT POC.

  python -m pipelines.chronos._ft.run_ft [--months N] [--steps N] [--smoke]
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
TICKERS_DEFAULT = ['ES', 'NQ']


def prep_arrow(out_path, months=6, tickers=None, tfs=None):
    """Our close series per ticker -> GluonTS Arrow ({start,target}).
    gluonts imported HERE only (conditional — not a framework dep)."""
    try:
        from gluonts.dataset.arrow import ArrowWriter
    except ImportError as e:                       # pragma: no cover
        raise ImportError(
            "gluonts is required only for the Chronos fine-tune path. "
            "Install: pip install gluonts pyarrow") from e
    series = []
    for tk in (tickers or TICKERS_DEFAULT):
        for tf in (tfs or ['3min']):
            p = _ROOT / 'data' / f'{tk}_{tf}.csv'
            if not p.exists():
                continue
            df = pd.read_csv(p, usecols=['datetime', 'close'])
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            if months and months > 0:
                df = df[df['datetime'] >= df['datetime'].max()
                        - pd.DateOffset(months=months)]
            series.append({'start': df['datetime'].iloc[0].to_datetime64(),
                           'target': df['close'].to_numpy(np.float32)})
    ArrowWriter(compression='lz4').write_to_file(series, path=str(out_path))
    return [len(s['target']) for s in series]


def _write_config(arrow, out_dir, steps):
    import yaml
    cfg = yaml.safe_load((_HERE / 'poc.yaml').read_text())
    cfg['training_data_paths'] = [str(arrow)]
    cfg['output_dir'] = str(out_dir)
    cfg['max_steps'] = int(steps)
    cfg['save_steps'] = int(steps)
    p = _HERE / '_poc_resolved.yaml'
    p.write_text(yaml.safe_dump(cfg))
    return p


def run(months=6, steps=200, smoke=False, tickers=None, tfs=None):
    if smoke:
        months, steps = 1, 2
    work = _ROOT / 'temp' / 'chronos_t5_ft'
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    arrow = work / 'data.arrow'
    lens = prep_arrow(arrow, months, tickers=tickers, tfs=tfs)
    print(f"arrow: {arrow.name} | series={len(lens)} lens={lens} "
          f"total_pts={sum(lens):,}")
    cfg = _write_config(arrow, work / 'out', steps)
    cmd = [sys.executable, str(_HERE / 'train.py'), '--config', str(cfg),
           '--model-id', 'amazon/chronos-t5-tiny', '--no-random-init',
           '--max-steps', str(steps)]
    print('train:', ' '.join(cmd))
    env = dict(os.environ, PYTHONPATH=str(_ROOT))
    r = subprocess.run(cmd, cwd=str(_ROOT), env=env)
    if r.returncode != 0:
        print(f"FAIL: train.py exit {r.returncode}")
        return None
    runs = sorted((work / 'out').glob('run-*'))
    ckpt = runs[-1] if runs else (work / 'out')
    final = ckpt / 'checkpoint-final'
    target = final if final.exists() else ckpt
    print(f"\nDONE — fine-tuned checkpoint: {target}")
    # ⚠ Wiring gap from 2026-05-19: downstream training does NOT auto-pick
    # up the fine-tuned checkpoint. The user must export CHRONOS_FT_CKPT
    # before any walk-forward / production run. Print the exact command so
    # nobody has to remember it.
    print(f"\n  ⚠ To actually USE this checkpoint downstream, export the "
          f"env var BEFORE the next walk-forward / production run:")
    print(f"\n    export CHRONOS_FT_CKPT={target}\n")
    print(f"  ⚠ Without this export, downstream scripts silently fall back "
          f"to the frozen vanilla `amazon/chronos-bolt-tiny`.")
    return target


if __name__ == '__main__':
    a = sys.argv[1:]
    def _arg(name, cast, dflt):
        return cast(a[a.index(name) + 1]) if name in a else dflt
    run(months=_arg('--months', int, 6),
        steps=_arg('--steps', int, 200),
        tickers=_arg('--tickers', lambda s: s.split(','), None),
        tfs=_arg('--tfs', lambda s: s.split(','), None),
        smoke='--smoke' in a)
