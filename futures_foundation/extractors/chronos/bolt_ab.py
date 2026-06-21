"""A/B: does domain-adapting Chronos-Bolt add value? — vanilla vs fine-tuned.

Runs ONE strategy's honest-ruler walk-forward TWICE — identical seeds, folds,
strategy config, XGBoost head — changing ONLY the backbone weights:
  arm A  VANILLA   : CHRONOS_FT_CKPT unset  -> amazon/chronos-bolt-tiny
  arm B  FINETUNED : CHRONOS_FT_CKPT=<ckpt> -> our futures-domain-adapted Bolt

Then prints the side-by-side verdict. The honest measure is **REAL − SHUFFLE**
(edge over the label-shuffle control), NOT raw REAL: a fine-tuned backbone can
shift the absolute embedding scale, and the shuffle control normalizes that
out. Fine-tuning ADDS VALUE iff the fine-tuned arm's REAL−SHUFFLE edge clearly
exceeds vanilla's, on the same strategy.

HONEST CAVEAT (state it in the verdict): the fine-tune optimized a FORECASTING
objective; we use the EMBEDDINGS for SELECTION. Better forecasting need not
mean better selection embeddings — "no material lift" is a legitimate, useful
result (it says vanilla is good enough; stop spending on fine-tuning).

Each arm runs in its OWN subprocess (so the env var + torch isolation are
clean per arm). The strategy is loaded from a colabs/*.py file by path.

Usage:
    python3 -m futures_foundation.extractors.chronos.bolt_ab \
        --strategy colabs/supertrend_chronos.py \
        --ckpt temp/chronos_bolt_ft/checkpoint-final \
        --seeds 0,1,2 [--max-folds N]
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def _run_arm(strategy_path, ckpt, seeds, max_folds, label):
    """Run one walk-forward arm in a subprocess; return parsed aggregate
    REAL/SHUFFLE/RANDOM meanR. `ckpt=None` => vanilla (env var unset)."""
    driver = (
        "import importlib.util, sys, json\n"
        f"spec=importlib.util.spec_from_file_location('strat', {str(strategy_path)!r})\n"
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        f"m.run(seeds=tuple({list(seeds)}), max_folds={max_folds})\n"
    )
    env = dict(os.environ, PYTHONPATH=str(_ROOT))
    if ckpt:
        env['CHRONOS_FT_CKPT'] = str(ckpt)
    else:
        env.pop('CHRONOS_FT_CKPT', None)
    print(f"\n{'#'*72}\n# ARM: {label}  (backbone={'FINETUNED '+str(ckpt) if ckpt else 'VANILLA bolt-tiny'})\n{'#'*72}", flush=True)
    r = subprocess.run([sys.executable, '-c', driver], cwd=str(_ROOT), env=env,
                       capture_output=True, text=True)
    out = r.stdout + "\n" + r.stderr
    # Parse the AGGREGATE block: lines like "[REAL   ] trades=.. meanR=+1.733"
    agg = {}
    in_agg = False
    for line in out.splitlines():
        if 'AGGREGATE' in line:
            in_agg = True
        if in_agg:
            mm = re.search(r'\[(REAL|SHUFFLE|RANDOM)\s*\].*?meanR=([+-][\d.]+)', line)
            if mm and mm.group(1) not in agg:
                agg[mm.group(1)] = float(mm.group(2))
        if in_agg and agg.get('RANDOM') is not None:
            break
    if r.returncode != 0 and not agg:
        print(out[-3000:])
        raise RuntimeError(f"arm {label} failed (exit {r.returncode})")
    # save full log for inspection
    logp = _ROOT / 'temp' / f'bolt_ab_{label}.log'
    logp.parent.mkdir(parents=True, exist_ok=True)
    logp.write_text(out)
    print(f"  [{label}] REAL={agg.get('REAL')}  SHUFFLE={agg.get('SHUFFLE')}  "
          f"RANDOM={agg.get('RANDOM')}  (full log: {logp.relative_to(_ROOT)})",
          flush=True)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', default='colabs/supertrend_chronos.py')
    ap.add_argument('--ckpt', default='temp/chronos_bolt_ft/checkpoint-final')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--max-folds', default='None')
    a = ap.parse_args()
    strat = (_ROOT / a.strategy).resolve()
    ckpt = (_ROOT / a.ckpt).resolve()
    seeds = [int(x) for x in a.seeds.split(',')]
    max_folds = None if a.max_folds == 'None' else int(a.max_folds)
    if not strat.exists():
        sys.exit(f"strategy not found: {strat}")
    if not (ckpt / 'config.json').exists():
        sys.exit(f"checkpoint not found: {ckpt}")

    van = _run_arm(strat, None, seeds, max_folds, 'vanilla')
    fin = _run_arm(strat, ckpt, seeds, max_folds, 'finetuned')

    def edge(a):
        return (a['REAL'] - a['SHUFFLE']) if a.get('REAL') is not None \
            and a.get('SHUFFLE') is not None else None
    ev, ef = edge(van), edge(fin)
    print(f"\n{'='*72}\n🔬 BOLT FINE-TUNE A/B VERDICT — strategy: {a.strategy}\n{'='*72}")
    print(f"  {'arm':<10} {'REAL':>8} {'SHUFFLE':>8} {'edge(REAL-SHUF)':>16}")
    print(f"  {'vanilla':<10} {van.get('REAL'):>8} {van.get('SHUFFLE'):>8} {ev:>16.3f}")
    print(f"  {'finetuned':<10} {fin.get('REAL'):>8} {fin.get('SHUFFLE'):>8} {ef:>16.3f}")
    if ev is not None and ef is not None:
        delta = ef - ev
        rel = (delta / abs(ev) * 100) if ev else float('inf')
        print(f"\n  Δ edge (finetuned − vanilla): {delta:+.3f}R  ({rel:+.0f}%)")
        if delta >= 0.10:
            print(f"  ✅ FINE-TUNE ADDS VALUE — edge improved ≥0.10R. Worth pursuing.")
        elif delta <= -0.10:
            print(f"  ❌ FINE-TUNE HURTS — edge dropped ≥0.10R. Keep vanilla.")
        else:
            print(f"  ⟂ NO MATERIAL DIFFERENCE (|Δ|<0.10R). Vanilla is good "
                  f"enough; domain fine-tune doesn't help this strategy.")
        print(f"\n  Caveat: fine-tune optimized FORECASTING; embeddings used for "
              f"SELECTION. This A/B measures the actual transfer for ONE "
              f"strategy/seeds — not a universal verdict.")
    print(f"{'='*72}")
    (_ROOT / 'temp' / 'bolt_ab_verdict.json').write_text(json.dumps(
        {'strategy': a.strategy, 'vanilla': van, 'finetuned': fin,
         'edge_vanilla': ev, 'edge_finetuned': ef}, indent=2))


if __name__ == '__main__':
    main()
