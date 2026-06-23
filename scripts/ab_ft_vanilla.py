"""Leak-free A/B: FT'd backbone vs vanilla on a strategy's honest ruler, scoped
to the held-out period (min_train_start >= test_cutoff) so the FT'd backbone is
OOS. Backbone selected by CHRONOS_FT_CKPT env (unset = vanilla).

Usage:
  python3 scripts/ab_ft_vanilla.py <strategy_module> [min_train_start]
  e.g. CHRONOS_FT_CKPT=checkpoints/chronos_bolt_ft python3 scripts/ab_ft_vanilla.py kalman_nw_chronos 2025-08-05
"""
import os, sys, json
sys.path.insert(0, '.')
os.environ.setdefault('FFM_TF', '3min')
import importlib

strat = sys.argv[1] if len(sys.argv) > 1 else 'kalman_nw_chronos'
CUT = sys.argv[2] if len(sys.argv) > 2 else '2025-08-05'
m = importlib.import_module(f'colabs.{strat}')

lab = None
for name in dir(m):
    o = getattr(m, name)
    if isinstance(o, type) and name.endswith('Chronos'):
        lab = o(); break
assert lab is not None, f"no labeler class found in {strat}"

from futures_foundation.pipeline import evaluate as ev
ck = os.environ.get('CHRONOS_FT_CKPT', 'VANILLA')
print(f"=== A/B {strat} | backbone={ck} | min_train_start={CUT} | TF={os.environ['FFM_TF']} ===", flush=True)

res = ev.run(lab, seeds=(0, 1, 2), min_train_start=CUT, return_verdict=True, loop=True)

keys = ('all_pass', 'generalizes', 'gap', 'thr', 'test_meanR', 'test_wr',
        'test_n', 'val_meanR', 'edge_shuffle', 'edge_random', 'edge_naive', 'auc')
if isinstance(res, dict):
    out = {k: res.get(k) for k in keys if k in res}
    print("VERDICT:", json.dumps(out, default=str))
else:
    print("RAW(type=%s):" % type(res).__name__, json.dumps(res, default=str)[:2000])
