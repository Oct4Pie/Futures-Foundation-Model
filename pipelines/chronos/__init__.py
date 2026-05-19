"""Chronos pipeline — a generic, strategy-pluggable fine-tune framework.

Independent, standalone (no FFM-transformer dependency). The Chronos-Bolt
backbone is fine-tuned with a small classification head on labels supplied
by a pluggable strategy; results are judged only under a leak-free,
shuffle/random/cost honest ruler. Strategy specifics live OUTSIDE this
public package (private colabs/); this package stays strategy-agnostic.

Files:
  backbone.py — the ONLY Chronos seam: Bolt load, pristine reset (each
                fine-tune independent), causal masked-mean encoder pooling.
  finetune.py — generic seeded/deterministic backbone+head supervised
                fine-tune (train/predict); no strategy or eval logic.
  strategy.py — StrategyLabeler protocol (calendar/build/evaluate);
                contract: causal contexts, leak-free purge, cost inside
                evaluate(). The single pluggable seam.
  evaluate.py — honest ruler: leak-free walk-forward x seeds x
                REAL/SHUFFLE/RANDOM; believe a result only if REAL clearly
                beats SHUFFLE and RANDOM.
  data.py     — 6-ticker bars -> long format; leak-free walk-forward splits
                (born-tested: tests/test_chronos_data.py).
  embed.py    — frozen Chronos mean-pooled embeddings (generic utility).

Non-negotiable: strictly causal (bars <= t), splits never leak forward, no
performance number believed until it clears shuffle + random + cost.
"""
