"""Domain-adapt Chronos-Bolt on futures data (self-supervised forecasting).

STRATEGY-AGNOSTIC generic infra: produces a futures-domain-adapted
`amazon/chronos-bolt-tiny` checkpoint by continuing its native forecasting
objective on our own bars. No strategy, no labels — it only makes the BACKBONE
better at futures price dynamics, which benefits every downstream
Chronos+XGBoost selection model. The deliverable is an A/B: the fine-tuned Bolt
vs vanilla Bolt on the honest-ruler walk-forward, to settle whether domain
adaptation improves embedding quality.

Distinct from the two sibling fine-tune paths in this package:
  - `finetune.py`      — SUPERVISED: backbone + classification head, CE on
                          strategy labels (task fine-tune).
  - `_ft/` (run_ft.py) — T5 forecasting domain-adapt (tokenizer/seq2seq path);
                          its `bolt.yaml` is scaffold-only, `--bolt` errors out.
  - THIS module        — BOLT forecasting domain-adapt. Bolt is patch-based and
                          tokenizer-FREE, so it needs a simpler sliding-window
                          collator, not the T5 data path. The model is already
                          trainable: `ChronosBoltModelForForecasting` is a
                          `T5PreTrainedModel` whose forward(context, target)
                          returns a quantile `loss` (chronos/chronos_bolt.py).

REPRESENTATION (matters for transfer): downstream `backbone.embed()` feeds
LOG-PRICE windows of length 128 (the *_chronos labelers use `lp=np.log(c)`,
`C.append(lp[i-CTX+1:i+1])`). So we domain-adapt on the SAME representation —
context = log(close) windows of `context_length` (default 128 to match embed
exactly) — so the encoder adapts to precisely what we later embed.

Torch/chronos are imported INSIDE run() (lazy) — never at module top — mirroring
backbone.py so the package stays import-safe in torch-free contexts.

CLI (Tier-2 defaults: 9 tickers, 1m+3m+5m, lr 1e-6, linear sched + warmup):
    python3 -m futures_foundation.chronos.bolt_finetune                 # full-FT Tier-2
    python3 -m futures_foundation.chronos.bolt_finetune --lora          # LoRA (lr auto 1e-4)
    python3 -m futures_foundation.chronos.bolt_finetune --smoke         # 2-step sanity
Then (per the wiring gap — NOT auto-applied — see backbone.stamp_active_source):
    python3 -m futures_foundation.chronos.bolt_ab \
        --strategy colabs/supertrend_chronos.py --ckpt <printed path>   # A/B vs vanilla
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# futures_foundation/chronos/bolt_finetune.py -> parents: [chronos, futures_foundation, REPO_ROOT]
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _ROOT / 'data'
OUT_DIR = _ROOT / 'temp' / 'chronos_bolt_ft'
MODEL_ID = 'amazon/chronos-bolt-tiny'

# Downstream signal models run on these 6 (equity + metal).
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI']
# Widest domain coverage for the backbone (CL/ZB/ZN add 1min+5min). Tier-2
# default: adapt on the broadest futures corpus available.
ALL_TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']


def _build_windows(context_length, prediction_length, stride, tfs, tickers,
                   months):
    """Sliding (context, target) windows over log(close), per ticker/tf.

    Returns (contexts, targets) float32:
      contexts: [N, context_length]    — log-price history (matches embed input)
      targets : [N, prediction_length] — the next prediction_length log-prices
    Causal by construction (target is strictly future bars; windows never cross
    ticker/tf boundaries). Missing ticker/tf files are skipped, not fatal — so
    e.g. CL/ZB/ZN (5min-only) contribute only their 5min windows.
    """
    ctxs, tgts = [], []
    span = context_length + prediction_length
    for tk in tickers:
        for tf in tfs:
            p = DATA_DIR / f'{tk}_{tf}.csv'
            if not p.exists():
                print(f"  skip {tk}_{tf} (no file)")
                continue
            df = pd.read_csv(p, usecols=['datetime', 'close'])
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            if months and months > 0:
                df = df[df['datetime'] >= df['datetime'].max()
                        - pd.DateOffset(months=months)]
            lp = np.log(df['close'].to_numpy(np.float64))
            lp = lp[np.isfinite(lp)]
            n = len(lp)
            if n < span:
                continue
            idx = np.arange(0, n - span + 1, stride)
            for i in idx:
                ctxs.append(lp[i:i + context_length])
                tgts.append(lp[i + context_length:i + span])
            print(f"  {tk}_{tf}: {len(idx)} windows ({n:,} bars)")
    if not ctxs:
        raise RuntimeError("no windows built — check data/ and --tfs")
    return np.asarray(ctxs, np.float32), np.asarray(tgts, np.float32)


def run(steps=1000, context_length=128, prediction_length=64, stride=32,
        tfs=('1min', '3min', '5min'), tickers=ALL_TICKERS, months=0, lr=1e-6,
        batch_size=256, smoke=False,
        warmup_ratio=0.0, lr_scheduler='linear', optim=None,
        lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.0):
    """Domain-adapt bolt-tiny on the given futures bars; save checkpoint.
    Returns the checkpoint Path (HF format — backbone.embed loads it directly).

    DEFAULTS = the OFFICIAL Chronos-2 `fit()` recipe (chronos2/pipeline.py),
    replicated for bolt-tiny (which has no fit()): lr=1e-6, max_steps=1000,
    batch_size=256, lr_scheduler='linear', warmup_ratio=0.0,
    optim='adamw_torch_fused' (auto→'adamw_torch' off-CUDA, e.g. MPS),
    LoRA r=8/alpha=16 on self-attention q/v/k/o + output_patch_embedding.output_layer.
    Two unavoidable bridges vs the Chronos-2 recipe: (1) fused AdamW is CUDA-only
    so MPS uses plain adamw_torch (same math); (2) LoRA target names mapped to
    bolt's module names. Everything else is the documented recipe — no deviation."""
    import torch
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainingArguments
    from chronos import BaseChronosPipeline

    if smoke:
        steps, months, stride = 2, 1, 256

    print("=== Chronos-Bolt domain-adapt (futures, self-supervised) ===")
    print(f"  base model   : {MODEL_ID}")
    print(f"  representation: log(close)  ctx={context_length}  "
          f"pred={prediction_length}  stride={stride}")
    print(f"  tickers={list(tickers)}  tfs={list(tfs)}  "
          f"months={months or 'ALL'}  steps={steps}")

    # Official optim is adamw_torch_fused (CUDA-only); fall back to adamw_torch
    # on MPS/CPU — same AdamW, no fused kernel.
    if optim is None:
        optim = ('adamw_torch_fused' if torch.cuda.is_available()
                 else 'adamw_torch')

    pipe = BaseChronosPipeline.from_pretrained(MODEL_ID)
    model = pipe.model                       # ChronosBoltModelForForecasting
    print(f"  recipe (OFFICIAL fit): lr={lr:g} steps={steps} batch={batch_size} "
          f"sched={lr_scheduler} warmup={warmup_ratio} optim={optim} lora={lora}"
          + (f" (r={lora_r},a={lora_alpha},drop={lora_dropout})" if lora else ""))
    if lora:
        from peft import LoraConfig, get_peft_model
        # Official Chronos-2 LoRA targets, mapped to bolt's module names:
        # self-attention q/v/k/o + the output projection head (same name in bolt).
        lc = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout, bias='none',
                        target_modules=['SelfAttention.q', 'SelfAttention.v',
                                        'SelfAttention.k', 'SelfAttention.o',
                                        'output_patch_embedding.output_layer'])
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()
    cfg_pred = model.chronos_config.prediction_length if not lora \
        else model.base_model.model.chronos_config.prediction_length
    if prediction_length > cfg_pred:
        print(f"  ⚠ requested pred={prediction_length} > model native "
              f"{cfg_pred}; clamping to {cfg_pred}")
        prediction_length = cfg_pred

    print("  building windows ...")
    ctxs, tgts = _build_windows(context_length, prediction_length, stride,
                                tfs, tickers, months)
    print(f"  total windows: {len(ctxs):,}  "
          f"(ctx {ctxs.shape}, tgt {tgts.shape})")

    class WinDS(Dataset):
        def __len__(self):
            return len(ctxs)

        def __getitem__(self, i):
            return {'context': torch.from_numpy(ctxs[i]),
                    'target': torch.from_numpy(tgts[i])}

    def collate(batch):
        return {'context': torch.stack([b['context'] for b in batch]),
                'target': torch.stack([b['target'] for b in batch])}

    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(                 # mirrors official Chronos-2 fit()
        output_dir=str(OUT_DIR / 'out'),
        max_steps=steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler,      # official: linear
        warmup_ratio=warmup_ratio,           # official: 0.0
        optim=optim,                         # official: adamw_torch_fused (→torch off-CUDA)
        gradient_accumulation_steps=1,       # official
        logging_strategy='steps', logging_steps=100,   # official
        save_strategy='no',                  # official: save only at end (save_pretrained)
        report_to='none',
        dataloader_num_workers=0,
        remove_unused_columns=False,         # keep 'context'/'target' keys
        use_cpu=(device == 'cpu'),
        fp16=False, bf16=False,              # MPS/CPU → fp32 (official bf16 is CUDA-sm80 only)
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=WinDS(), data_collator=collate)
    print(f"  device={device} | training {steps} steps ...")
    trainer.train()

    if lora:
        # Fold adapters into base weights so backbone.embed loads a plain ckpt.
        model = model.merge_and_unload()
    final = OUT_DIR / 'checkpoint-final'
    model.save_pretrained(str(final))
    print(f"\nDONE — domain-adapted Bolt checkpoint: {final}")
    print(f"\n  ⚠ To A/B vs vanilla Bolt, export BEFORE the walk-forward:")
    print(f"\n    export CHRONOS_FT_CKPT={final}\n")
    print(f"    python3 colabs/supertrend_chronos.py   # REAL vs SHUFFLE on FT-Bolt")
    print(f"  Then unset CHRONOS_FT_CKPT and re-run for the vanilla baseline.")
    print(f"  ⚠ Without the export, downstream silently uses vanilla {MODEL_ID}.")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1000, help='official: 1000')
    ap.add_argument('--context-length', type=int, default=128,
                    help='must match downstream embed CTX (128) for cleanest transfer')
    ap.add_argument('--prediction-length', type=int, default=64)
    ap.add_argument('--stride', type=int, default=32, help='bars between windows')
    ap.add_argument('--tfs', default='1min,3min,5min')
    ap.add_argument('--tickers', default=','.join(ALL_TICKERS),
                    help='default = all 9 futures')
    ap.add_argument('--months', type=int, default=0, help='0 = all available')
    ap.add_argument('--lr', type=float, default=None,
                    help='official: 1e-6 (both full and lora)')
    ap.add_argument('--batch-size', type=int, default=256, help='official: 256')
    ap.add_argument('--warmup-ratio', type=float, default=0.0, help='official: 0.0')
    ap.add_argument('--lr-scheduler', default='linear', help='official: linear')
    ap.add_argument('--optim', default=None,
                    help='official: adamw_torch_fused (auto→adamw_torch off-CUDA)')
    ap.add_argument('--lora', action='store_true',
                    help='parameter-efficient LoRA fine-tune (peft)')
    ap.add_argument('--lora-r', type=int, default=8, help='official: 8')
    ap.add_argument('--lora-alpha', type=int, default=16, help='official: 16')
    ap.add_argument('--lora-dropout', type=float, default=0.0, help='official: 0.0')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    lr = a.lr if a.lr is not None else 1e-6     # official: 1e-6 for both modes
    run(steps=a.steps, context_length=a.context_length,
        prediction_length=a.prediction_length, stride=a.stride,
        tfs=tuple(a.tfs.split(',')), tickers=tuple(a.tickers.split(',')),
        months=a.months, lr=lr, batch_size=a.batch_size, smoke=a.smoke,
        warmup_ratio=a.warmup_ratio, lr_scheduler=a.lr_scheduler, optim=a.optim,
        lora=a.lora, lora_r=a.lora_r, lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout)


if __name__ == '__main__':
    main()
