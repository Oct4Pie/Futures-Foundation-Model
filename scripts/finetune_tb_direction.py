"""Strategy-free triple-barrier DIRECTION fine-tune of chronos-bolt-tiny.

The SELECTION-ALIGNED objective (forecasting-FT is capped — this is the one open
lever): teach bolt to read direction from the normalized shape by classifying the
symmetric ±1·ATR triple-barrier outcome (UP/DOWN/NEITHER) on the strategy-free
corpus (temp/tb_corpus, 9 tickers × 1/3/5min, 1.37M windows).

Architecture (stays our extractor pattern — XGBoost remains the classifier later):
  bolt encoder (LoRA) → masked-mean pool → 3-class head, cross-entropy.
Two stages:
  PROBE   — FROZEN bolt + head on cached embeddings → baseline (can frozen bolt
            predict direction? expected ~0.50 binary — that's WHY we FT).
  FULL FT — LoRA the encoder + head, Tier-2 recipe (lr 1e-6, linear sched +
            warmup, adamw_torch_fused, LoRA r8/α8). Steps cover a few epochs
            (bolt-tiny needs more than the documented 1000).
Saves the FT'd backbone (LoRA merged) for the extractor (set CHRONOS_FT_CKPT).

    python3 scripts/finetune_tb_direction.py --epochs 3 --lr 1e-6 --lora
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MODEL_ID = 'amazon/chronos-bolt-tiny'
D_MODEL, N_CLS = 256, 3
CORPUS = 'temp/tb_corpus'


def _pool(model, ctx):
    """Masked-mean pool of encoder hidden states (differentiable). Mirrors
    backbone.pool so FT == inference embedding."""
    h, _ls, _emb, mask = model.encode(context=ctx)
    w = mask.unsqueeze(-1).to(h.dtype)
    return (h * w).sum(1) / w.sum(1).clamp(min=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=float, default=3.0, help='full-FT epochs (bolt-tiny needs >documented 1000 steps)')
    ap.add_argument('--lr', type=float, default=1e-6, help='Tier-2 encoder/FT lr')
    ap.add_argument('--probe-epochs', type=float, default=2.0)
    ap.add_argument('--probe-lr', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=256, help='Tier-2 batch')
    ap.add_argument('--warmup', type=float, default=0.05, help='linear-warmup ratio')
    ap.add_argument('--lora', action='store_true', default=True)
    ap.add_argument('--full-ft', dest='lora', action='store_false', help='full backbone FT instead of LoRA')
    ap.add_argument('--out', default='temp/chronos_bolt_tb_ft')
    args = ap.parse_args()

    dev = ('mps' if torch.backends.mps.is_available()
           else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={dev} | recipe: lr={args.lr} batch={args.batch_size} "
          f"warmup={args.warmup} mode={'LoRA' if args.lora else 'full-FT'} epochs={args.epochs}")

    X = np.load(f'{CORPUS}/X.npy')
    y = np.load(f'{CORPUS}/y.npy').astype(np.int64)
    n = len(X)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    ntr = int(0.9 * n)
    tr_ix, va_ix = perm[:ntr], perm[ntr:]
    print(f"corpus: {n:,} | train {len(tr_ix):,} val {len(va_ix):,} | "
          f"y dist {np.bincount(y, minlength=3).tolist()}")
    Xt, yt = torch.tensor(X), torch.tensor(y)

    pipe = __import__('chronos').BaseChronosPipeline.from_pretrained(MODEL_ID)
    model = pipe.model.to(dev)
    head = nn.Linear(D_MODEL, N_CLS).to(dev)
    ce = nn.CrossEntropyLoss()

    def embed_all(ix, bs=512):
        """Cache frozen embeddings for the probe (no grad)."""
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(ix), bs):
                xb = Xt[ix[s:s+bs]].to(dev)
                out.append(_pool(model, xb).float().cpu())
        return torch.cat(out)

    def val_acc_from_emb(emb, yv):
        head.eval()
        with torch.no_grad():
            pred = head(emb.to(dev)).argmax(1).cpu()
        return (pred == yv).float().mean().item()

    # ── STAGE 1: PROBE (frozen bolt + head on cached embeddings) ──────────────
    for p in model.parameters():
        p.requires_grad_(False)
    t0 = time.time()
    print("\n[PROBE] caching frozen embeddings ...")
    Etr = embed_all(tr_ix); Eva = embed_all(va_ix)
    yva = yt[va_ix]
    print(f"  cached ({time.time()-t0:.0f}s). training head ...")
    opt = torch.optim.AdamW(head.parameters(), lr=args.probe_lr)
    pds = DataLoader(TensorDataset(Etr, yt[tr_ix]), batch_size=1024, shuffle=True)
    head.train()
    for ep in range(int(args.probe_epochs)):
        for eb, yb in pds:
            opt.zero_grad()
            loss = ce(head(eb.to(dev)), yb.to(dev))
            loss.backward(); opt.step()
        print(f"  probe ep{ep}: val acc={val_acc_from_emb(Eva, yva):.4f}")
    probe_acc = val_acc_from_emb(Eva, yva)
    print(f"[PROBE] frozen-bolt direction val acc = {probe_acc:.4f}  "
          f"(0.50 = no directional signal in frozen embedding)")

    # ── STAGE 2: FULL FT (LoRA encoder + head, Tier-2) ────────────────────────
    if args.lora:
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.05,
                        target_modules=['SelfAttention.q', 'SelfAttention.v',
                                        'SelfAttention.k', 'SelfAttention.o',
                                        'output_patch_embedding.output_layer'])
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()
    else:
        for p in model.parameters():
            p.requires_grad_(True)
    model.to(dev)
    head = nn.Linear(D_MODEL, N_CLS).to(dev)            # fresh head for the FT
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    try:
        opt = torch.optim.AdamW(params, lr=args.lr, fused=True)
    except Exception:
        opt = torch.optim.AdamW(params, lr=args.lr)
    steps_per_ep = (len(tr_ix) + args.batch_size - 1) // args.batch_size
    total_steps = int(steps_per_ep * args.epochs)
    warm = int(args.warmup * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm
        else max(0.0, (total_steps - s) / max(1, total_steps - warm)))  # linear warmup+decay
    print(f"\n[FT] {total_steps} steps ({args.epochs} ep × {steps_per_ep}/ep), warmup {warm}")
    tds = DataLoader(TensorDataset(Xt[tr_ix], yt[tr_ix]),
                     batch_size=args.batch_size, shuffle=True)
    step = 0; t0 = time.time()
    for ep in range(int(np.ceil(args.epochs))):
        model.train(); head.train()
        for xb, yb in tds:
            if step >= total_steps:
                break
            opt.zero_grad()
            loss = ce(head(_pool(model, xb.to(dev))), yb.to(dev))
            loss.backward(); opt.step(); sched.step(); step += 1
            if step % 500 == 0:
                print(f"  step {step}/{total_steps} loss={loss.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} ({(time.time()-t0)/60:.1f}m)", flush=True)
        # val acc each epoch (re-encode val with the FT'd encoder)
        model.eval(); head.eval(); cor = tot = 0
        with torch.no_grad():
            for s in range(0, len(va_ix), 512):
                xb = Xt[va_ix[s:s+512]].to(dev)
                pred = head(_pool(model, xb)).argmax(1).cpu()
                cor += (pred == yt[va_ix[s:s+512]]).sum().item(); tot += len(xb)
        print(f"[FT] ep{ep}: val acc={cor/tot:.4f}  (probe {probe_acc:.4f})", flush=True)
        if step >= total_steps:
            break

    # ── SAVE (LoRA-merged backbone for the extractor) ─────────────────────────
    os.makedirs(args.out, exist_ok=True)
    save_model = model.merge_and_unload() if args.lora else model
    save_model.save_pretrained(args.out)
    pipe.tokenizer.save_pretrained(args.out) if hasattr(pipe, 'tokenizer') else None
    torch.save(head.state_dict(), Path(args.out) / 'tb_head.pt')
    print(f"\n✅ saved FT'd backbone → {args.out}  (set CHRONOS_FT_CKPT to use it)")
    print(f"   PROBE acc {probe_acc:.4f} → FT final acc {cor/tot:.4f}  "
          f"(lift = what the FT taught the encoder)")


if __name__ == '__main__':
    main()
