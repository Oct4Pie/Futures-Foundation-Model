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


def _locscale_on():
    return os.environ.get('CHRONOS_POOL_LOCSCALE') == '1'


def _pool(model, ctx):
    """Masked-mean pool of encoder hidden states (differentiable). Mirrors
    backbone.pool so FT == inference embedding. With CHRONOS_POOL_LOCSCALE=1,
    appends bolt's own loc+scale (the instance-norm de-norm terms it otherwise
    DISCARDS) → dim 256→258, restoring level/vol for direction."""
    h, ls, _emb, mask = model.encode(context=ctx)
    w = mask.unsqueeze(-1).to(h.dtype)
    pooled = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
    if _locscale_on():
        loc, scale = ls
        pooled = torch.cat([pooled,
                            loc.reshape(loc.shape[0], -1).to(pooled.dtype),
                            scale.reshape(scale.shape[0], -1).to(pooled.dtype)], dim=1)
    return pooled


def _pool_dim():
    return D_MODEL + (2 if _locscale_on() else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=float, default=3.0, help='full-FT epochs (bolt-tiny needs >documented 1000 steps)')
    ap.add_argument('--lr', type=float, default=1e-5, help='LoRA encoder lr — Amazon fit() docstring: "When finetune_mode=lora, we recommend a higher learning rate, such as 1e-5"')
    ap.add_argument('--head-lr', type=float, default=1e-3, help='fresh classification-head lr (higher than encoder)')
    ap.add_argument('--probe-epochs', type=float, default=2.0)
    ap.add_argument('--probe-lr', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=256, help='Tier-2 batch')
    ap.add_argument('--warmup', type=float, default=0.0, help='linear-warmup ratio (Amazon fit(): warmup_steps=0)')
    ap.add_argument('--lora', action='store_true', default=True)
    ap.add_argument('--full-ft', dest='lora', action='store_false', help='full backbone FT instead of LoRA')
    ap.add_argument('--out', default='temp/chronos_bolt_tb_ft')
    args = ap.parse_args()

    dev = ('mps' if torch.backends.mps.is_available()
           else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={dev} | recipe: lr={args.lr} batch={args.batch_size} "
          f"warmup={args.warmup} mode={'LoRA' if args.lora else 'full-FT'} epochs={args.epochs}")

    import pandas as pd
    X = np.load(f'{CORPUS}/X.npy')
    y = np.load(f'{CORPUS}/y.npy').astype(np.int64)
    ts = np.load(f'{CORPUS}/ts.npy')                       # per-window bar time (ns)
    n = len(X)
    # TEMPORAL split — NO random leak. train < T1, val [T1,T2), TEST >= T2 is held
    # OUT of the FT entirely (the downstream honest-ruler A/B runs on >= T2 so the
    # backbone is OOS). 1-day embargo at boundaries so context/label windows (CTX
    # back, H fwd) don't straddle a split.
    T1, T2 = (int(q) for q in np.quantile(ts, [0.70, 0.85]))
    EMB = 24 * 3600 * 1_000_000_000                        # 1-day embargo (ns)
    tr_ix = np.where(ts < T1 - EMB)[0]
    va_ix = np.where((ts >= T1) & (ts < T2 - EMB))[0]
    n_test = int((ts >= T2).sum())
    np.save(f'{CORPUS}/test_cutoff_ns.npy', np.array([T2], np.int64))  # boundary for the A/B
    print(f"corpus: {n:,} | TEMPORAL: train<{pd.Timestamp(T1)} | "
          f"val..{pd.Timestamp(T2)} | TEST>={pd.Timestamp(T2)} (held out, {n_test:,})")
    print(f"  train {len(tr_ix):,} | val {len(va_ix):,} | y dist {np.bincount(y, minlength=3).tolist()}")
    Xt, yt = torch.tensor(X), torch.tensor(y)

    pipe = __import__('chronos').BaseChronosPipeline.from_pretrained(MODEL_ID)
    model = pipe.model.to(dev)
    head = nn.Linear(_pool_dim(), N_CLS).to(dev)
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
        lc = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,   # Amazon Chronos-2 fit() defaults
                        target_modules=['SelfAttention.q', 'SelfAttention.v',
                                        'SelfAttention.k', 'SelfAttention.o',
                                        'output_patch_embedding.output_layer'])
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()
    else:
        for p in model.parameters():
            p.requires_grad_(True)
    model.to(dev)
    head = nn.Linear(_pool_dim(), N_CLS).to(dev)            # fresh head for the FT
    # differential LRs: LoRA encoder at args.lr (Amazon LoRA rec ~1e-5), fresh
    # head at the higher args.head_lr (random-init, must learn the mapping fast)
    groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': args.lr},
              {'params': list(head.parameters()), 'lr': args.head_lr}]
    try:
        opt = torch.optim.AdamW(groups, fused=True)
    except Exception:
        opt = torch.optim.AdamW(groups)
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
    best_val, best_ep = -1.0, -1                          # overfit guard: keep best-val epoch
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
        va = cor / tot
        print(f"[FT] ep{ep}: val acc={va:.4f}  (frozen baseline {probe_acc:.4f})", flush=True)
        # OVERFIT GUARD — keep the BEST-val epoch (early stopping), not the last
        if va > best_val and args.lora:
            best_val, best_ep = va, ep
            model.save_pretrained(args.out + '_adapter')
            torch.save(head.state_dict(), Path(args.out + '_adapter') / 'tb_head.pt')
            print(f"    ↑ new best val {best_val:.4f} (ep{ep}) — adapter snapshot saved")
        elif va > best_val:
            best_val, best_ep = va, ep
        if step >= total_steps:
            break

    # ── GEN-GATE — promote ONLY if the FT meaningfully beats frozen on held-out
    # val (analog of the strategies' gen-gate). Otherwise keep vanilla. ──────────
    GEN_MARGIN = 0.01
    print(f"\n{'='*60}\nGEN-GATE: best val {best_val:.4f} (ep{best_ep}) vs frozen "
          f"{probe_acc:.4f}  → lift {best_val - probe_acc:+.4f}")
    if best_val <= probe_acc + GEN_MARGIN:
        print(f"❌ NO generalizing lift (≤ frozen + {GEN_MARGIN}). Direction not "
              f"learnable here on held-out time — KEEP VANILLA, NOT promoting. "
              f"(no merged save)")
        return
    print(f"✅ LIFT +{best_val - probe_acc:.4f} — merging best adapter (ep{best_ep}) → {args.out}")
    os.makedirs(args.out, exist_ok=True)
    if args.lora:
        from peft import PeftModel
        base = __import__('chronos').BaseChronosPipeline.from_pretrained(MODEL_ID).model.to(dev)
        merged = PeftModel.from_pretrained(base, args.out + '_adapter').merge_and_unload()
    else:
        merged = model
    merged.save_pretrained(args.out)
    if hasattr(pipe, 'tokenizer'):
        pipe.tokenizer.save_pretrained(args.out)
    torch.save(head.state_dict(), Path(args.out) / 'tb_head.pt')
    print(f"\n✅ saved FT'd backbone (best ep{best_ep}, val {best_val:.4f}) → {args.out}")
    print(f"   NEXT: set CHRONOS_FT_CKPT + run the downstream A/B (FT vs vanilla on "
          f"the strategies, held-out ≥test_cutoff) — must beat vanilla ≥+0.10R to promote.")


if __name__ == '__main__':
    main()
