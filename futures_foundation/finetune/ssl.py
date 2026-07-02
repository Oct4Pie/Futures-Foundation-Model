"""Masked-modeling SSL pretraining of the Mantis backbone (orchestrator) — "BERT for
futures".

BERT-style masked modeling on raw OHLCV across the 9 futures tickers x {1,3,5,15}min:
mask a fraction of bars and reconstruct them from context (in _ssl_torch.train_ssl_mask),
so the encoder learns regime/volatility/structure. Produces an ADAPTED ENCODER CHECKPOINT
(saved to Drive on Colab) that downstream classifier finetuning starts from
(build_model(..., backbone_ckpt=...) / BACKBONE_CKPT=... in the WF/produce driver).

Torch-free at import (the GPU trainer in _ssl_torch + the probe's torch bits load lazily)
so data assembly, the generalization gate, the Optuna search wiring, and the contract are
testable without the torch/mantis stack.

Generalization is PROBE-GATED + OPTUNA-TUNED, mirroring WF/produce:
  * TIME-SPLIT val reconstruction early-stop  (generalize forward; 2026 EXCLUDED)
  * GATE = a linear PROBE shows the frozen embedding predicts regime / vol / structure +
    forward buy/sell move BETTER than vanilla Mantis (the classification-relevance test)
  * if it doesn't pass -> OPTUNA tunes lr/reg/capacity/mask_ratio to MAXIMIZE the probe
  * REAL vs SHUFFLE vs RANDOM = probe-based diagnostic (did temporal order contribute)

Colab usage: see colab/mantis_ssl_pretrain.py.
"""
import argparse
import json
import os

import numpy as np

from . import ssl_data
from .pretext import PRETEXTS, PretextTask, get_pretext   # noqa: F401 (pluggable pretext registry)


def assemble(streams, *, seq, max_jitter, val_frac, holdout_start, forecast_parent=0, verbose=True):
    """Concatenate all stream OHLCV into one big [T, 5] array + global parent-window start
    positions for the (leak-safe, 2026-excluded) train/val split. Each window reserves enough
    bars for its consumers: seq+max_jitter (probe/mask) OR forecast_parent (stage-2 = max context
    length + max horizon), whichever is larger — so context+future stay in-stream."""
    parent_len = max(seq + max_jitter, int(forecast_parent))
    bigs, tr_starts, va_starts, base = [], [], [], 0
    for s in streams:
        oh = s['ohlcv']
        tr_idx, va_idx = ssl_data.time_split(s['ts'], val_frac, holdout_start)
        ts = ssl_data.window_starts(tr_idx, parent_len)
        vs = ssl_data.window_starts(va_idx, parent_len)
        if len(ts):
            tr_starts.append(ts + base)
        if len(vs):
            va_starts.append(vs + base)
        bigs.append(oh)
        base += len(oh)
        if verbose:
            print(f"  [assemble] {s['sid']} train_win={len(ts)} val_win={len(vs)}",
                  flush=True)
    big = np.concatenate(bigs, 0).astype(np.float32)
    tr = np.concatenate(tr_starts) if tr_starts else np.array([], np.int64)
    va = np.concatenate(va_starts) if va_starts else np.array([], np.int64)
    return big, tr, va


# --------------------------------------------------------------------------- train + probe
# Pretext tasks (Mask / Forecast / Contrastive) live in the pluggable `pretext` package —
# futures_foundation/finetune/pretext/. Add a new pretrain experiment there, not here. The
# orchestrator only resolves a task via get_pretext(...) and calls reserve/train/gate/finalize.
def _train(big, tr, va, cfg, control='real'):
    """Train one config under a control via its pretext task -> (best_encoder_state, history)."""
    return get_pretext(cfg.get('pretext', 'mask')).train(big, tr, va, cfg, control)


def _probe_state(big, va, seq, state, *, model_id, device, seed, folds=1, verbose=True):
    """Probe a trained encoder state vs vanilla -> the probe dict (regime/vol/structure).
    Saves to a temp ckpt so ssl_probe can load it through the normal path. folds>1 -> k-fold CV
    per probe (robust deltas for ranking candidates)."""
    import tempfile
    import torch
    from . import ssl_probe
    fd, tmp = tempfile.mkstemp(suffix='.pt'); os.close(fd)
    torch.save(state, tmp)
    try:
        return ssl_probe.run_probe(big, va, seq, tmp, model_id=model_id, device=device,
                                   seed=seed, folds=folds, verbose=verbose)
    finally:
        os.remove(tmp)


def _passes(probe_res, std, margin=0.0, dir_margin=0.0, pretext='mask'):
    """Report-only gate on the PROBE (representation content), delegated to the pretext task.
    Each task (MaskTask/ForecastTask/ContrastiveTask) owns its own pass/fail rule — see
    PretextTask.gate + `_decide`. Kept as a thin function for callers/tests."""
    return get_pretext(pretext).gate(probe_res, std, margin, dir_margin)


# ------------------------------------------------------------------------------- save/probe
def _finalize(big, tr, va, state, probe_res, cfg, *, out_path, controls, holdout_start,
              val_frac, streams, history, verbose):
    """Save the chosen encoder + report. Controls are PROBE-BASED diagnostics: train
    shuffle/random with the chosen cfg and probe EACH vs vanilla -> real_delta vs
    control_delta (real - shuffle > 0 => temporal order contributed to the useful
    representation). The contrastive loss is not used for the verdict."""
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    torch.save(state, out_path)                          # adapted ENCODER state_dict

    ctrl_delta = {}
    for ctrl in controls:
        if ctrl == 'real':
            continue
        if verbose:
            print(f"\n=== control={ctrl} (probe-based diagnostic) ===", flush=True)
        st, _ = _train(big, tr, va, cfg, ctrl)
        r = _probe_state(big, va, cfg['seq'], st, model_id=cfg['model_id'],
                         device=cfg['device'], seed=cfg['seed'],
                         folds=cfg.get('probe_folds', 1), verbose=verbose)
        ctrl_delta[ctrl] = float(r['mean_core_delta'])

    real_delta = (None if probe_res is None else float(probe_res['mean_core_delta']))
    temporal = (None if (real_delta is None or 'shuffle' not in ctrl_delta)
                else real_delta - ctrl_delta['shuffle'])
    verdict = {
        'all_pass': bool(probe_res is not None and probe_res['learns_regime_vol_structure']),
        'learns_regime_vol_structure': (None if probe_res is None
                                        else bool(probe_res['learns_regime_vol_structure'])),
        'real_delta': real_delta,
        'control_delta': ctrl_delta,
        'temporal_signal': temporal,        # real - shuffle (>0 => order contributed)
    }
    report = {'verdict': verdict, 'probe': probe_res, 'control_delta': ctrl_delta,
              'config': {k: cfg[k] for k in cfg if k not in
                         ('verbose', 'device', 'model_id', 'compile_model')},
              'holdout_start': holdout_start, 'val_frac': val_frac, 'bars': int(len(big)),
              'tickers': sorted({s['ticker'] for s in streams}),
              'tfs': sorted({s['tf'] for s in streams}), 'history': history, 'ckpt': out_path}
    with open(out_path + '.report.json', 'w') as f:
        json.dump(report, f, indent=2, default=float)
    if verbose:
        print(f"\n[ssl] saved encoder -> {out_path}\n[ssl] VERDICT: {verdict}", flush=True)
    return verdict


# ------------------------------------------------------------------------------- entrypoints
def _load_assemble(data_dir, tickers, tfs, seq, max_jitter, val_frac, holdout_start, verbose,
                   forecast_parent=0):
    streams = ssl_data.load_ohlcv(data_dir, tickers, tfs, verbose=verbose)
    big, tr, va = assemble(streams, seq=seq, max_jitter=max_jitter, val_frac=val_frac,
                           holdout_start=holdout_start, forecast_parent=forecast_parent, verbose=verbose)
    if verbose:
        print(f"[ssl] bars={len(big)} train_win={len(tr)} val_win={len(va)} "
              f"streams={len(streams)}", flush=True)
    if len(tr) == 0 or len(va) == 0:
        raise ValueError("no train/val windows — check seq/max_jitter vs data length")
    return streams, big, tr, va


def _base_cfg(**kw):
    """Default SSL config (one place). seq = the probe/embed window; max_jitter reserves the
    probe's forward horizon. Stage-2 forecast knobs: horizons (multi-horizon candle prediction),
    context_lengths (variable input). Only known keys are kept."""
    d = dict(seq=64, max_jitter=16, new_channels=8, mask_ratio=0.4, epochs=60,
             steps_per_epoch=200, batch=1024, lr=1e-4, weight_decay=0.05, patience=8,
             model_id='paris-noah/Mantis-8M', compile_model=False, device=None,
             seed=0, verbose=True, backbone_ckpt=None,
             pretext='mask',                                  # 'mask' (1) | 'forecast' (2) | 'contrastive' (3)
             # stage-2 multi-horizon / variable-context candle forecasting:
             horizons=(5, 10, 20, 25), context_lengths=(64, 100, 150, 200),
             grad_clip=1.0, clamp=10.0,
             # forecast supervision OBJECTIVE (pluggable, no if-chains): 'candle_mse' (original) |
             # 'candle_direction' (candle MSE + BCE on sign(fwd close move) via dir_weight). The Optuna
             # sweep searches this + the knobs below to maximize downstream WR.
             objective='candle_mse',
             # OPTIONAL forecast direction-head squeeze (0 = off / backward-compat; >0 adds BCE on
             # sign of the forward close move -> trains the encoder to be direction-aware for WR):
             dir_weight=0.0, dir_close_ch=3,
             # stage-3 trend contrastive v3 (FORWARD trend-vs-chop, BARRIER-EXCURSION key): multi-
             # positive InfoNCE grouped by the FUTURE window's direction x barrier excursion (how far
             # price RAN before retracing one unit — the WR@3R-shaped ordering statistic the forecast
             # doesn't regress; v2's path-efficiency was redundant with stage-2 -> key_gap ~0).
             # Self-supervised, anti-shortcut (key is target-side, not computable from the input).
             # contrast_horizon = future bars the key reads; pos_cap caps key-positives per anchor.
             temperature=0.1, crop_max=0.2, proj_dim=128, contrast_horizon=25, pos_cap=64,
             # crash-safe progressive best-save + resume + anti-forgetting layer-freeze (ALL pretexts,
             # real run only; controls never touch the ckpt). ckpt_path is set to out_path by loop_ssl.
             ckpt_path=None, resume=False, freeze_encoder_layers=0,
             probe_folds=1)                                   # k-fold CV per probe (robust)
    d.update({k: v for k, v in kw.items() if v is not None and k in d})
    return d


def loop_ssl(data_dir=None, *, tickers=None, tfs=None, controls=('shuffle', 'random'),
             out_path='mantis_ssl_ohlcv.pt', probe=True, probe_margin=0.0, dir_margin=0.0,
             holdout_start='2026-01-01', val_frac=0.1, **cfg_over):
    """Train the SSL encoder ONCE and save it (no Optuna). pretext='mask' = stage-1 masked
    modeling; pretext='forecast' = stage-2 multi-horizon / variable-context candle seq2seq
    (warm-started from stage-1 via backbone_ckpt). Then PROBE vs vanilla + shuffle/random controls
    as diagnostics (gate = report-only), and write the encoder + report."""
    cfg = _base_cfg(**cfg_over)
    cfg['ckpt_path'] = out_path              # progressive best-save target (crash-safe); real run only
    verbose = cfg['verbose']
    pretext = cfg.get('pretext', 'mask')
    # each pretext task declares how much window to reserve (forecast: ctx+horizon;
    # contrastive: ctx; mask: none) — no pretext if-chain here.
    fc_reserve = get_pretext(pretext).reserve(cfg)
    streams, big, tr, va = _load_assemble(data_dir, tickers, tfs, cfg['seq'], cfg['max_jitter'],
                                           val_frac, holdout_start, verbose, forecast_parent=fc_reserve)
    state, hist = _train(big, tr, va, cfg, 'real')
    std = float(hist[-1]['std'])
    best_ep = min(hist, key=lambda h: h['val_loss'])
    fc_skill = best_ep.get('skill')                       # forecast skill vs copy-now (None for mask)
    probe_res = (_probe_state(big, va, cfg['seq'], state, model_id=cfg['model_id'],
                              device=cfg['device'], seed=cfg['seed'],
                              folds=cfg.get('probe_folds', 1), verbose=verbose) if probe else None)
    ok, detail = _passes(probe_res, std, probe_margin, dir_margin, pretext)
    history = [{'source': 'default', 'best_val': float(best_ep['val_loss']), 'std': std,
                'forecast_skill': fc_skill, 'gate_ok': bool(ok), **detail}]
    verdict = _finalize(big, tr, va, state, probe_res, cfg, out_path=out_path, controls=controls,
                        holdout_start=holdout_start, val_frac=val_frac, streams=streams,
                        history=history, verbose=verbose)
    verdict['history'] = history
    get_pretext(pretext).finalize_verdict(verdict, fc_skill, probe_res)   # pretext-specific fields
    return verdict


def run_ssl(data_dir=None, *, controls=('shuffle', 'random'),
            out_path='mantis_ssl_ohlcv.pt', probe=True, holdout_start='2026-01-01',
            val_frac=0.1, tickers=None, tfs=None, **cfg_over):
    """Thin alias of loop_ssl (kept for callers/tests). loop_ssl trains once and saves."""
    return loop_ssl(data_dir, tickers=tickers, tfs=tfs, controls=controls, out_path=out_path,
                    probe=probe, holdout_start=holdout_start, val_frac=val_frac, **cfg_over)


def main():
    p = argparse.ArgumentParser(description="Mantis OHLCV SSL — masked (stage 1) or multi-horizon seq2seq (stage 2)")
    p.add_argument('--data-dir', default=os.environ.get('DATA_DIR'))
    p.add_argument('--out', default=os.environ.get('OUT_PATH', 'mantis_ssl_ohlcv.pt'))
    p.add_argument('--tickers', default=os.environ.get('TICKERS'))
    p.add_argument('--tfs', default=os.environ.get('TFS', '1min,3min,5min,15min'))
    p.add_argument('--pretext', default=os.environ.get('PRETEXT', 'mask'), choices=['mask', 'forecast'])
    p.add_argument('--backbone-ckpt', default=os.environ.get('BACKBONE_CKPT'))  # warm-start (stage 2)
    p.add_argument('--horizons', default=os.environ.get('HORIZONS', '5,10,20,25,50'))
    p.add_argument('--context-lengths', default=os.environ.get('CONTEXT_LENGTHS', '64,100,150,200'))
    p.add_argument('--seq', type=int, default=int(os.environ.get('SEQ', '64')))
    p.add_argument('--max-jitter', type=int, default=int(os.environ.get('MAX_JITTER', '16')))
    p.add_argument('--new-channels', type=int, default=int(os.environ.get('NEW_C', '8')))
    p.add_argument('--batch', type=int, default=int(os.environ.get('BATCH', '1024')))
    p.add_argument('--epochs', type=int, default=int(os.environ.get('EPOCHS', '60')))
    p.add_argument('--steps', type=int, default=int(os.environ.get('STEPS', '200')))
    p.add_argument('--lr', type=float, default=float(os.environ.get('LR', '1e-4')))
    p.add_argument('--val-frac', type=float, default=float(os.environ.get('VAL_FRAC', '0.1')))
    p.add_argument('--holdout-start', default=os.environ.get('HOLDOUT_START', '2026-01-01'))
    p.add_argument('--controls', default=os.environ.get('CONTROLS', 'shuffle,random'))
    p.add_argument('--no-probe', action='store_true', default=os.environ.get('NO_PROBE') == '1')
    p.add_argument('--device', default=os.environ.get('DEVICE'))
    p.add_argument('--compile', action='store_true', default=os.environ.get('COMPILE') == '1')
    p.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '0')))
    a = p.parse_args()
    loop_ssl(data_dir=a.data_dir, out_path=a.out,
             tickers=(a.tickers.split(',') if a.tickers else None), tfs=a.tfs.split(','),
             controls=tuple(a.controls.split(',')), probe=not a.no_probe,
             holdout_start=a.holdout_start, val_frac=a.val_frac, seq=a.seq, max_jitter=a.max_jitter,
             new_channels=a.new_channels, batch=a.batch, epochs=a.epochs, steps_per_epoch=a.steps,
             lr=a.lr, device=a.device, compile_model=a.compile, seed=a.seed, pretext=a.pretext,
             backbone_ckpt=a.backbone_ckpt,
             horizons=tuple(int(x) for x in a.horizons.split(',')),
             context_lengths=tuple(int(x) for x in a.context_lengths.split(',')))


if __name__ == '__main__':
    main()
