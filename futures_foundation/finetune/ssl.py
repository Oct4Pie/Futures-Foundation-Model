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


def assemble(streams, *, seq, max_jitter, val_frac, holdout_start, horizon=0, verbose=True):
    """Concatenate all stream OHLCV into one big [T, 5] array + global parent-window
    start positions for the (leak-safe, 2026-excluded) train/val split. The parent window
    reserves room for BOTH the probe's forward horizon (max_jitter) and, for the seq2seq
    forecast pretext, the forecast horizon — so a window holds context+future in-stream."""
    parent_len = seq + max(max_jitter, horizon)
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
def _train(big, tr, va, cfg, control='real'):
    """Train one config under a control ('real'|'shuffle'|'random'). pretext='mask' = BERT-
    style masked modeling (stage 1); pretext='forecast' = causal seq2seq forecasting (stage 2,
    warm-started via backbone_ckpt). Both trainers swallow unknown kwargs (**_ignore), so the
    shared cfg (carrying both mask_ratio and horizon) is safe to pass either way. Returns
    (best_encoder_state, history)."""
    from . import _ssl_torch
    kw = {k: v for k, v in cfg.items() if k != 'pretext'}
    if cfg.get('pretext', 'mask') == 'forecast':
        return _ssl_torch.train_ssl_forecast(big, tr, va, control=control, **kw)
    return _ssl_torch.train_ssl_mask(big, tr, va, control=control, **kw)


def _probe_state(big, va, seq, state, *, model_id, device, seed, verbose=True):
    """Probe a trained encoder state vs vanilla -> the probe dict (regime/vol/structure).
    Saves to a temp ckpt so ssl_probe can load it through the normal path."""
    import tempfile
    import torch
    from . import ssl_probe
    fd, tmp = tempfile.mkstemp(suffix='.pt'); os.close(fd)
    torch.save(state, tmp)
    try:
        return ssl_probe.run_probe(big, va, seq, tmp, model_id=model_id, device=device,
                                   seed=seed, verbose=verbose)
    finally:
        os.remove(tmp)


def _passes(probe_res, std, margin=0.0, dir_margin=0.0, pretext='mask'):
    """GATE on the PROBE (representation content), NOT the loss.

    pretext='mask' (ORIGINAL stage-1, UNCHANGED): REAL must encode regime/vol/structure better
    than vanilla (mean_core_delta > margin) and not collapse.

    pretext='forecast' (stage-2, ANTI-SHORTCUT): a shortcut embedding can lift the easy in-window
    DESCRIPTIVE stats (vol/trend_eff/range_expand) while the genuinely predictive FORWARD targets
    barely move — so the descriptive average is not enough. We additionally require, vs vanilla:
      * descriptive content does NOT regress      (descriptive_delta >= 0)
      * FORWARD MOVE SIZE improves                 (fwd_absmove_delta > margin)
      * FORWARD DIRECTION does NOT regress         (fwd_dir_delta >= dir_margin; default 0 =
        non-regression — directional AUC is noisy, so the floor is 'don't get worse', with the
        actual value always reported for the human / downstream-edge verdict)
    """
    no_collapse = bool(std > 0.01)
    detail = {'no_collapse': no_collapse}
    if probe_res is None:
        return no_collapse, {**detail, 'probe': None}
    detail.update({'mean_core_delta': float(probe_res['mean_core_delta']),
                   'descriptive_delta': float(probe_res.get('descriptive_delta', 0.0)),
                   'fwd_absmove_delta': float(probe_res.get('fwd_absmove_delta', 0.0)),
                   'fwd_dir_delta': float(probe_res.get('fwd_dir_delta', 0.0)),
                   'forward_score': float(probe_res.get('forward_score', 0.0)),
                   'learns_regime_vol_structure': bool(probe_res['learns_regime_vol_structure'])})
    if pretext == 'forecast':
        desc_ok = bool(probe_res.get('descriptive_delta', 0.0) >= -1e-9)
        fwd_size_ok = bool(probe_res.get('fwd_absmove_delta', 0.0) > margin)
        fwd_dir_ok = bool(probe_res.get('fwd_dir_delta', 0.0) >= dir_margin)
        ok = bool(no_collapse and desc_ok and fwd_size_ok and fwd_dir_ok)
        detail.update({'descriptive_ok': desc_ok, 'fwd_size_ok': fwd_size_ok,
                       'fwd_dir_ok': fwd_dir_ok})
        return ok, detail
    ok = bool(probe_res['mean_core_delta'] > margin and no_collapse)   # original mask gate
    return ok, detail


# ------------------------------------------------------------------------------- optuna
def _suggest_ssl(trial, pretext='mask'):
    """Search the SSL knobs that govern generalization (maximizing the probe delta): optimizer,
    capacity, and the pretext-specific knob — mask ratio (masked) or forecast horizon (seq2seq)."""
    d = dict(lr=trial.suggest_float('lr', 3e-5, 5e-4, log=True),
             weight_decay=trial.suggest_float('weight_decay', 0.01, 0.3, log=True),
             new_channels=trial.suggest_int('new_channels', 4, 12))
    if pretext == 'forecast':
        d['horizon'] = trial.suggest_int('horizon', 8, 32)
    else:
        d['mask_ratio'] = trial.suggest_float('mask_ratio', 0.2, 0.6)
    return d


def _tune_ssl(big, tr, va, base_cfg, *, n_trials=10, tune_epochs=8, tune_steps=80,
              seed=0, verbose=True):
    """Optuna MAXIMIZING the probe — NOT a lower loss (loss is blind: noise scores as well as
    real). Objective is pretext-matched: forecast (stage 2) maximizes the FORWARD-predictive
    score (fwd move size + direction vs vanilla; anti-shortcut — the easy descriptive average
    can't carry it); mask (stage 1, ORIGINAL) maximizes mean_core_delta as before. Returns the
    best config if it beats the base, else base."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pretext = base_cfg.get('pretext', 'mask')

    def delta_of(cfg):
        st, _ = _train(big, tr, va, dict(cfg, epochs=tune_epochs,
                                         steps_per_epoch=tune_steps, verbose=False), 'real')
        r = _probe_state(big, va, cfg['seq'], st, model_id=cfg['model_id'],
                         device=cfg['device'], seed=seed, verbose=False)
        return float(r['forward_score'] if pretext == 'forecast' else r['mean_core_delta'])

    base = delta_of(base_cfg)

    def objective(trial):
        return delta_of(dict(base_cfg, **_suggest_ssl(trial, base_cfg.get('pretext', 'mask'))))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    improved = study.best_value > base + 1e-4
    best = dict(base_cfg, **study.best_params) if improved else dict(base_cfg)
    if verbose:
        print(f"  [optuna ssl] base_fwd={base:+.4f} best_fwd={study.best_value:+.4f} "
              f"{'-> use tuned' if improved else '-> keep defaults'}", flush=True)
    return best, {'base_delta': base, 'best_delta': study.best_value, 'improved': improved}


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
                         device=cfg['device'], seed=cfg['seed'], verbose=verbose)
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
                   horizon=0):
    streams = ssl_data.load_ohlcv(data_dir, tickers, tfs, verbose=verbose)
    big, tr, va = assemble(streams, seq=seq, max_jitter=max_jitter, val_frac=val_frac,
                           holdout_start=holdout_start, horizon=horizon, verbose=verbose)
    if verbose:
        print(f"[ssl] bars={len(big)} train_win={len(tr)} val_win={len(va)} "
              f"streams={len(streams)}", flush=True)
    if len(tr) == 0 or len(va) == 0:
        raise ValueError("no train/val windows — check seq/max_jitter vs data length")
    return streams, big, tr, va


def _base_cfg(**kw):
    """Default masked-modeling config (one place; loop_ssl tunes a subset). max_jitter
    reserves the forward-probe horizon (in-stream). Only known keys are kept, so stray
    callers can't inject junk into the trainer."""
    d = dict(seq=64, max_jitter=16, new_channels=8, mask_ratio=0.4, epochs=60,
             steps_per_epoch=200, batch=1024, lr=1e-4, weight_decay=0.05, patience=8,
             model_id='paris-noah/Mantis-8M', compile_model=False, device=None,
             seed=0, verbose=True,
             pretext='mask', horizon=16, backbone_ckpt=None)  # stage-2 seq2seq: forecast + warm-start
    d.update({k: v for k, v in kw.items() if v is not None and k in d})
    return d


def loop_ssl(data_dir=None, *, tickers=None, tfs=None, controls=('shuffle', 'random'),
             out_path='mantis_ssl_ohlcv.pt', probe=True, n_trials=10, max_iters=2,
             probe_margin=0.0, dir_margin=0.0, holdout_start='2026-01-01', val_frac=0.1,
             **cfg_over):
    """Probe-GATED, Optuna-tuned SSL. Each iter: train REAL -> PROBE vs vanilla -> gate on
    the PROBE (does it encode regime/vol/structure better than vanilla), NOT on the
    contrastive loss (which is blind — noise scores as well as real). If it doesn't pass,
    Optuna MAXIMIZES the probe delta and we re-run. Saves the best-probe encoder + report
    (with probe-based shuffle/random controls as the temporal diagnostic)."""
    cfg = _base_cfg(**cfg_over)
    verbose = cfg['verbose']
    # forecast pretext: reserve room for the LARGEST horizon Optuna may later pick (suggest_int
    # upper bound = 32) so a tuned-up horizon never reads past the window into another stream /
    # the holdout. mask pretext reserves nothing extra (0).
    fc_reserve = max(int(cfg.get('horizon', 0)), 32) if cfg.get('pretext') == 'forecast' else 0
    streams, big, tr, va = _load_assemble(data_dir, tickers, tfs, cfg['seq'],
                                          cfg['max_jitter'], val_frac, holdout_start, verbose,
                                          horizon=fc_reserve)
    history, best = [], None
    for it in range(max_iters):
        src = 'default' if it == 0 else 'optuna-tuned'
        if verbose:
            print(f"\n[ssl-loop] iter {it} · {src} config", flush=True)
        state, hist = _train(big, tr, va, cfg, 'real')
        std = float(hist[-1]['std'])
        best_ep = min(hist, key=lambda h: h['val_loss'])
        best_val = float(best_ep['val_loss'])
        fc_skill = best_ep.get('skill')               # forecast skill vs copy-last-bar (None for mask)
        probe_res = (_probe_state(big, va, cfg['seq'], state, model_id=cfg['model_id'],
                                  device=cfg['device'], seed=cfg['seed'], verbose=verbose)
                     if probe else None)
        pretext = cfg.get('pretext', 'mask')
        ok, detail = _passes(probe_res, std, probe_margin, dir_margin, pretext)
        history.append({'iter': it, 'source': src, 'best_val': best_val, 'std': std,
                        'forecast_skill': fc_skill, **detail})
        # rank iters by the SAME quantity the gate keys on: forecast (stage 2) = FORWARD-
        # predictive score (anti-shortcut); mask (stage 1, original) = descriptive mean_core.
        delta = (-1e9 if probe_res is None else
                 probe_res['forward_score'] if pretext == 'forecast' else probe_res['mean_core_delta'])
        if best is None or delta > best['delta']:
            best = {'state': state, 'probe': probe_res, 'cfg': dict(cfg), 'delta': delta,
                    'skill': fc_skill}
        if ok or it == max_iters - 1:
            break
        if verbose:
            metric = 'forward_score' if pretext == 'forecast' else 'mean_core_delta'
            print(f"[ssl-loop] {metric}={delta:+.4f} (gate not passed) -> Optuna", flush=True)
        cfg, _ = _tune_ssl(big, tr, va, cfg, n_trials=n_trials, seed=cfg['seed'], verbose=verbose)
        cfg = _base_cfg(**cfg)                            # re-fill any popped defaults

    verdict = _finalize(big, tr, va, best['state'], best['probe'], best['cfg'],
                        out_path=out_path, controls=controls, holdout_start=holdout_start,
                        val_frac=val_frac, streams=streams, history=history, verbose=verbose)
    verdict['history'] = history
    if cfg.get('pretext') == 'forecast':                 # stage-2: forward-predictive diagnostics
        verdict['forecast_skill'] = best.get('skill')    # >0 => beat copy-last-bar (anti-shortcut)
        verdict['forward_score'] = best.get('delta')
        if best.get('probe') is not None:
            verdict['fwd_absmove_delta'] = float(best['probe'].get('fwd_absmove_delta', 0.0))
            verdict['fwd_dir_delta'] = float(best['probe'].get('fwd_dir_delta', 0.0))
    return verdict


def run_ssl(data_dir=None, *, controls=('shuffle', 'random'),
            out_path='mantis_ssl_ohlcv.pt', probe=True, holdout_start='2026-01-01',
            val_frac=0.1, tickers=None, tfs=None, **cfg_over):
    """Single-config SSL run (no Optuna). Thin wrapper used for simple/fast runs and by
    tests; loop_ssl is the full probe-gated process."""
    return loop_ssl(data_dir, tickers=tickers, tfs=tfs, controls=controls,
                    out_path=out_path, probe=probe, n_trials=0, max_iters=1,
                    holdout_start=holdout_start, val_frac=val_frac, **cfg_over)


def main():
    p = argparse.ArgumentParser(description="Mantis OHLCV masked-modeling SSL (probe-gated + Optuna)")
    p.add_argument('--data-dir', default=os.environ.get('DATA_DIR'))
    p.add_argument('--out', default=os.environ.get('OUT_PATH', 'mantis_ssl_ohlcv.pt'))
    p.add_argument('--tickers', default=os.environ.get('TICKERS'))
    p.add_argument('--tfs', default=os.environ.get('TFS', '1min,3min,5min,15min'))
    p.add_argument('--pretext', default=os.environ.get('PRETEXT', 'mask'),
                   choices=['mask', 'forecast'])   # stage 1 = mask (BERT); stage 2 = seq2seq forecast
    p.add_argument('--horizon', type=int, default=int(os.environ.get('HORIZON', '16')))
    p.add_argument('--backbone-ckpt', default=os.environ.get('BACKBONE_CKPT'))  # warm-start (stage 2)
    p.add_argument('--dir-margin', type=float, default=float(os.environ.get('DIR_MARGIN', '0.0')))
    p.add_argument('--seq', type=int, default=int(os.environ.get('SEQ', '64')))
    p.add_argument('--max-jitter', type=int, default=int(os.environ.get('MAX_JITTER', '8')))
    p.add_argument('--new-channels', type=int, default=int(os.environ.get('NEW_C', '8')))
    p.add_argument('--batch', type=int, default=int(os.environ.get('BATCH', '1024')))
    p.add_argument('--epochs', type=int, default=int(os.environ.get('EPOCHS', '60')))
    p.add_argument('--steps', type=int, default=int(os.environ.get('STEPS', '200')))
    p.add_argument('--lr', type=float, default=float(os.environ.get('LR', '1e-4')))
    p.add_argument('--val-frac', type=float, default=float(os.environ.get('VAL_FRAC', '0.1')))
    p.add_argument('--holdout-start', default=os.environ.get('HOLDOUT_START', '2026-01-01'))
    p.add_argument('--controls', default=os.environ.get('CONTROLS', 'real,shuffle,random'))
    p.add_argument('--n-trials', type=int, default=int(os.environ.get('N_TRIALS', '10')))
    p.add_argument('--max-iters', type=int, default=int(os.environ.get('MAX_ITERS', '2')))
    p.add_argument('--no-probe', action='store_true', default=os.environ.get('NO_PROBE') == '1')
    p.add_argument('--device', default=os.environ.get('DEVICE'))
    p.add_argument('--compile', action='store_true', default=os.environ.get('COMPILE') == '1')
    p.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '0')))
    a = p.parse_args()
    loop_ssl(data_dir=a.data_dir, out_path=a.out,
             tickers=(a.tickers.split(',') if a.tickers else None), tfs=a.tfs.split(','),
             controls=tuple(a.controls.split(',')), probe=not a.no_probe,
             n_trials=a.n_trials, max_iters=a.max_iters, holdout_start=a.holdout_start,
             val_frac=a.val_frac, seq=a.seq, max_jitter=a.max_jitter,
             new_channels=a.new_channels, batch=a.batch, epochs=a.epochs,
             steps_per_epoch=a.steps, lr=a.lr, device=a.device, compile_model=a.compile,
             seed=a.seed, pretext=a.pretext, horizon=a.horizon, backbone_ckpt=a.backbone_ckpt,
             dir_margin=a.dir_margin)


if __name__ == '__main__':
    main()
