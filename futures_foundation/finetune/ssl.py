"""Generic self-supervised pretraining of the Mantis backbone (orchestrator).

Temporal contrastive learning on raw OHLCV across the 9 futures tickers x
{1,3,5,15}min. Produces an ADAPTED ENCODER CHECKPOINT (saved to Drive on Colab) that
downstream classifier finetuning starts from
(build_model(..., backbone_ckpt=...) / BACKBONE_CKPT=... in the WF/produce driver).

Torch-free at import (the GPU trainer in _ssl_torch + the probe's torch bits load
lazily) so data assembly, the generalization gate, the Optuna search wiring, and the
contract are testable without the torch/mantis stack.

Generalization is GATED + OPTUNA-TUNED, mirroring WF/produce:
  * TIME-SPLIT val NT-Xent early-stop      (generalize forward; 2026 EXCLUDED)
  * REAL vs SHUFFLE vs RANDOM controls     (REAL must beat both -> real structure)
  * representation-COLLAPSE guard          (embed std / alignment / uniformity)
  * if it doesn't generalize -> OPTUNA tunes lr/temp/reg/aug for a config that does
  * FINAL check: a linear PROBE shows the frozen embedding encodes regime / vol /
    structure better than vanilla Mantis (ssl_probe) — "useful for downstream"

Colab usage: see colab/mantis_ssl_pretrain.py.
"""
import argparse
import json
import os

import numpy as np

from . import ssl_data

_AUG_KEYS = ('resize', 'jitter', 'scale', 'warp')


def assemble(streams, *, seq, max_jitter, val_frac, holdout_start, verbose=True):
    """Concatenate all stream OHLCV into one big [T, 5] array + global parent-window
    start positions for the (leak-safe, 2026-excluded) train/val split."""
    parent_len = seq + max_jitter
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


# --------------------------------------------------------------------------- config split
def _split_cfg(cfg):
    """Separate flat config into (train_kwargs, aug-dict) for train_ssl."""
    cfg = dict(cfg)
    aug = {k: cfg.pop(k) for k in _AUG_KEYS if k in cfg}
    return cfg, aug


# --------------------------------------------------------------------------- one config run
def _run_once(big, tr, va, *, controls, cfg, verbose=True):
    """Train each control with one config; return per-control summaries + REAL state."""
    from . import _ssl_torch
    tk, aug = _split_cfg(cfg)
    by, real_state = {}, None
    for ctrl in controls:
        if verbose:
            print(f"\n=== SSL control={ctrl} ===", flush=True)
        state, hist = _ssl_torch.train_ssl(big, tr, va, control=ctrl, aug=aug, **tk)
        best_val = min(h['val_loss'] for h in hist)
        final_val = hist[-1]['val_loss']
        by[ctrl] = {'best_val': best_val, 'final_std': hist[-1]['std'],
                    'val_gap': final_val - best_val, 'n_epochs': len(hist)}
        if ctrl == 'real':
            real_state = state
    return by, real_state


def _generalizes(by, gap_tol=0.5):
    """REAL beats controls + no collapse + val did not diverge -> generalizes."""
    real = by['real']
    shuf = by.get('shuffle', {}).get('best_val')
    rand = by.get('random', {}).get('best_val')
    beats = ((shuf is None or real['best_val'] < shuf - 1e-3)
             and (rand is None or real['best_val'] < rand - 1e-3))
    no_collapse = real['final_std'] > 0.01
    stable = real['val_gap'] <= gap_tol
    return bool(beats and no_collapse and stable), {
        'real_beats_controls': bool(beats), 'no_collapse': bool(no_collapse),
        'val_stable': bool(stable), 'val_gap': float(real['val_gap'])}


# ------------------------------------------------------------------------------- optuna
def _suggest_ssl(trial):
    """Search the knobs that govern contrastive generalization: optimizer, temperature,
    regularization, and augmentation strength (too weak -> trivial/collapse; too strong
    -> can't align positives)."""
    return dict(
        lr=trial.suggest_float('lr', 3e-5, 5e-4, log=True),
        temp=trial.suggest_float('temp', 0.07, 0.5, log=True),
        weight_decay=trial.suggest_float('weight_decay', 0.01, 0.3, log=True),
        new_channels=trial.suggest_int('new_channels', 4, 12),
        jitter=trial.suggest_float('jitter', 0.0, 0.15),
        warp=trial.suggest_float('warp', 0.0, 0.2),
        resize=(trial.suggest_float('resize_lo', 0.5, 0.9), 1.0),
    )


def _tune_ssl(big, tr, va, base_cfg, *, n_trials=10, tune_epochs=8, tune_steps=80,
              seed=0, verbose=True):
    """Optuna: short REAL-only runs minimizing val NT-Xent (collapse-safe — a collapsed
    encoder cannot lower NT-Xent). Returns the best config merged onto base_cfg if it
    beats the base val, else base_cfg (mirrors tune.tune)."""
    import optuna
    from . import _ssl_torch
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def val_of(cfg):
        tk, aug = _split_cfg(dict(cfg, epochs=tune_epochs, steps_per_epoch=tune_steps,
                                  verbose=False))
        _, hist = _ssl_torch.train_ssl(big, tr, va, control='real', aug=aug, **tk)
        return min(h['val_loss'] for h in hist)

    base_val = val_of(base_cfg)

    def objective(trial):
        return val_of(dict(base_cfg, **_suggest_ssl(trial)))

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    improved = study.best_value < base_val - 1e-3
    best = dict(base_cfg, **study.best_params) if improved else dict(base_cfg)
    best.pop('resize_lo', None)
    if 'resize_lo' in study.best_params and improved:
        best['resize'] = (study.best_params['resize_lo'], 1.0)
    if verbose:
        print(f"  [optuna ssl] base_val={base_val:.4f} best={study.best_value:.4f} "
              f"{'-> use tuned' if improved else '-> keep defaults'}", flush=True)
    return best, {'base_val': base_val, 'best_val': study.best_value, 'improved': improved}


# ------------------------------------------------------------------------------- save/probe
def _finalize(big, va, real_state, by, cfg, gen_detail, *, out_path, probe, holdout_start,
              val_frac, streams, model_id, device, seed, verbose):
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    torch.save(real_state, out_path)                     # adapted ENCODER state_dict

    probe_res = None
    if probe:
        from . import ssl_probe
        if verbose:
            print("\n=== FINAL CHECK: linear probe (regime / vol / structure) ===", flush=True)
        probe_res = ssl_probe.run_probe(big, va, cfg['seq'], out_path, model_id=model_id,
                                        device=device, seed=seed, verbose=verbose)

    verdict = {
        **gen_detail,
        'real_best_val': float(by['real']['best_val']),
        'shuffle_best_val': by.get('shuffle', {}).get('best_val'),
        'random_best_val': by.get('random', {}).get('best_val'),
        'final_std': float(by['real']['final_std']),
        'learns_regime_vol_structure': (None if probe_res is None
                                        else probe_res['learns_regime_vol_structure']),
        'probe_mean_core_delta': (None if probe_res is None
                                  else probe_res['mean_core_delta']),
    }
    verdict['all_pass'] = bool(gen_detail['generalizes']
                               and (probe_res is None
                                    or probe_res['learns_regime_vol_structure']))
    report = {'verdict': verdict, 'config': {k: cfg[k] for k in cfg if k not in
              ('verbose', 'device', 'model_id', 'compile_model')},
              'controls': by, 'probe': probe_res, 'holdout_start': holdout_start,
              'val_frac': val_frac, 'bars': int(len(big)),
              'tickers': sorted({s['ticker'] for s in streams}),
              'tfs': sorted({s['tf'] for s in streams}), 'ckpt': out_path}
    with open(out_path + '.report.json', 'w') as f:
        json.dump(report, f, indent=2, default=float)
    if verbose:
        print(f"\n[ssl] saved encoder -> {out_path}\n[ssl] VERDICT: {verdict}", flush=True)
    return verdict


# ------------------------------------------------------------------------------- entrypoints
def _load_assemble(data_dir, tickers, tfs, seq, max_jitter, val_frac, holdout_start, verbose):
    streams = ssl_data.load_ohlcv(data_dir, tickers, tfs, verbose=verbose)
    big, tr, va = assemble(streams, seq=seq, max_jitter=max_jitter, val_frac=val_frac,
                           holdout_start=holdout_start, verbose=verbose)
    if verbose:
        print(f"[ssl] bars={len(big)} train_win={len(tr)} val_win={len(va)} "
              f"streams={len(streams)}", flush=True)
    if len(tr) == 0 or len(va) == 0:
        raise ValueError("no train/val windows — check seq/max_jitter vs data length")
    return streams, big, tr, va


def _base_cfg(**kw):
    """Default training config (one place; loop_ssl tunes a subset)."""
    d = dict(seq=64, max_jitter=8, new_channels=8, proj_dim=128, temp=0.2, epochs=60,
             steps_per_epoch=200, batch=1024, lr=1e-4, weight_decay=0.05, patience=8,
             model_id='paris-noah/Mantis-8M', compile_model=False, device=None,
             seed=0, verbose=True, resize=(0.7, 1.0), jitter=0.05, scale=0.1, warp=0.1)
    d.update({k: v for k, v in kw.items() if v is not None})
    return d


def loop_ssl(data_dir=None, *, tickers=None, tfs=None, controls=('real', 'shuffle', 'random'),
             out_path='mantis_ssl_ohlcv.pt', probe=True, n_trials=10, max_iters=2,
             gap_tol=0.5, holdout_start='2026-01-01', val_frac=0.1, **cfg_over):
    """Overfit-gated, Optuna-tuned SSL: run default -> if it doesn't generalize, tune a
    config that does -> re-run -> final probe -> save the generalized encoder. Returns
    the verdict dict (+ writes <out_path> and <out_path>.report.json)."""
    cfg = _base_cfg(**cfg_over)
    verbose = cfg['verbose']
    streams, big, tr, va = _load_assemble(data_dir, tickers, tfs, cfg['seq'],
                                          cfg['max_jitter'], val_frac, holdout_start, verbose)
    history = []
    for it in range(max_iters):
        src = 'default' if it == 0 else 'optuna-tuned'
        if verbose:
            print(f"\n[ssl-loop] iter {it} · {src} config", flush=True)
        by, real_state = _run_once(big, tr, va, controls=controls, cfg=cfg, verbose=verbose)
        gen, detail = _generalizes(by, gap_tol)
        detail['generalizes'] = gen
        history.append({'iter': it, 'source': src, 'real_best_val': by['real']['best_val'],
                        **detail})
        if gen or it == max_iters - 1:
            break
        if verbose:
            print(f"[ssl-loop] does NOT generalize ({detail}) -> Optuna", flush=True)
        cfg, _ = _tune_ssl(big, tr, va, cfg, n_trials=n_trials, seed=cfg['seed'],
                           verbose=verbose)
        cfg = _base_cfg(**cfg)                            # re-fill any popped defaults

    detail['generalizes'] = gen
    verdict = _finalize(big, va, real_state, by, cfg, detail, out_path=out_path, probe=probe,
                        holdout_start=holdout_start, val_frac=val_frac, streams=streams,
                        model_id=cfg['model_id'], device=cfg['device'], seed=cfg['seed'],
                        verbose=verbose)
    verdict['history'] = history
    return verdict


def run_ssl(data_dir=None, *, controls=('real', 'shuffle', 'random'),
            out_path='mantis_ssl_ohlcv.pt', probe=True, holdout_start='2026-01-01',
            val_frac=0.1, tickers=None, tfs=None, **cfg_over):
    """Single-config SSL run (no Optuna). Thin wrapper used for simple/fast runs and by
    tests; loop_ssl is the full generalization process."""
    return loop_ssl(data_dir, tickers=tickers, tfs=tfs, controls=controls,
                    out_path=out_path, probe=probe, n_trials=0, max_iters=1,
                    holdout_start=holdout_start, val_frac=val_frac, **cfg_over)


def main():
    p = argparse.ArgumentParser(description="Mantis OHLCV contrastive SSL (gated + Optuna)")
    p.add_argument('--data-dir', default=os.environ.get('DATA_DIR'))
    p.add_argument('--out', default=os.environ.get('OUT_PATH', 'mantis_ssl_ohlcv.pt'))
    p.add_argument('--tickers', default=os.environ.get('TICKERS'))
    p.add_argument('--tfs', default=os.environ.get('TFS', '1min,3min,5min,15min'))
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
             seed=a.seed)


if __name__ == '__main__':
    main()
