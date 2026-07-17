"""Masked-modeling SSL pretraining of the Mantis backbone (orchestrator) — "BERT for
futures".

BERT-style masked modeling on raw OHLCV across the 9 futures tickers x {1,3,5,15,30,60}min:
mask a fraction of bars and reconstruct them from context (in _ssl_torch.train_ssl_mask),
so the encoder learns regime/volatility/structure. Produces a DOMAIN-ADAPTED ENCODER CHECKPOINT
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
import hashlib
import json
import os

import numpy as np

from . import ssl_data
from .pretext import PRETEXTS, PretextTask, get_pretext   # noqa: F401 (pluggable pretext registry)


def assemble(streams, *, seq, max_jitter, val_frac, holdout_start, forecast_parent=0,
             val_start=None, train_start=None, return_groups=False, verbose=True,
             allow_aligned_market_gaps=False):
    """Concatenate all stream OHLCV into one big [T, 5] array + global parent-window start
    positions for the (leak-safe, 2026-excluded) train/val split. Each window reserves enough
    bars for its consumers: seq+max_jitter (probe/mask) OR forecast_parent (stage-2 = max context
    length + max horizon), whichever is larger — so context+future stay in-stream."""
    parent_len = max(seq + max_jitter, int(forecast_parent))
    bigs, tr_starts, va_starts, base = [], [], [], 0
    tr_bounds, va_bounds, tr_labels, va_labels, row_bounds = [], [], [], [], []
    objective_row_bounds = []
    tr_times, va_times, stream_bar_ns = [], [], []
    tr_cursor = va_cursor = 0
    for s in streams:
        oh = s['ohlcv']
        # Purge one full parent window at temporal boundaries. This is stricter than merely
        # preventing overlap and keeps checkpoint selection away from train/holdout adjacency.
        tr_idx, va_idx = ssl_data.time_split(s['ts'], val_frac, holdout_start,
                                             embargo=parent_len - 1, val_start=val_start,
                                             train_start=train_start)
        import pandas as pd
        tf_delta = pd.Timedelta(s['tf'])
        stream_bar_ns.append(int(tf_delta.value))
        # A context longer than one futures session must span maintenance/weekends. Treat
        # consecutive observed chart bars as consecutive without fabricating/filling closures;
        # require grid alignment and cap closures at four days. Contract rolls remain hard
        # boundaries, and longer outages still split. Shorter contexts retain exact cadence.
        crosses_session = (bool(allow_aligned_market_gaps) or
                           parent_len * tf_delta > pd.Timedelta('23h'))
        session_gap = pd.Timedelta('4D') if crosses_session else None
        start_kw = dict(timestamps=s['ts'], expected_delta=tf_delta, max_gap=session_gap,
                        segment_ids=s.get('contract_id'))
        ts = ssl_data.window_starts(tr_idx, parent_len, **start_kw)
        vs = ssl_data.window_starts(va_idx, parent_len, **start_kw)
        # Defense in depth: OOS rows are not merely absent from eligible starts; they are
        # physically absent from the tensor handed to torch.  A future sampler bug therefore
        # cannot accidentally index July-2025+ data during this V2 lineage.
        if holdout_start is not None:
            cut = ssl_data._utc_timestamp(holdout_start)
            usable_rows = int(np.searchsorted(
                pd.DatetimeIndex(s['ts']).asi8, cut.value, side='left'))
        else:
            usable_rows = len(oh)
        row_bounds.append((base, base + usable_rows))
        contract_id = s.get('contract_id')
        if usable_rows:
            if contract_id is None:
                local_edges = np.asarray([0, usable_rows], np.int64)
            else:
                contract_id = np.asarray(contract_id)[:usable_rows]
                local_edges = np.r_[
                    0, np.flatnonzero(contract_id[1:] != contract_id[:-1]) + 1, usable_rows,
                ].astype(np.int64)
            objective_row_bounds.extend(
                (base + int(lo), base + int(hi))
                for lo, hi in zip(local_edges[:-1], local_edges[1:]) if hi > lo
            )
        if len(ts):
            tr_starts.append(ts + base)
            tr_times.append(pd.DatetimeIndex(s['ts']).asi8[ts])
            tr_bounds.append((tr_cursor, tr_cursor + len(ts)))
            tr_labels.append(s['sid'])
            tr_cursor += len(ts)
        if len(vs):
            va_starts.append(vs + base)
            va_times.append(pd.DatetimeIndex(s['ts']).asi8[vs])
            va_bounds.append((va_cursor, va_cursor + len(vs)))
            va_labels.append(s['sid'])
            va_cursor += len(vs)
        bigs.append(oh[:usable_rows])
        base += usable_rows
        if verbose:
            print(f"  [assemble] {s['sid']} train_win={len(ts)} val_win={len(vs)}",
                  flush=True)
    big = np.concatenate(bigs, 0).astype(np.float32)
    tr = np.concatenate(tr_starts) if tr_starts else np.array([], np.int64)
    va = np.concatenate(va_starts) if va_starts else np.array([], np.int64)
    if not return_groups:
        return big, tr, va
    groups = {
        'train_bounds': np.asarray(tr_bounds, dtype=np.int64).reshape(-1, 2),
        'val_bounds': np.asarray(va_bounds, dtype=np.int64).reshape(-1, 2),
        'row_bounds': np.asarray(row_bounds, dtype=np.int64).reshape(-1, 2),
        # Objective labels that inspect future rows must be computed independently inside these
        # stream/contract segments.  Concatenated-array boundaries are never valid market legs.
        'objective_row_bounds': np.asarray(objective_row_bounds, dtype=np.int64).reshape(-1, 2),
        'train_labels': tuple(tr_labels), 'val_labels': tuple(va_labels),
        # Aligned one-to-one with train/validation starts. Stage 2 consumes these internally so
        # temporal rules are expressed in elapsed time, not in incomparable per-timeframe bars.
        'train_start_times_ns': (np.concatenate(tr_times).astype(np.int64) if tr_times
                                 else np.array([], dtype=np.int64)),
        'val_start_times_ns': (np.concatenate(va_times).astype(np.int64) if va_times
                               else np.array([], dtype=np.int64)),
        'stream_bar_ns': np.asarray(stream_bar_ns, dtype=np.int64),
    }
    return big, tr, va, groups


# --------------------------------------------------------------------------- train + probe
# Pretext tasks (Mask / Forecast / Contrastive) live in the pluggable `pretext` package —
# futures_foundation/finetune/pretext/. Add a new pretrain experiment there, not here. The
# orchestrator only resolves a task via get_pretext(...) and calls reserve/train/gate/finalize.
def _train(big, tr, va, cfg, control='real'):
    """Train one config under a control via its pretext task -> (best_encoder_state, history)."""
    return get_pretext(cfg.get('pretext', 'mask')).train(big, tr, va, cfg, control)


def _probe_state(big, va, seq, state, *, model_id, device, seed, model_version=None,
                 folds=1, group_ids=None, timestamps=None, span_ns=None, verbose=True,
                 preprocessing=None):
    """Probe a trained encoder state vs vanilla -> the probe dict (regime/vol/structure).
    Saves to a temp ckpt so ssl_probe can load it through the normal path. Production probes use
    stream-aware expanding walk-forward folds with a full context+target embargo."""
    import tempfile
    import torch
    from . import ssl_probe
    fd, tmp = tempfile.mkstemp(suffix='.pt'); os.close(fd)
    torch.save(state, tmp)
    try:
        return ssl_probe.run_probe(big, va, seq, tmp, model_id=model_id,
                                   model_version=model_version, device=device,
                                   seed=seed, folds=folds, group_ids=group_ids,
                                   timestamps=timestamps, span_ns=span_ns,
                                   preprocessing=preprocessing,
                                   verbose=verbose)
    finally:
        os.remove(tmp)


def _passes(probe_res, std, margin=0.0, dir_margin=0.0, pretext='mask'):
    """Report-only gate on the PROBE (representation content), delegated to the pretext task.
    Each task (MaskTask/ForecastTask/ContrastiveTask) owns its own pass/fail rule — see
    PretextTask.gate + `_decide`. Kept as a thin function for callers/tests."""
    return get_pretext(pretext).gate(probe_res, std, margin, dir_margin)


# ------------------------------------------------------------------------------- save/probe
def _finalize(big, tr, va, state, probe_res, cfg, *, out_path, controls, val_start,
              holdout_start, val_frac, streams, history, probe_starts,
              probe_group_ids, probe_timestamps, probe_span_ns, verbose):
    """Save the chosen encoder + report. Controls are PROBE-BASED diagnostics: train
    shuffle/random with the chosen cfg and probe EACH vs vanilla -> real_delta vs
    control_delta (real - shuffle > 0 => temporal order contributed to the useful
    representation). The contrastive loss is not used for the verdict."""
    import torch
    from .pretext._torch.common import make_deployment_bundle
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    # Keep the already-valid checkpoint intact if the process or machine dies while
    # writing the final selected encoder.  The epoch-level best saves use the same
    # temp-file + atomic-replace discipline in pretext/_torch/common.py.
    tmp_out = f'{out_path}.tmp'
    try:
        torch.save(state, tmp_out)
        os.replace(tmp_out, out_path)                    # adapted ENCODER state_dict
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
    bundle_path = out_path + '.bundle.pt'
    train_contexts = (cfg.get('context_lengths') if cfg.get('pretext') == 'forecast'
                      else (cfg.get('seq'),))
    bundle = make_deployment_bundle(
        state, model_id=cfg['model_id'], model_version=cfg.get('model_version'),
        channels=int(big.shape[1]), train_context_lengths=train_contexts,
        preprocessing=cfg['preprocessing'])
    tmp_bundle = bundle_path + '.tmp'
    try:
        torch.save(bundle, tmp_bundle)
        os.replace(tmp_bundle, bundle_path)
    finally:
        if os.path.exists(tmp_bundle):
            os.remove(tmp_bundle)

    ctrl_delta = {}
    ctrl_probe = {}
    for ctrl in controls:
        if ctrl == 'real':
            continue
        if verbose:
            print(f"\n=== control={ctrl} (probe-based diagnostic) ===", flush=True)
        st, _ = _train(big, tr, va, cfg, ctrl)
        r = _probe_state(big, probe_starts, cfg['seq'], st, model_id=cfg['model_id'],
                         model_version=cfg.get('model_version'),
                         device=cfg['device'], seed=(cfg.get('probe_seed')
                                                     if cfg.get('probe_seed') is not None
                                                     else cfg['seed']),
                         folds=cfg.get('probe_folds', 1), group_ids=probe_group_ids,
                         timestamps=probe_timestamps, span_ns=probe_span_ns,
                         preprocessing=cfg['preprocessing'],
                         verbose=verbose)
        ctrl_probe[ctrl] = r
        ctrl_delta[ctrl] = float(r['mean_core_delta'])

    real_delta = (None if probe_res is None else float(probe_res['mean_core_delta']))
    temporal = (None if (real_delta is None or 'shuffle' not in ctrl_delta)
                else real_delta - ctrl_delta['shuffle'])
    verdict = {
        'all_pass': bool(history and history[0].get('gate_ok')),
        'promotion_schema': (history[0].get('gate_schema') if history else None),
        'learns_regime_vol_structure': (None if probe_res is None
                                        else bool(probe_res['learns_regime_vol_structure'])),
        'real_delta': real_delta,
        'control_delta': ctrl_delta,
        'temporal_signal': temporal,        # real - shuffle (>0 => order contributed)
    }
    probe_hash = hashlib.sha256()
    for value in (probe_starts, probe_group_ids, probe_timestamps):
        array = np.ascontiguousarray(value)
        probe_hash.update(str(array.shape).encode())
        probe_hash.update(array.view(np.uint8))
    report = {'verdict': verdict, 'probe': probe_res, 'control_delta': ctrl_delta,
              'control_probe': ctrl_probe,
              'config': {k: cfg[k] for k in cfg if k not in
                         ('verbose', 'device', 'compile_model', 'train_group_bounds',
                          'val_group_bounds', 'train_start_times_ns', 'val_start_times_ns',
                          'stream_bar_ns', 'objective_row_bounds')},
              'val_start': val_start, 'holdout_start': holdout_start,
              'val_frac': val_frac, 'bars': int(len(big)),
              'tickers': sorted({s['ticker'] for s in streams}),
              'tfs': sorted({s['tf'] for s in streams}),
              'sampling': {
                  'mode': 'uniform_stream_then_uniform_window',
                  'gap_policy': (
                      ('all contexts allow aligned observed-market closures <=4D; '
                       'no fill; contract rolls and longer gaps split')
                      if cfg.get('allow_aligned_market_gaps') else
                      ('contexts >23h allow aligned observed-bar closures <=4D; '
                       'no fill; contract rolls and longer gaps split')),
                  'stream_order': [s['sid'] for s in streams],
                  'objective_segments': int(len(cfg.get('objective_row_bounds', ()))),
                  'probe_sample_sha256': probe_hash.hexdigest(),
                  'train_group_windows': np.diff(np.asarray(cfg['train_group_bounds']), axis=1)
                                           .reshape(-1).tolist(),
                  'validation_group_windows': np.diff(np.asarray(cfg['val_group_bounds']), axis=1)
                                                .reshape(-1).tolist(),
              },
              'history': history, 'ckpt': out_path, 'deployment_bundle': bundle_path,
              'training_state': out_path + '.train.pt'}
    with open(out_path + '.report.json', 'w') as f:
        json.dump(report, f, indent=2, default=float)
    if verbose:
        print(f"\n[ssl] saved encoder -> {out_path}\n"
              f"[ssl] saved deployment bundle -> {bundle_path}\n[ssl] VERDICT: {verdict}",
              flush=True)
    return verdict


# ------------------------------------------------------------------------------- entrypoints
def _load_assemble(data_dir, tickers, tfs, seq, max_jitter, val_frac, holdout_start, verbose,
                   forecast_parent=0, val_start=None, train_start=None, return_groups=False,
                   allow_aligned_market_gaps=False):
    from pathlib import Path
    cache_manifest = Path(data_dir) / 'TOURNAMENT_CACHE.json'
    if cache_manifest.is_file():
        # Lazy import avoids a module cycle at import time (tournament_data itself reuses
        # assemble). The cache contains only the immutable train+validation interval.
        from .tournament_data import load_cache
        streams = load_cache(data_dir, tickers, tfs, verbose=verbose)
    else:
        streams = ssl_data.load_ohlcv(
            data_dir, tickers, tfs, verbose=verbose,
            start=train_start, end=holdout_start if train_start is not None else None,
        )
    assembled = assemble(streams, seq=seq, max_jitter=max_jitter, val_frac=val_frac,
                         holdout_start=holdout_start, forecast_parent=forecast_parent,
                         val_start=val_start, train_start=train_start,
                         return_groups=return_groups, verbose=verbose,
                         allow_aligned_market_gaps=allow_aligned_market_gaps)
    big, tr, va = assembled[:3]
    if verbose:
        print(f"[ssl] bars={len(big)} train_win={len(tr)} val_win={len(va)} "
              f"streams={len(streams)}", flush=True)
    if len(tr) == 0 or len(va) == 0:
        raise ValueError("no train/val windows — check seq/max_jitter vs data length")
    if return_groups and (len(assembled[3]['train_bounds']) != len(streams)
                          or len(assembled[3]['val_bounds']) != len(streams)):
        have_tr, have_va = set(assembled[3]['train_labels']), set(assembled[3]['val_labels'])
        missing_tr = [s['sid'] for s in streams if s['sid'] not in have_tr]
        missing_va = [s['sid'] for s in streams if s['sid'] not in have_va]
        raise ValueError(f"stream coverage failure: zero train={missing_tr}, zero val={missing_va}")
    return (streams, big, tr, va, assembled[3]) if return_groups else (streams, big, tr, va)


def _balanced_group_sample(starts, bounds, max_windows=20000, seed=0,
                           return_group_ids=False):
    """Deterministically cap each stream equally for macro-balanced representation probes."""
    starts, bounds = np.asarray(starts, np.int64), np.asarray(bounds, np.int64)
    if len(bounds) == 0:
        return (starts, np.zeros(len(starts), np.int64)) if return_group_ids else starts
    rng = np.random.default_rng(seed)
    per = max(1, int(max_windows) // len(bounds))
    chunks, group_chunks = [], []
    for group, (lo, hi) in enumerate(bounds):
        rows = np.arange(int(lo), int(hi), dtype=np.int64)
        if len(starts) > max_windows and len(rows) > per:
            rows = rng.choice(rows, per, replace=False)
        chunks.append(starts[np.sort(rows)])
        group_chunks.append(np.full(len(rows), group, np.int64))
    sampled = np.concatenate(chunks)
    sampled_groups = np.concatenate(group_chunks)
    order = np.argsort(sampled, kind='stable')
    sampled, sampled_groups = sampled[order], sampled_groups[order]
    return (sampled, sampled_groups) if return_group_ids else sampled


def _probe_start_timestamps(starts, group_ids, streams, row_bounds):
    """Map concatenated tensor row starts back to their source stream timestamps."""
    starts, group_ids = np.asarray(starts, np.int64), np.asarray(group_ids, np.int64)
    row_bounds = np.asarray(row_bounds, np.int64)
    if len(starts) != len(group_ids) or len(row_bounds) != len(streams):
        raise ValueError("invalid probe timestamp mapping inputs")
    out = np.empty(len(starts), np.int64)
    for group in np.unique(group_ids):
        rows = np.flatnonzero(group_ids == group)
        local = starts[rows] - int(row_bounds[group, 0])
        ts = np.asarray(streams[int(group)]['ts']).astype('datetime64[ns]').astype(np.int64)
        if np.any(local < 0) or np.any(local >= len(ts)):
            raise ValueError(f"probe starts escape stream group {group}")
        out[rows] = ts[local]
    return out


def _base_cfg(**kw):
    """Default SSL config (one place). seq = the probe/embed window; max_jitter reserves the
    probe's forward horizon. Stage-2 forecast knobs: horizons (multi-horizon candle prediction),
    context_lengths (variable input). Only known keys are kept."""
    d = dict(seq=64, max_jitter=16, new_channels=5, mask_ratio=0.4, epochs=60,
             steps_per_epoch=200, batch=1024, lr=1e-4, weight_decay=0.05, patience=8,
             model_id='paris-noah/Mantis-8M', model_version=None,
             compile_model=False, device=None,
             seed=0, verbose=True, backbone_ckpt=None,
             preprocessing='per_window_per_channel_zscore_v1',
             pretext='mask',                                  # canonical: mask (1) -> contrastive (2) -> forecast (3)
             # mask SpanBERT mode (shared by the mask pretext): span_mean>0 = corrupt CONTIGUOUS
             # multi-bar spans (geometric mean span_mean, clipped span_max); 0 = single-bar masking.
             span_mean=0.0, span_max=10,
             # Anti-forgetting: keep clean embeddings close to the encoder state at stage entry.
             feature_anchor_weight=0.0,
             # stage-3 multi-horizon / variable-context candle forecasting:
             horizons=(5, 10, 20, 25), context_lengths=(64, 100, 150, 200),
             grad_clip=1.0, clamp=10.0,
             # forecast supervision OBJECTIVE (pluggable, no if-chains): 'candle_mse' (original) |
             # 'candle_direction' (candle MSE + BCE on sign(fwd close move) via dir_weight). The Optuna
             # sweep searches this + the knobs below to maximize downstream WR.
             objective='candle_mse',
             # OPTIONAL forecast direction-head squeeze (0 = off / backward-compat; >0 adds BCE on
             # sign of the forward close move -> trains the encoder to be direction-aware for WR):
             dir_weight=0.0, dir_close_ch=3,
             # stage-2.5 forecast_dist faithfulness knobs (defaults = the original refine-study
             # behavior): mse_weight 0 = PURE Chronos loss (no MSE anchor); quantile_taus 'bolt9'
             # = the full 9-level quantile head; bins_k = bin-classification resolution.
             mse_weight=1.0, quantile_taus='lohi', bins_k=41,
             # Stage-2 v2 uses elapsed-time offsets relative to each context's wall-clock span,
             # independent per-observation augmentation, equal anchor weighting, and conservative
             # synchronized-negative masking. bar_offset_v1 remains an explicit audit baseline.
             temperature=0.1, crop_max=0.2, proj_dim=128,
             pos_deltas=(2, 16, 64), far_min=512, aug_noise=0.10, aug_scale=0.20,
             aug_tmask=0.15, vol_weight=0.0, w_clip=4.0, metrics_n=768,
             contrastive_objective='elapsed_time_v2',
             contrastive_reserve_contexts=None,
             positive_gap_fractions=(0.6, 1.0, 2.0), max_positive_overlap=0.5,
             positive_tolerance_fraction=0.20, negative_min_contexts=4.0,
             sync_exclusion_minutes=60.0, min_valid_negatives=1,
             # stage-4 TURN-ELECTRA (replaced-TURN detection — the discriminative slot): spans are
             # CENTERED ON DETECTED TURNS (local swing highs/lows, neighborhood ±turn_w) with prob
             # turn_bias (0 = uniform span-ELECTRA, the placement ablation); a weak generator
             # (gen_width) fills each masked turn = a SYNTHETIC FAKE TURN; the encoder labels every
             # bar real/replaced (rtd_weight) while the encoder-side recon anchor (recon_weight)
             # keeps the embedding tied to the data (0 = pure discrimination / drift risk).
             # span_mean/span_max (shared above) set span lengths; electra coerces span_mean<=0 to 4.
             rtd_weight=5.0, recon_weight=1.0, gen_width=48, turn_w=3, turn_bias=0.85,
             # stage-2.6 NEXT-LEG forecasting (bars; pure-fractal pivots, NO ATR):
             leg_cap=256, leg_w=1.0, leg_k=2,
             # std_guard: IN-LOOP drift halt — training stops (without saving that epoch)
             # the moment emb_std exceeds it; 0 = off. Guards the anchored-discrimination
             # runs against slow drift that val loss rewards (val micro-improves while the
             # representation walks off the data).
             std_guard=1.6,
             # Separate crash-safe artifacts: best encoder deployment weights + exact full
             # epoch-boundary training state. Controls never touch either artifact.
             ckpt_path=None, resume=False, freeze_encoder_layers=0,
             train_group_bounds=None, val_group_bounds=None,
             train_start_times_ns=None, val_start_times_ns=None, stream_bar_ns=None,
             objective_row_bounds=None,
             val_batches=None, allow_aligned_market_gaps=False, probe_seed=None,
             probe_folds=1)                         # expanding walk-forward probe folds
    d.update({k: v for k, v in kw.items() if v is not None and k in d})
    return d


def loop_ssl(data_dir=None, *, tickers=None, tfs=None, controls=('shuffle', 'random'),
             out_path='mantis_ssl_ohlcv.pt', probe=True, probe_margin=0.0, dir_margin=0.0,
             train_start=None, val_start=None, holdout_start='2026-01-01', val_frac=0.1,
             **cfg_over):
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
    streams, big, tr, va, groups = _load_assemble(
        data_dir, tickers, tfs, cfg['seq'], cfg['max_jitter'], val_frac, holdout_start,
        verbose, forecast_parent=fc_reserve, val_start=val_start, train_start=train_start,
        return_groups=True,
        allow_aligned_market_gaps=cfg['allow_aligned_market_gaps'])
    cfg['train_group_bounds'] = groups['train_bounds']
    cfg['val_group_bounds'] = groups['val_bounds']
    cfg['train_start_times_ns'] = groups['train_start_times_ns']
    cfg['val_start_times_ns'] = groups['val_start_times_ns']
    cfg['stream_bar_ns'] = groups['stream_bar_ns']
    cfg['objective_row_bounds'] = groups['objective_row_bounds']
    state, hist = _train(big, tr, va, cfg, 'real')
    std = float(hist[-1]['std'])
    best_ep = min(hist, key=lambda h: h['val_loss'])
    fc_skill = best_ep.get('skill')                       # forecast skill vs copy-now (None for mask)
    probe_starts, probe_group_ids = _balanced_group_sample(
        va, groups['val_bounds'], seed=(cfg.get('probe_seed')
                                       if cfg.get('probe_seed') is not None else cfg['seed']),
        return_group_ids=True)
    probe_timestamps = _probe_start_timestamps(
        probe_starts, probe_group_ids, streams, groups['row_bounds'])
    import pandas as pd
    probe_span_ns = int(max(pd.Timedelta(s['tf']).value for s in streams)
                        * (cfg['seq'] + cfg['max_jitter']))
    probe_res = (_probe_state(big, probe_starts, cfg['seq'], state, model_id=cfg['model_id'],
                              model_version=cfg.get('model_version'),
                              device=cfg['device'], seed=(cfg.get('probe_seed')
                                                          if cfg.get('probe_seed') is not None
                                                          else cfg['seed']),
                              folds=cfg.get('probe_folds', 1), group_ids=probe_group_ids,
                              timestamps=probe_timestamps, span_ns=probe_span_ns,
                              preprocessing=cfg['preprocessing'],
                              verbose=verbose) if probe else None)
    ok, detail = _passes(probe_res, std, probe_margin, dir_margin, pretext)
    history = [{'source': 'default', 'best_val': float(best_ep['val_loss']), 'std': std,
                'forecast_skill': fc_skill, 'gate_ok': bool(ok),
                'task_diagnostics': {k: v for k, v in best_ep.items()
                                     if k not in ('epoch', 'train_loss', 'val_loss', 'std')},
                **detail}]
    verdict = _finalize(big, tr, va, state, probe_res, cfg, out_path=out_path,
                        controls=controls, val_start=val_start, holdout_start=holdout_start,
                        val_frac=val_frac, streams=streams, history=history,
                        probe_starts=probe_starts, probe_group_ids=probe_group_ids,
                        probe_timestamps=probe_timestamps, probe_span_ns=probe_span_ns,
                        verbose=verbose)
    verdict['history'] = history
    verdict['epochs'] = hist                 # per-epoch trainer history (val_loss + task extras,
    #                                          e.g. electra rtd_bal_acc) — learning verification
    get_pretext(pretext).finalize_verdict(verdict, fc_skill, probe_res)   # pretext-specific fields
    return verdict


def run_ssl(data_dir=None, *, controls=('shuffle', 'random'),
            out_path='mantis_ssl_ohlcv.pt', probe=True, holdout_start='2026-01-01',
            val_start=None, val_frac=0.1, tickers=None, tfs=None, **cfg_over):
    """Thin alias of loop_ssl (kept for callers/tests). loop_ssl trains once and saves."""
    return loop_ssl(data_dir, tickers=tickers, tfs=tfs, controls=controls, out_path=out_path,
                    probe=probe, val_start=val_start, holdout_start=holdout_start,
                    val_frac=val_frac, **cfg_over)


def main():
    p = argparse.ArgumentParser(description="Mantis OHLCV SSL — masked (stage 1) or multi-horizon seq2seq (stage 2)")
    p.add_argument('--data-dir', default=os.environ.get('DATA_DIR'))
    p.add_argument('--out', default=os.environ.get('OUT_PATH', 'mantis_ssl_ohlcv.pt'))
    p.add_argument('--tickers', default=os.environ.get('TICKERS'))
    p.add_argument('--tfs', default=os.environ.get('TFS', '1min,3min,5min,15min,30min,60min'))
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
    p.add_argument('--val-start', default=os.environ.get('VAL_START'))
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
             val_start=a.val_start, holdout_start=a.holdout_start, val_frac=a.val_frac,
             seq=a.seq, max_jitter=a.max_jitter,
             new_channels=a.new_channels, batch=a.batch, epochs=a.epochs, steps_per_epoch=a.steps,
             lr=a.lr, device=a.device, compile_model=a.compile, seed=a.seed, pretext=a.pretext,
             backbone_ckpt=a.backbone_ckpt,
             horizons=tuple(int(x) for x in a.horizons.split(',')),
             context_lengths=tuple(int(x) for x in a.context_lengths.split(',')))


if __name__ == '__main__':
    main()
