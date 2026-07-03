# ==============================================================================
# MANTIS SSL STAGE 2.5 — DISTRIBUTIONAL MIXTURE REFINE (Moirai mixture-NLL) (Colab GPU)
# ==============================================================================
#
# Full-trains the forecast_dist v3 sweep WINNER (trial 4) into a NEW encoder checkpoint.
# WARM-STARTS from the PROMOTED stage-2 seq2seq encoder (mantis_ssl_seq2seq.pt — the proven
# 54.7%@1/d 2026 OOS base) and continues training with the Moirai MIXTURE-DENSITY objective:
#
#   * candle_mixture — per horizon, predict the future close move's full DISTRIBUTION as a
#     K=3 mixture (Student-t fat tail + Normal body + low-variance 'chop' Normal), trained by
#     mixture NLL + an anti-collapse load-balance penalty. WR@3R is a TAIL question; the MSE
#     mean-regressor is structurally blind to exactly that — the mixture gives it gradient.
#   * mse_weight=1 keeps the candle-MSE anchor (sweep: pure-Chronos mse=0 lost; the mean stays).
#   * Same SSL targets as stage-2 (context-standardized future candles, raw OHLCV, no ATR/R,
#     no labels) — purely self-supervised; only the LOSS GEOMETRY changes.
#
# SWEEP EVIDENCE (forecast_dist_wr_sweep_9tk_4tf_v3, stopped converged at 30/50): trial 4 =
# WR@3R 50.7% x avgWinR 13.35 -> SCORE 6.769 on pre-2026 val vs warm-start baseline 6.684
# (49.8%); unbeaten for 26 trials; the mixture/nc=3 region replicated 50.0-50.9% WR across 6+
# configs (trials 20/25/26/27); trial 4 is LINEAR-CLEAN (mlpWR 50.6% ~ linear -> ships with the
# plain logistic head, no MLP needed). Guards healthy (mix_entropy ~0.81, mix_mean_df ~3.4).
#
# SAFETY: writes a DISTINCT checkpoint (mantis_ssl_mixture.pt). It NEVER overwrites
# mantis_ssl_seq2seq.pt (the incumbent) or mantis_ssl_ohlcv.pt (stage-1) — a preflight guard
# hard-fails if OUT_PATH resolves to a protected name. seq2seq stays the shipped base until
# THIS beats it on the one-shot 2026 (feedback_holdout_offlimits + the promote gate).
#
# OUTPUT: an adapted ENCODER checkpoint, used downstream exactly like seq2seq:
#   S3_CKPT=<this .pt>  python3 colabs/mantis_2026_benchmark.py   (A/B vs seq2seq on 2026)
# ==============================================================================


# ======================================= CELL 1 — SETUP (clone FFM @ mantis, install) ==========
import os, subprocess
os.chdir('/content')

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

print('Cloning FFM repo (mantis branch)...')
os.system('rm -rf /content/Futures-Foundation-Model')
r = subprocess.run(['git', 'clone', '--branch', 'mantis',
                    'https://github.com/johnamcruz/Futures-Foundation-Model.git',
                    '/content/Futures-Foundation-Model'], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr); raise RuntimeError('git clone failed')
print('Cloned')

os.chdir('/content/Futures-Foundation-Model')
os.system('pip install -e . -q 2>&1 | tail -1')
os.system('pip install mantis-tsfm -q 2>&1 | tail -1')

try:
    from futures_foundation.finetune import ssl, ssl_data
    print('FFM + SSL modules import OK')
except ImportError as e:
    print(f'Import failed: {e}\nRestarting runtime — re-run this cell after restart...')
    os.kill(os.getpid(), 9)


# ======================================= CELL 2 — CONFIG + pre-flight ==========================
import os, torch

# ── PATHS (Drive) ──
# WARM_CKPT = the PROMOTED stage-2 seq2seq (the refine builds ON it, never from scratch).
# OUT_PATH  = a NEW, DISTINCT file — the mixture-refine CANDIDATE. seq2seq is NOT touched: it
# stays the incumbent until this candidate beats it on the one-shot 2026 benchmark.
DATA_DIR  = '/content/drive/MyDrive/Futures Data'
WARM_CKPT = '/content/drive/MyDrive/AI_Models/mantis_ssl_seq2seq.pt'     # stage-2 seq2seq (warm-start)
OUT_PATH  = '/content/drive/MyDrive/AI_Models/mantis_ssl_mixture.pt'     # stage-2.5 mixture (NEW file)

# ── CORPUS (same universe as stage 1/2 — the ruler must not drift) ──
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']      # all 9
TFS     = ['1min', '3min', '5min', '15min']                            # all 4 TFs
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1

# ── MULTI-HORIZON / VARIABLE-CONTEXT (same reserve as stage-2 — comparable window starts) ──
HORIZONS        = (5, 10, 20, 25)             # predict the CANDLE at each (near..far), in bars
CONTEXT_LENGTHS = (64, 100, 150, 200)         # sample a context length per step (short..long)

# ── SWEEP WINNER — trial 4 of forecast_dist_wr_sweep_9tk_4tf_v3 (do NOT round the floats:
# lr/dir_weight are paired with BATCH=512; rounding changes the config that won) ──
OBJECTIVE    = 'candle_mixture'
MSE_WEIGHT   = 1.0                            # keep the candle-MSE anchor (mse=0 pure-Chronos lost)
DIR_WEIGHT   = 0.5478940832105661             # mixture-NLL term weight (the plumbed mix knob)
LR           = 7.131662673882961e-05          # sweep-winner lr (~7.1e-5)
NEW_CHANNELS = 3                              # adapter OHLCV=5 -> 3 (the reproducible top region)
WEIGHT_DECAY = 0.0                            # sweep winner
FREEZE_ENCODER_LAYERS = 0                     # full fine-tune (trial 4). If the WF honest ruler
#   flags DRIFT from the seq2seq base, the fallback is a frozen re-run (freeze=4) to preserve
#   seq2seq's generalization — the sweep's freeze=4 configs are the ready alternates.
QUANTILE_TAUS = 'lohi'                        # inert for candle_mixture (quantile-only knob)
BINS_K        = 41                            # inert for candle_mixture (bins-only knob)

# ── TRAINING (BATCH MATCHES THE SWEEP so the tuned LR transfers) ──
BATCH   = 512         # PARITY with the sweep (lr was tuned at 512; 1024 would need a different lr)
EPOCHS  = 60          # full budget (the sweep used a short 12-epoch proxy to RANK; train to convergence)
STEPS   = 200         # steps/epoch
PATIENCE = 8
CLAMP, GRAD_CLIP = 10.0, 1.0                    # stability
CONTROLS = ()                                  # skip shuffle/random retrains here — the WF honest
                                               # ruler (next step) runs REAL vs SHUFFLE/RANDOM.
PROBE = True                                    # probe vs vanilla (diagnostic, not a gate)
SEED = 0

# ── CRASH-SAFE (best saved PROGRESSIVELY to OUT_PATH; Colab-disconnect resilient) ──
RESUME  = False        # True -> resume from the best saved to OUT_PATH (crash recovery)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
if device.type != 'cuda':
    print('⚠️  No CUDA — SSL is designed for a Colab GPU runtime (Runtime > Change runtime type > GPU).')

# ── PRE-FLIGHT ──
# HARD GUARD: never overwrite the incumbent seq2seq or the stage-1 base. The refine must land on
# a DISTINCT file so promotion is a deliberate decision (post-2026), never an accidental clobber.
_PROTECTED = {'mantis_ssl_seq2seq.pt', 'mantis_ssl_ohlcv.pt'}
if os.path.basename(OUT_PATH) in _PROTECTED:
    raise SystemExit(f'❌ OUT_PATH would overwrite a PROTECTED checkpoint '
                     f'({os.path.basename(OUT_PATH)}) — the refine must write a NEW file.')
if os.path.abspath(OUT_PATH) == os.path.abspath(WARM_CKPT):
    raise SystemExit('❌ OUT_PATH == WARM_CKPT — the refine would overwrite its own warm-start.')
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f'DATA_DIR does not exist:\n  {DATA_DIR}')
if not os.path.exists(WARM_CKPT):
    raise FileNotFoundError(f'WARM_CKPT (stage-2 seq2seq encoder) not found:\n  {WARM_CKPT}\n'
                            f'Point at the promoted base (or run scripts/mantis_ssl_seq2seq.py).')
found = [f'{tk}_{tf}' for tk in TICKERS for tf in TFS
         if os.path.exists(os.path.join(DATA_DIR, f'{tk}_{tf}.csv'))]
if not found:
    raise FileNotFoundError(f'No {{TICKER}}_{{TF}}.csv files under {DATA_DIR}.')
print(f'✅ PRE-FLIGHT: {len(found)}/{len(TICKERS)*len(TFS)} CSVs | warm-start <- {WARM_CKPT}')
print(f'   SWEEP-WINNER (trial 4): obj={OBJECTIVE} mse_w={MSE_WEIGHT} dir_w={DIR_WEIGHT:.4g} '
      f'lr={LR:.2e} nc={NEW_CHANNELS} frz={FREEZE_ENCODER_LAYERS} wd={WEIGHT_DECAY}')
print(f'   BATCH={BATCH} (parity w/ sweep) EPOCHS={EPOCHS} | horizons={HORIZONS} '
      f'context_lengths={CONTEXT_LENGTHS}')
print(f'   OUTPUT -> {OUT_PATH}   (mantis_ssl_seq2seq.pt UNTOUCHED)')


# ======================================= CELL 3 — TRAIN (single run, no Optuna) ================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH, tickers=TICKERS, tfs=TFS,
    pretext='forecast_dist', backbone_ckpt=WARM_CKPT,           # <- stage-2.5 refine, warm-start seq2seq
    horizons=HORIZONS, context_lengths=CONTEXT_LENGTHS, objective=OBJECTIVE,
    mse_weight=MSE_WEIGHT, quantile_taus=QUANTILE_TAUS, bins_k=BINS_K, dir_weight=DIR_WEIGHT,
    new_channels=NEW_CHANNELS, batch=BATCH, epochs=EPOCHS, steps_per_epoch=STEPS, lr=LR,
    weight_decay=WEIGHT_DECAY, patience=PATIENCE, clamp=CLAMP, grad_clip=GRAD_CLIP, val_frac=VAL_FRAC,
    holdout_start=HOLDOUT_START, controls=CONTROLS, probe=PROBE,
    resume=RESUME, freeze_encoder_layers=FREEZE_ENCODER_LAYERS,
    device=device.type, seed=SEED)

print('\n' + '=' * 60 + '\n  SSL STAGE 2.5 (distributional mixture refine) VERDICT\n' + '=' * 60)
for k, v in verdict.items():
    if k != 'history':
        print(f'  {k:>22}: {v}')
print('=' * 60)
print(f'\nadapted encoder  -> {OUT_PATH}   (mantis_ssl_seq2seq.pt UNTOUCHED)')
print(f'report           -> {OUT_PATH}.report.json')
print('\nNext: WF honest ruler on this checkpoint (REAL vs SHUFFLE/RANDOM), THEN the one-shot 2026:')
print(f'  S3_CKPT={OUT_PATH}  python3 colabs/mantis_2026_benchmark.py   (A/B vs seq2seq; promote iff it wins).')
