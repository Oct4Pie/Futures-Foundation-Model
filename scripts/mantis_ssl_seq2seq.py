# ==============================================================================
# MANTIS SSL STAGE 2 — SEQ2SEQ FORECASTING FT on raw OHLCV (Colab GPU)
# ==============================================================================
#
# A SECOND self-supervised stage that CONTINUES (fine-tunes) the masked-SSL encoder
# (scripts/mantis_ssl_pretrain.py -> mantis_ssl_ohlcv.pt) with a CAUSAL SEQ2SEQ
# FORECASTING pretext: encode a CONTEXT window of past bars, predict the NEXT
# `horizon` bars (full OHLCV). To forecast the future the encoder MUST model forward
# dynamics — momentum continuation, volatility persistence, trend vs. reversal — i.e.
# the trend-prediction the downstream buy/sell classifier needs. Learning the data
# better as a forecaster is the path to better trend prediction.
#
# PRICE-ACTION ONLY: raw OHLCV (open/high/low/close/volume). No derived/handcraft
# features anywhere in this stage — same corpus as stage 1.
#
# WARM-START: starts from the stage-1 checkpoint (BACKBONE_CKPT) — this is an FT on
# top of the current SSL encoder, not a fresh pretrain.
#
# OUTPUT: an adapted ENCODER checkpoint. Downstream classifier finetuning starts from
# it exactly like stage 1:  BACKBONE_CKPT=<this .pt>  (WF/produce), or
# build_model(..., backbone_ckpt=<this .pt>).
#
# Validity gates (identical rig to stage 1):
#   * TIME-SPLIT val forecast MSE early-stop      (generalize forward; 2026 EXCLUDED)
#   * PROBE vs vanilla = the GATE: frozen embedding must predict regime/vol/structure +
#     forward buy/sell move BETTER than the un-adapted backbone
#   * REAL vs SHUFFLE vs RANDOM = probe-based diagnostic (only REAL has a predictable
#     continuation; time-scrambled / noise do not)
# ==============================================================================


# ==============================================================================
# CELL 1 — SETUP  (clone FFM @ mantis branch, install, mantis-tsfm)
# ==============================================================================
import os, subprocess
os.chdir('/content')

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

print('Cloning FFM repo (mantis branch)...')
os.system('rm -rf /content/Futures-Foundation-Model')
r = subprocess.run(
    ['git', 'clone', '--branch', 'mantis',
     'https://github.com/johnamcruz/Futures-Foundation-Model.git',
     '/content/Futures-Foundation-Model'],
    capture_output=True, text=True)
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


# ==============================================================================
# CELL 2 — CONFIGURATION  (Drive paths + hyperparameters + pre-flight)
# ==============================================================================
import os, torch

# ── PATHS (Drive) ──
# DATA_DIR must contain the raw CSVs named  {TICKER}_{TF}.csv  with columns
# datetime,open,high,low,close,volume  (e.g. ES_3min.csv).  SAME corpus as stage 1.
DATA_DIR  = '/content/drive/MyDrive/Futures Data'
WARM_CKPT = '/content/drive/MyDrive/AI_Models/mantis_ssl_ohlcv.pt'      # stage-1 encoder (warm-start)
OUT_PATH  = '/content/drive/MyDrive/AI_Models/mantis_ssl_seq2seq.pt'    # stage-2 adapted encoder

# ── CORPUS (identical to stage 1) ──
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']      # all 9
TFS     = ['1min', '3min', '5min', '15min']                            # all 4 TFs
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1                    # last 10% of each stream's pre-2026 bars = val

# ── SEQ2SEQ FORECASTING ──
SEQ          = 64      # context window (bars); interpolated to Mantis's native 512 internally
HORIZON      = 16      # forecast horizon: predict the next HORIZON bars from the context
NEW_CHANNELS = 8       # channel-combiner output (OHLCV=5 -> NEW_CHANNELS)
MAX_JITTER   = 16      # forward bars reserved for the buy/sell probe (in-stream)

# ── TRAINING (GPU-max) ──
BATCH        = 1024    # drop if OOM (512 / 768).
EPOCHS       = 60
STEPS        = 200     # steps per epoch
LR           = 1e-4
PATIENCE     = 8
# ── STABILITY (anti-blowup) ──
# A FLAT/compressed context window has ~0 std, so the forward delta target (fut-last)/ctx_std
# explodes -> diverging loss (train ~5e4). These bound it: clamp the standardized values, clip
# the gradient norm. Lower CLAMP (e.g. 6) = more conservative; raise only if losses stay tame.
CLAMP        = 10.0
GRAD_CLIP    = 1.0
CONTROLS     = ['shuffle', 'random']   # probe-based diagnostics (did temporal order help)
COMPILE      = False   # torch.compile(encoder) — try True on A100/L4 for extra speed
SEED         = 0

# ── CHANNEL-WEIGHTED LOSS (price-path option) ──
# Per-channel loss weights (O,H,L,C,V). None = equal (every channel weighted the same).
# PRICE-PATH hypothesis (emphasize close, ignore near-unpredictable volume) for trend detection:
#   CHANNEL_WEIGHTS = [1.0, 1.0, 1.0, 2.0, 0.0]
# Leave None to let the Optuna scan (below) FIND the best weighting instead of guessing.
CHANNEL_WEIGHTS = None

# ── GENERALIZATION (probe gate -> Optuna, like stage 1 / WF / produce) ──
N_TRIALS     = 20      # Optuna trials maximizing the FORWARD-predictive probe score
MAX_ITERS    = 2       # default run, then one Optuna-tuned re-run
PROBE        = True    # GATE: probe regime/vol/structure + forward buy/sell move vs vanilla
# OPTUNA_SCAN=True -> deliberately run the Optuna scan to FIND the best channel weights + knobs
# (lr/horizon/capacity), maximizing the forward-predictive score, even if the default passes the
# gate. This is the "scan to be sure" mode for picking the weighting. False = only tune on gate-fail.
OPTUNA_SCAN  = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
if device.type != 'cuda':
    print('⚠️  No CUDA — SSL is designed for a Colab GPU runtime (Runtime > Change runtime type > GPU).')

# ── PRE-FLIGHT: fail in seconds if a path is wrong (before any GPU) ──
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f'DATA_DIR does not exist:\n  {DATA_DIR}')
if not os.path.exists(WARM_CKPT):
    raise FileNotFoundError(
        f'WARM_CKPT (stage-1 encoder) not found:\n  {WARM_CKPT}\n'
        f'Run scripts/mantis_ssl_pretrain.py first, or point WARM_CKPT at the masked-SSL ckpt.')
found = [f'{tk}_{tf}' for tk in TICKERS for tf in TFS
         if os.path.exists(os.path.join(DATA_DIR, f'{tk}_{tf}.csv'))]
if not found:
    raise FileNotFoundError(
        f'No {{TICKER}}_{{TF}}.csv files found under {DATA_DIR}.\n'
        f'Expected e.g. {DATA_DIR}/ES_3min.csv with columns '
        f'datetime,open,high,low,close,volume.')
print(f'✅ PRE-FLIGHT: found {len(found)}/{len(TICKERS)*len(TFS)} CSVs under {DATA_DIR}')
print(f'   seq2seq FORECAST | warm-start <- {WARM_CKPT}')
print(f'   corpus {TICKERS} x {TFS} | SEQ={SEQ} HORIZON={HORIZON} BATCH={BATCH} '
      f'EPOCHS={EPOCHS} controls={CONTROLS}')
print(f'   OUTPUT -> {OUT_PATH}')


# ==============================================================================
# CELL 3 — RUN SEQ2SEQ FT  (probe-gated + Optuna; saves encoder ckpt + .report.json)
# ==============================================================================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH,
    tickers=TICKERS, tfs=TFS,
    pretext='forecast', horizon=HORIZON, backbone_ckpt=WARM_CKPT,   # <- FT on stage-1 encoder
    grad_clip=GRAD_CLIP, clamp=CLAMP,                               # <- stability (anti-blowup)
    channel_weights=CHANNEL_WEIGHTS,                               # <- price-path option (None=equal)
    seq=SEQ, max_jitter=MAX_JITTER, new_channels=NEW_CHANNELS,
    batch=BATCH, epochs=EPOCHS, steps_per_epoch=STEPS, lr=LR,
    patience=PATIENCE, val_frac=VAL_FRAC, holdout_start=HOLDOUT_START,
    controls=tuple(CONTROLS), probe=PROBE, n_trials=N_TRIALS, max_iters=MAX_ITERS,
    force_tune=OPTUNA_SCAN,                                         # <- scan weights+knobs "to be sure"
    device=device.type, compile_model=COMPILE, seed=SEED,
)

print('\n' + '=' * 60)
print('  SSL STAGE 2 (SEQ2SEQ FORECAST) VERDICT')
print('=' * 60)
for k, v in verdict.items():
    print(f'  {k:>22}: {v}')
print('=' * 60)
print(f'\nadapted encoder  -> {OUT_PATH}')
print(f'report           -> {OUT_PATH}.report.json')
print('\nDownstream use:  BACKBONE_CKPT=<this .pt>  in the WF/produce driver, or')
print('                 build_model(..., backbone_ckpt=<this .pt>)')
