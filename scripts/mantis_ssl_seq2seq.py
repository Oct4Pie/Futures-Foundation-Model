# ==============================================================================
# MANTIS SSL STAGE 2 — MULTI-HORIZON, VARIABLE-CONTEXT CANDLE seq2seq (Colab GPU)
# ==============================================================================
#
# The stage-2 pretext that teaches the encoder PRICE-ACTION DYNAMICS AT MULTIPLE SCALES.
# Warm-starts from the stage-1 masked-SSL encoder (mantis_ssl_ohlcv.pt) and continues
# training with a forecasting objective:
#
#   * VARIABLE CONTEXT  — each step samples a context length (64/100/150/200 bars), so the
#     encoder learns short AND long history (scale-invariant; interpolated to Mantis's 512).
#   * MULTI-HORIZON     — predict the future CANDLE (OHLCV) at EACH horizon 5/10/20/25 bars ahead
#     (near AND far), forcing multi-timescale trend understanding.
#   * CANDLE TARGET      — predict the actual future candles, CONTEXT-STANDARDIZED (per-channel
#     z-score by the context's own mean/std). NO ATR / NO R (that's volatility-relative). All
#     channels equal; the model uses AND predicts all OHLCV (incl volume).
#
# Anti-shortcut: target = move FROM now (copy-now == predict-zero, punished). All channels equal.
# clamp + grad-clip keep training stable. Price-action ONLY (raw OHLCV) — no derived features.
#
# OUTPUT: an adapted ENCODER checkpoint. Downstream uses it exactly like stage-1:
#   BACKBONE_CKPT=<this .pt>  (WF/produce / benchmark), or build_model(..., backbone_ckpt=<this>).
# The scorecard: does OHLCV-only (embedding, no handcraft) predict trend BETTER than stage-1.
# ==============================================================================


# ======================================= CELL 1 — SETUP (clone FFM @ mantis, install) ==========
import os, subprocess
os.chdir('/content')

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

print('Cloning FFM repo (mantis branch)...')
os.system('rm -rf /content/Futures-Foundation-Model')
r = subprocess.run(['git', 'clone', '--branch', 'main',
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
# This IS the stage-2 seq2seq encoder — the Optuna sweep (forecast_wr_sweep_9tk_4tf trial 1, WR@3R
# 50.3% on pre-2026 val) just found its config. The SHIP GATE is the repo checkpoint + BASE_BACKBONE_CKPT
# pointer: only commit this over checkpoints/mantis_ssl_seq2seq.pt AFTER it beats the current base on
# the one-shot 2026. (The old Drive copy is overwritten, but the current base stays in the repo until then.)
# ENV-OVERRIDABLE so this script serves BOTH the default stage-2 AND the REORDER's forecast-last step:
#   DEFAULT (mask->forecast): warm from stage-1 mask -> mantis_ssl_seq2seq.pt
#   REORDER (mask->contrastive->forecast) STEP 2: warm from the contrastive-from-mask checkpoint,
#     FREEZE early blocks so forecast can't erase the regime, write a DISTINCT file:
#     WARM_CKPT=.../mantis_ssl_regime_from_mask.pt  OUT_PATH=.../mantis_ssl_seq2seq_reordered.pt \
#       FREEZE_ENCODER_LAYERS=3  python3 scripts/mantis_ssl_seq2seq.py
DATA_DIR  = os.environ.get('DATA_DIR', '/content/drive/MyDrive/Futures Data')
WARM_CKPT = os.environ.get('WARM_CKPT', '/content/drive/MyDrive/AI_Models/mantis_ssl_ohlcv.pt')
OUT_PATH  = os.environ.get('OUT_PATH', '/content/drive/MyDrive/AI_Models/mantis_ssl_seq2seq.pt')

# ── CORPUS (same as stage 1) ──
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']      # all 9
TFS     = ['1min', '3min', '5min', '15min']                            # all 4 TFs
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1

# ── MULTI-HORIZON / VARIABLE-CONTEXT candle forecast (sweep-winner) ──
HORIZONS        = (5, 10, 20, 25)             # predict the CANDLE at each (near..far), in bars
CONTEXT_LENGTHS = (64, 100, 150, 200)         # sample a context length per step (short..long)
OBJECTIVE       = 'candle_mse'                # sweep winner: plain candle MSE beat the direction head
NEW_CHANNELS    = 3                            # sweep: nc=3 (adapter OHLCV=5 -> 3) edged 4/5
DIR_WEIGHT      = 0.0                          # candle_mse -> no direction head (0)

# ── TRAINING (sweep-winner; BATCH MATCHES THE SWEEP so the tuned LR transfers) ──
BATCH   = 512         # PARITY with the sweep (lr was tuned at 512; 1024 would need a different lr)
EPOCHS  = 60          # full budget (the sweep used a short 12-epoch proxy to RANK; train to convergence)
STEPS   = 200         # steps/epoch
LR      = 0.00013623475359251814   # sweep-winner lr (~1.4e-4); do NOT round — paired with BATCH=512
WEIGHT_DECAY = 0.1    # sweep winner
PATIENCE = 8
CLAMP, GRAD_CLIP = 10.0, 1.0                    # stability
CONTROLS = ()                                  # skip shuffle/random retrains (fast iteration; judge
                                               # offline). Re-enable ('shuffle','random') for a fresh
                                               # anti-shortcut check on a NEW pretext/target change.
PROBE = True                                    # probe vs vanilla (diagnostic, not a gate)
SEED = 0

# ── CRASH-SAFE + ANTI-FORGETTING (best saved PROGRESSIVELY to OUT_PATH; Colab-disconnect resilient) ──
RESUME  = False        # True -> resume from the best saved to OUT_PATH (crash recovery)
# sweep winner for the DEFAULT lineage = frz=0 (full fine-tune). For the REORDER's forecast-last
# step, set FREEZE_ENCODER_LAYERS=3 to protect the regime the contrastive stage just built.
FREEZE_ENCODER_LAYERS = int(os.environ.get('FREEZE_ENCODER_LAYERS', '0'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
if device.type != 'cuda':
    print('⚠️  No CUDA — SSL is designed for a Colab GPU runtime (Runtime > Change runtime type > GPU).')

# ── PRE-FLIGHT ──
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f'DATA_DIR does not exist:\n  {DATA_DIR}')
if not os.path.exists(WARM_CKPT):
    raise FileNotFoundError(f'WARM_CKPT (warm-start encoder) not found:\n  {WARM_CKPT}')
found = [f'{tk}_{tf}' for tk in TICKERS for tf in TFS
         if os.path.exists(os.path.join(DATA_DIR, f'{tk}_{tf}.csv'))]
if not found:
    raise FileNotFoundError(f'No {{TICKER}}_{{TF}}.csv files under {DATA_DIR}.')
print(f'✅ PRE-FLIGHT: {len(found)}/{len(TICKERS)*len(TFS)} CSVs | warm-start <- {WARM_CKPT}')
print(f'   SWEEP-WINNER: obj={OBJECTIVE} lr={LR:.2e} nc={NEW_CHANNELS} frz={FREEZE_ENCODER_LAYERS} '
      f'wd={WEIGHT_DECAY} BATCH={BATCH} (parity w/ sweep) EPOCHS={EPOCHS}')
print(f'   horizons={HORIZONS} context_lengths={CONTEXT_LENGTHS}')
print(f'   OUTPUT -> {OUT_PATH}')


# ======================================= CELL 3 — TRAIN (single run, no Optuna) ================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH, tickers=TICKERS, tfs=TFS,
    pretext='forecast', backbone_ckpt=WARM_CKPT,               # <- stage-2 forecast, warm-start stage-1
    horizons=HORIZONS, context_lengths=CONTEXT_LENGTHS, objective=OBJECTIVE,
    new_channels=NEW_CHANNELS, batch=BATCH, epochs=EPOCHS, steps_per_epoch=STEPS, lr=LR,
    weight_decay=WEIGHT_DECAY, patience=PATIENCE, clamp=CLAMP, grad_clip=GRAD_CLIP, val_frac=VAL_FRAC,
    holdout_start=HOLDOUT_START, controls=CONTROLS, probe=PROBE,
    resume=RESUME, freeze_encoder_layers=FREEZE_ENCODER_LAYERS, dir_weight=DIR_WEIGHT,
    device=device.type, seed=SEED)

print('\n' + '=' * 60 + '\n  SSL STAGE 2 (multi-horizon seq2seq) VERDICT\n' + '=' * 60)
for k, v in verdict.items():
    if k != 'history':
        print(f'  {k:>22}: {v}')
print('=' * 60)
print(f'\nadapted encoder  -> {OUT_PATH}')
print(f'report           -> {OUT_PATH}.report.json')
print('\nDownstream: BACKBONE_CKPT=<this .pt>  — then benchmark OHLCV-only WR@3R vs stage-1.')
