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
DATA_DIR  = '/content/drive/MyDrive/Futures Data'
WARM_CKPT = '/content/drive/MyDrive/AI_Models/mantis_ssl_ohlcv.pt'      # stage-1 encoder (warm-start)
OUT_PATH  = '/content/drive/MyDrive/AI_Models/mantis_ssl_seq2seq.pt'    # stage-2 adapted encoder

# ── CORPUS (same as stage 1) ──
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']      # all 9
TFS     = ['1min', '3min', '5min', '15min']                            # all 4 TFs
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1

# ── MULTI-HORIZON / VARIABLE-CONTEXT candle forecast ──
HORIZONS        = (5, 10, 20, 25)             # predict the CANDLE at each (near..far), in bars
CONTEXT_LENGTHS = (64, 100, 150, 200)         # sample a context length per step (short..long)
NEW_CHANNELS    = 8                            # channel-combiner output (OHLCV=5 -> NEW_CHANNELS)
DIR_WEIGHT      = 0.5                          # >0 = ADD a DIRECTION-head squeeze: BCE on sign(fwd close
                                              # move) alongside the candle MSE -> trains the encoder to
                                              # be DIRECTION-aware (WR is sign-dominated; MSE alone
                                              # mean-regresses the sign away). Watch val 'dir'>0.5.
                                              # 0 = original pure-candle stage-2 (backward-compat).

# ── TRAINING (GPU-max) ──
BATCH   = 1024        # drop to 512/768 if OOM
EPOCHS  = 60
STEPS   = 200         # steps/epoch
LR      = 1e-4
PATIENCE = 8
CLAMP, GRAD_CLIP = 10.0, 1.0                    # stability
CONTROLS = ()                                  # skip shuffle/random retrains (fast iteration; judge
                                               # offline). Re-enable ('shuffle','random') for a fresh
                                               # anti-shortcut check on a NEW pretext/target change.
PROBE = True                                    # probe vs vanilla (diagnostic, not a gate)
SEED = 0

# ── CRASH-SAFE + ANTI-FORGETTING (best saved PROGRESSIVELY to OUT_PATH; Colab-disconnect resilient) ──
RESUME  = False        # True -> resume from the best saved to OUT_PATH (crash recovery)
FREEZE_ENCODER_LAYERS = 0   # 0 = full fine-tune (stage-2 default). Raise (e.g. 3-4) to freeze the
                            # tokenizer + first N of 6 Mantis layers if stage-1 drift is a concern.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
if device.type != 'cuda':
    print('⚠️  No CUDA — SSL is designed for a Colab GPU runtime (Runtime > Change runtime type > GPU).')

# ── PRE-FLIGHT ──
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f'DATA_DIR does not exist:\n  {DATA_DIR}')
if not os.path.exists(WARM_CKPT):
    raise FileNotFoundError(f'WARM_CKPT (stage-1 encoder) not found:\n  {WARM_CKPT}\n'
                            f'Run scripts/mantis_ssl_pretrain.py first.')
found = [f'{tk}_{tf}' for tk in TICKERS for tf in TFS
         if os.path.exists(os.path.join(DATA_DIR, f'{tk}_{tf}.csv'))]
if not found:
    raise FileNotFoundError(f'No {{TICKER}}_{{TF}}.csv files under {DATA_DIR}.')
print(f'✅ PRE-FLIGHT: {len(found)}/{len(TICKERS)*len(TFS)} CSVs | warm-start <- {WARM_CKPT}')
print(f'   horizons={HORIZONS} context_lengths={CONTEXT_LENGTHS} BATCH={BATCH} EPOCHS={EPOCHS}')
print(f'   OUTPUT -> {OUT_PATH}')


# ======================================= CELL 3 — TRAIN (single run, no Optuna) ================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH, tickers=TICKERS, tfs=TFS,
    pretext='forecast', backbone_ckpt=WARM_CKPT,               # <- stage-2 forecast, warm-start stage-1
    horizons=HORIZONS, context_lengths=CONTEXT_LENGTHS,
    new_channels=NEW_CHANNELS, batch=BATCH, epochs=EPOCHS, steps_per_epoch=STEPS, lr=LR,
    patience=PATIENCE, clamp=CLAMP, grad_clip=GRAD_CLIP, val_frac=VAL_FRAC,
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
