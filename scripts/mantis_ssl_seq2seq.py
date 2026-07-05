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


# ======================================= CELL 1 — SETUP (clone FFM @ main, install) ==========
import os, subprocess
# Reduce CUDA fragmentation OOM (must be set BEFORE torch inits CUDA -> after a runtime restart).
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
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

# ── PATHS (Drive) — DEFAULT = the REORDER's forecast-last step (STEP 2). Env-overridable. ──
# REORDER (mask->contrastive->forecast): warm from the contrastive-from-mask regime checkpoint,
#   FREEZE early blocks (=3) so forecast learns ON TOP of the regime instead of erasing it, and
#   write a DISTINCT file so the shipped seq2seq stays put. Copy-and-run as-is for step 2.
# Old default lineage (mask->forecast) if ever needed:
#   WARM_CKPT=.../mantis_ssl_ohlcv.pt OUT_PATH=.../mantis_ssl_seq2seq.pt FREEZE_ENCODER_LAYERS=0
DATA_DIR  = os.environ.get('DATA_DIR', '/content/drive/MyDrive/Futures Data')
WARM_CKPT = os.environ.get('WARM_CKPT', '/content/drive/MyDrive/AI_Models/mantis_ssl_regime.pt')
# DEFAULT OUT = the TUNED reorder (Optuna sweep winner, trial 3) — a DISTINCT file so the manual
# freeze=3 anchor (mantis_ssl_seq2seq_reordered.pt, 52.6%) is NEVER overwritten. The two are the
# freeze-2-vs-3 A/B: this tuned freeze=2 winner vs the anchor freeze=3, both vs seq2seq.
OUT_PATH  = os.environ.get('OUT_PATH', '/content/drive/MyDrive/AI_Models/mantis_ssl_seq2seq_reorder_tuned.pt')

# ── CORPUS (same as stage 1) ──
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']      # all 9
TFS     = ['1min', '3min', '5min', '15min']                            # all 4 TFs
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1

# ── MULTI-HORIZON / VARIABLE-CONTEXT candle forecast (sweep-winner) ──
HORIZONS        = (5, 10, 20, 25)             # predict the CANDLE at each (near..far), in bars
CONTEXT_LENGTHS = (64, 100, 150, 200)         # sample a context length per step (short..long)
# DEFAULTS = the REORDER-sweep winner (trial 3): candle_mse / nc=3 / wd=0 / freeze=2 / lr=1.19e-4.
# All env-overridable so any sweep config can be full-trained without editing (e.g. OBJECTIVE=
# candle_direction DIR_WEIGHT=0.29 for trial 5; NEW_CHANNELS=4 LR=1.1e-4 for trial 6).
OBJECTIVE       = os.environ.get('OBJECTIVE', 'candle_mse')
NEW_CHANNELS    = int(os.environ.get('NEW_CHANNELS', '3'))
DIR_WEIGHT      = float(os.environ.get('DIR_WEIGHT', '0.0'))   # >0 only for candle_direction

# ── TRAINING (sweep-winner; BATCH MATCHES THE SWEEP so the tuned LR transfers) ──
BATCH   = 512         # PARITY with the sweep (lr was tuned at 512; 1024 would need a different lr)
EPOCHS  = 60          # full budget (the sweep used a short 12-epoch proxy to RANK; train to convergence)
STEPS   = 200         # steps/epoch
LR      = float(os.environ.get('LR', '0.0001188117389055629'))   # reorder-sweep winner (trial 3, ~1.19e-4)
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', '0.0'))       # trial 3 (env-overridable)
PATIENCE = 8
CLAMP, GRAD_CLIP = 10.0, 1.0                    # stability
CONTROLS = ()                                  # skip shuffle/random retrains (fast iteration; judge
                                               # offline). Re-enable ('shuffle','random') for a fresh
                                               # anti-shortcut check on a NEW pretext/target change.
PROBE = True                                    # probe vs vanilla (diagnostic, not a gate)
SEED = 0

# ── CRASH-SAFE + ANTI-FORGETTING (best saved PROGRESSIVELY to OUT_PATH; Colab-disconnect resilient) ──
RESUME  = False        # True -> resume from the best saved to OUT_PATH (crash recovery)
# DEFAULT = 3 for the REORDER's forecast-last step: freeze the first 3 (of 6) blocks so forecast
# learns on top of the regime instead of erasing it. Set FREEZE_ENCODER_LAYERS=0 for the old
# default lineage (mask->forecast full fine-tune, the original sweep winner).
FREEZE_ENCODER_LAYERS = int(os.environ.get('FREEZE_ENCODER_LAYERS', '2'))   # trial 3 = freeze=2

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
_live = '/content/drive/MyDrive/AI_Models/mantis_ssl_seq2seq.pt'
_safe = '   (live mantis_ssl_seq2seq.pt UNTOUCHED)' if os.path.abspath(OUT_PATH) != os.path.abspath(_live) else '   ⚠️ WRITES THE LIVE seq2seq'
print(f'   OUTPUT -> {OUT_PATH}{_safe}')


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
