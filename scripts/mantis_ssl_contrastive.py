# ==============================================================================
# MANTIS SSL STAGE 3 (EXPERIMENT) — TREND CONTRASTIVE (multi-positive InfoNCE) — Colab GPU
# ==============================================================================
#
# Sharpen the encoder's TREND-vs-CHOP separation with contrastive learning. Warm-starts from
# the stage-2 candle seq2seq encoder (mantis_ssl_seq2seq.pt = ctx200) and continues training
# with a contrastive objective ADAPTED for trend detection:
#
#   * REUSES Mantis's InfoNCE machinery — normalized-similarity + temperature(0.1) + a
#     projection head + RandomCropResize augmentation (crop 0-20%, resize back).
#   * ADAPTS the objective: single-positive instance-discrimination -> MULTI-POSITIVE
#     (SupCon mechanics) grouped by a SELF-SUPERVISED, CAUSAL TREND KEY (direction x magnitude
#     of the trailing slope, from PAST bars only). Same-trend windows become POSITIVES (pulled
#     together) instead of negatives -> the embedding GROUPS trends and separates chop.
#     Trend vs chop now; trend CONTINUATION is a future stage.
#
# Self-supervised (no labels -> no leak). 2026 EXCLUDED from SSL. Apples-to-apples shuffle/random
# controls (trend key always from real context; only the model INPUT is corrupted).
#
# EXPERIMENT: FALLBACK = stage-2 (ctx200). Judge OFFLINE on the SAME baseline as ctx200 —
# trend-AUC + decile spread (scratchpad/trend_learn_analysis.py) and OHLCV-only WR@3R at each
# per-ticker/day operating point (scratchpad/ridgeR_2026_wr_lean.py). SHIP only if it beats
# ctx200 (AUC-3R 0.554 / spread +14.8 / WR@3R) WITHOUT cratering signal COUNT (keep good entries).
#
# OUTPUT: an adapted ENCODER checkpoint, consumed downstream exactly like stage-1/2:
#   BACKBONE_CKPT=<this .pt>  (embed / benchmark), or build_model(..., backbone_ckpt=<this>).
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
WARM_CKPT = '/content/drive/MyDrive/AI_Models/mantis_ssl_seq2seq.pt'      # stage-2 encoder (warm-start)
OUT_PATH  = '/content/drive/MyDrive/AI_Models/mantis_ssl_contrastive.pt'  # stage-3 adapted encoder

# ── CORPUS (same as stage 1/2) ──
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']      # all 9
TFS     = ['1min', '3min', '5min', '15min']                            # all 4 TFs
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1

# ── TREND CONTRASTIVE (variable-context, self-supervised trend key) ──
CONTEXT_LENGTHS = (64, 100, 150, 200)          # same variable context as stage-2
TEMPERATURE     = 0.1                           # Mantis InfoNCE temperature
CROP_MAX        = 0.2                            # RandomCropResize: crop up to 20%
PROJ_DIM        = 128                            # projection-head output (discarded after)
NEW_CHANNELS    = 8                             # channel-combiner output (OHLCV=5 -> min(8,5)=5)

# ── TRAINING (GPU-max; Mantis contrastive recipe) ──
BATCH   = 512         # Mantis contrastive batch; drop if OOM
EPOCHS  = 60
STEPS   = 200         # steps/epoch
LR      = 2e-3        # Mantis contrastive base LR (AdamW, wd=0.05)
PATIENCE = 8
CLAMP, GRAD_CLIP = 10.0, 1.0
CONTROLS = ('shuffle', 'random')               # apples-to-apples diagnostics
PROBE = True                                   # probe vs vanilla (diagnostic, not the gate)
SEED = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
if device.type != 'cuda':
    print('⚠️  No CUDA — SSL is designed for a Colab GPU runtime (Runtime > Change runtime type > GPU).')

# ── PRE-FLIGHT ──
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f'DATA_DIR does not exist:\n  {DATA_DIR}')
if not os.path.exists(WARM_CKPT):
    raise FileNotFoundError(f'WARM_CKPT (stage-2 encoder) not found:\n  {WARM_CKPT}\n'
                            f'Run scripts/mantis_ssl_seq2seq.py first.')
found = [f'{tk}_{tf}' for tk in TICKERS for tf in TFS
         if os.path.exists(os.path.join(DATA_DIR, f'{tk}_{tf}.csv'))]
if not found:
    raise FileNotFoundError(f'No {{TICKER}}_{{TF}}.csv files under {DATA_DIR}.')
print(f'✅ PRE-FLIGHT: {len(found)}/{len(TICKERS)*len(TFS)} CSVs | warm-start <- {WARM_CKPT}')
print(f'   context_lengths={CONTEXT_LENGTHS} temp={TEMPERATURE} crop_max={CROP_MAX} BATCH={BATCH}')
print(f'   OUTPUT -> {OUT_PATH}')


# ======================================= CELL 3 — TRAIN (single run, no Optuna) ================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH, tickers=TICKERS, tfs=TFS,
    pretext='contrastive', backbone_ckpt=WARM_CKPT,           # <- stage-3 contrastive, warm-start stage-2
    context_lengths=CONTEXT_LENGTHS, temperature=TEMPERATURE, crop_max=CROP_MAX,
    proj_dim=PROJ_DIM, new_channels=NEW_CHANNELS, batch=BATCH, epochs=EPOCHS,
    steps_per_epoch=STEPS, lr=LR, patience=PATIENCE, clamp=CLAMP, grad_clip=GRAD_CLIP,
    val_frac=VAL_FRAC, holdout_start=HOLDOUT_START, controls=CONTROLS, probe=PROBE,
    device=device.type, seed=SEED)

print('\n' + '=' * 60 + '\n  SSL STAGE 3 (trend contrastive) VERDICT\n' + '=' * 60)
for k, v in verdict.items():
    if k != 'history':
        print(f'  {k:>22}: {v}')
print('=' * 60)
print(f'\nadapted encoder  -> {OUT_PATH}')
print(f'report           -> {OUT_PATH}.report.json')
print('\nDownstream: BACKBONE_CKPT=<this .pt>  — then measure OHLCV-only trend-AUC + WR@3R vs ctx200.')
print('EXPERIMENT: ship only if it beats ctx200 (AUC-3R 0.554 / spread +14.8 / WR@3R) AND keeps signal COUNT.')
