# ==============================================================================
# MANTIS SSL PRETRAIN — Masked Modeling ("BERT for futures") on raw OHLCV (Colab GPU)
# ==============================================================================
#
# Self-supervised domain-adaptation of the Mantis-8M backbone on our futures
# corpus (9 tickers x {1,3,5,15}min raw OHLCV). BERT-style MASKED MODELING: mask a
# fraction of bars and reconstruct them from context. To reconstruct a masked bar the
# encoder MUST model regime/volatility (bar size), temporal dynamics (trend
# continuation) and cross-channel coupling — the market-context the downstream buy/sell
# classifier needs. It is NOT gameable by a distributional shortcut, and its REAL/SHUFFLE/
# RANDOM controls are meaningful (only real data has predictable context to reconstruct).
#
# OUTPUT: an adapted ENCODER checkpoint saved to Drive. Downstream classifier
# finetuning starts from it:  build_model(..., backbone_ckpt=<this .pt>)  (and
# the WF/produce driver via BACKBONE_CKPT=<this .pt>).
#
# Validity gates:
#   * TIME-SPLIT val reconstruction early-stop   (generalize forward; 2026 EXCLUDED)
#   * PROBE vs vanilla = the GATE: frozen embedding must predict regime/vol/structure +
#     forward buy/sell move BETTER than the un-adapted backbone (learns_regime_vol_structure)
#   * REAL vs SHUFFLE vs RANDOM = probe-based diagnostic (did temporal order contribute)
#
# GPU-maximized: all bars resident on GPU, vectorized GPU augmentations, large
# batch, CUDA AMP (fp16 + GradScaler).
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
os.system('pip install optuna -q 2>&1 | tail -1')      # probe-gate Optuna tuning (on gate-fail)

try:
    from futures_foundation.finetune import ssl, ssl_data
    print('FFM + SSL modules import OK')
except ImportError as e:
    print(f'Import failed: {e}\nRestarting runtime — re-run this cell after restart...')
    os.kill(os.getpid(), 9)


# ==============================================================================
# CELL 2 — CONFIGURATION  (Drive paths + hyperparameters + pre-flight)
# ==============================================================================
import glob, torch

# ── PATHS (Drive) ──
# DATA_DIR must contain the raw CSVs named  {TICKER}_{TF}.csv  with columns
# datetime,open,high,low,close,volume  (e.g. ES_3min.csv).
DATA_DIR = '/content/drive/MyDrive/Futures Data'
OUT_PATH = '/content/drive/MyDrive/AI_Models/mantis_ssl_ohlcv.pt'   # adapted encoder ckpt

# ── CORPUS ──
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']   # all 9
TFS     = ['1min', '3min', '5min', '15min']
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1                    # last 10% of each stream's pre-2026 bars = val

# ── MASKED MODELING ──
MASK_RATIO   = 0.4     # fraction of bars masked per window, reconstructed from context

# ── MODEL ──
SEQ          = 64      # model input window (bars); interpolated to Mantis's native 512 internally
MAX_JITTER   = 16      # forward horizon reserved for the buy/sell probe (in-stream)
NEW_CHANNELS = 8       # channel-combiner output (OHLCV=5 -> NEW_CHANNELS)

# ── TRAINING (GPU-max) ──
BATCH        = 1024    # drop if OOM (512 / 768).
EPOCHS       = 60
STEPS        = 200     # steps per epoch
LR           = 1e-4
PATIENCE     = 8
CONTROLS     = ['shuffle', 'random']   # probe-based diagnostics (did temporal order help)
COMPILE      = False   # torch.compile(encoder) — try True on A100/L4 for extra speed
SEED         = 0

# ── GENERALIZATION (probe gate -> Optuna, like WF/produce) ──
N_TRIALS     = 10      # Optuna trials (MAXIMIZE probe delta) if the default doesn't pass
MAX_ITERS    = 2       # default run, then (if needed) one Optuna-tuned re-run
PROBE        = True    # GATE: probe regime/vol/structure + forward buy/sell move vs vanilla

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
if device.type != 'cuda':
    print('⚠️  No CUDA — SSL is designed for a Colab GPU runtime (Runtime > Change runtime type > GPU).')

# ── PRE-FLIGHT: fail in seconds if the data path is wrong (before any GPU) ──
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f'DATA_DIR does not exist:\n  {DATA_DIR}')
found = []
for tk in TICKERS:
    for tf in TFS:
        if os.path.exists(os.path.join(DATA_DIR, f'{tk}_{tf}.csv')):
            found.append(f'{tk}_{tf}')
if not found:
    raise FileNotFoundError(
        f'No {{TICKER}}_{{TF}}.csv files found under {DATA_DIR}.\n'
        f'Expected e.g. {DATA_DIR}/ES_3min.csv with columns '
        f'datetime,open,high,low,close,volume.')
print(f'✅ PRE-FLIGHT: found {len(found)}/{len(TICKERS)*len(TFS)} CSVs under {DATA_DIR}')
print(f'   masked modeling | corpus {TICKERS} x {TFS} | SEQ={SEQ} MASK={MASK_RATIO} '
      f'BATCH={BATCH} EPOCHS={EPOCHS} controls={CONTROLS}')
print(f'   OUTPUT -> {OUT_PATH}')


# ==============================================================================
# CELL 3 — RUN SSL PRETRAIN  (probe-gated + Optuna; saves encoder ckpt + .report.json)
# ==============================================================================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH,
    tickers=TICKERS, tfs=TFS, mask_ratio=MASK_RATIO,
    seq=SEQ, max_jitter=MAX_JITTER, new_channels=NEW_CHANNELS,
    batch=BATCH, epochs=EPOCHS, steps_per_epoch=STEPS, lr=LR,
    patience=PATIENCE, val_frac=VAL_FRAC, holdout_start=HOLDOUT_START,
    controls=tuple(CONTROLS), probe=PROBE, n_trials=N_TRIALS, max_iters=MAX_ITERS,
    device=device.type, compile_model=COMPILE, seed=SEED,
)

print('\n' + '=' * 60)
print('  SSL VERDICT')
print('=' * 60)
for k, v in verdict.items():
    print(f'  {k:>22}: {v}')
print('=' * 60)
print(f'\nadapted encoder  -> {OUT_PATH}')
print(f'report           -> {OUT_PATH}.report.json')
print('\nDownstream use:  BACKBONE_CKPT=<this .pt>  in the WF/produce driver, or')
print('                 build_model(..., backbone_ckpt=<this .pt>)')
