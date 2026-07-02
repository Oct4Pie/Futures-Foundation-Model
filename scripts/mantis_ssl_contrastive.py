# ==============================================================================
# MANTIS SSL STAGE 3 v3 — FORWARD BARRIER-EXCURSION CONTRASTIVE (multi-positive InfoNCE) — Colab GPU
# ==============================================================================
#
# Sharpen the encoder's TREND-vs-CHOP separation. Warm-starts from the BEST stage-2 forecast
# encoder (the Optuna-sweep winner) and refines with Mantis's own contrastive machinery —
# normalized-similarity InfoNCE + temperature(0.1) + projection head + RandomCropResize —
# adapted MULTI-POSITIVE (SupCon mechanics). Only the POSITIVE-PAIR DEFINITION changes per version:
#
#   * v1 (WASHED, 42.3 vs 42.8 WR@3R): key = TRAILING slope of the input — computable from the
#     window (shortcut; taught nothing forward) and past-tense (pulled together windows whose
#     futures differ — coils about to break out vs dead whipsaw — erasing the tell).
#   * v2 (WASHED, sweep 2026-07-02: best 6.667 < 6.720 baseline, key_gap ~0.003 flat): key =
#     future direction x path EFFICIENCY (|net|/sum|steps|) — a statistic the stage-2 candle
#     forecast ALREADY encodes (it regresses the future path), so the loss had nothing new to
#     carve: frozen configs reproduced stage-2, unfrozen ones only forgot it.
#   * v3: key = the FUTURE window's BARRIER EXCURSION — how far the close path RAN in one
#     direction (in calibrated context-sigma units) before RETRACING one unit against that run,
#     x direction. An ORDERING statistic the candle-MSE forecast does NOT regress (a +2-then-crash
#     fakeout and a clean crash have the same net but different keys) and the exact shape of the
#     ship metric (WR@3R = reach 3R before -1R). Low run = future CHOP (price ran nowhere cleanly).
#     Same anti-shortcut property (key is target-side, from the NEXT `HORIZON` bars) and same SSL
#     discipline: raw closes only, context-standardized (the pipeline's universal z-score) — NO
#     ATR / NO R / NO indicators / NO labels. Bucket edges FIXED (calibrated once from train
#     windows: retrace unit = |net| tercile, run edges = run terciles), not per-batch.
#
# Self-supervised (future candles are data, not labels -> no leak). 2026 EXCLUDED from SSL.
# Controls: key always from the REAL future; only the model INPUT is corrupted.
#
# EXPERIMENT: FALLBACK = stage-2 (sweep winner). Judge OFFLINE on the same ruler —
# trend-AUC + decile spread (watch the BOTTOM decile = chop-filter quality) + OHLCV-only WR@3R
# (scratchpad/trend_learn_analysis.py + wr3r_logistic_bench.py). SHIP only if it beats the
# stage-2 winner WITHOUT cratering signal COUNT. Watch 'key_gap' in the log = the trend/chop
# separation forming in embedding space (val is FIXED-batch, so it's comparable across epochs).
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

# ── FORWARD TREND-vs-CHOP CONTRASTIVE (variable context; key = future dir x path efficiency) ──
CONTEXT_LENGTHS = (64, 100, 150, 200)          # same variable context as stage-2
HORIZON         = 25                            # FUTURE bars the key reads (matches stage-2's far
                                               # horizon = the ~2R/3R trade timescale). Key = dir x
                                               # efficiency of these bars; NOT computable from input.
POS_CAP         = 64                            # max key-positives per anchor (huge buckets -> the
                                               # loss degenerates to bucket-centroid averaging)
TEMPERATURE     = 0.1                           # Mantis InfoNCE temperature
CROP_MAX        = 0.2                            # RandomCropResize: crop up to 20%
PROJ_DIM        = 128                            # projection-head output (discarded after)
NEW_CHANNELS    = 8                             # channel-combiner output (OHLCV=5 -> min(8,5)=5)

# ── TRAINING (GPU-max; Mantis contrastive recipe) ──
BATCH   = 512         # Mantis contrastive batch; drop if OOM
EPOCHS  = 60
STEPS   = 200         # steps/epoch
LR      = 2e-4        # gentle warm-start REFINE (2e-3 = Mantis from-scratch, too hot here: it
                     # drove the emb_std runaway 6->9 = drift/forgetting of ctx200). AdamW wd=0.05.
PATIENCE = 8
CLAMP, GRAD_CLIP = 10.0, 1.0
CONTROLS = ()                                  # skip shuffle/random retrains (fast iteration; judge
                                               # offline). Re-enable ('shuffle','random') for a fresh
                                               # anti-shortcut check on a NEW pretext/target change.
PROBE = True                                   # probe vs vanilla (diagnostic, not the gate)
SEED = 0

# ── CRASH-SAFE + ANTI-FORGETTING (Colab disconnects; ctx200 drift) ──
RESUME  = False        # True -> resume from the best saved to OUT_PATH (crash recovery). Best is
                       # saved PROGRESSIVELY every val improvement regardless, so nothing is lost.
FREEZE_ENCODER_LAYERS = 4   # anti-forgetting: freeze tokenizer + first N of 6 Mantis layers; train
                            # the rest + adapter + projection. Anchors ctx200 vs the emb_std drift
                            # we saw. 0 = full fine-tune; raise toward 5 if it still drifts.

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
    pretext='contrastive', backbone_ckpt=WARM_CKPT,           # <- stage-3 v2, warm-start stage-2 winner
    context_lengths=CONTEXT_LENGTHS, contrast_horizon=HORIZON, pos_cap=POS_CAP,
    temperature=TEMPERATURE, crop_max=CROP_MAX,
    proj_dim=PROJ_DIM, new_channels=NEW_CHANNELS, batch=BATCH, epochs=EPOCHS,
    steps_per_epoch=STEPS, lr=LR, patience=PATIENCE, clamp=CLAMP, grad_clip=GRAD_CLIP,
    val_frac=VAL_FRAC, holdout_start=HOLDOUT_START, controls=CONTROLS, probe=PROBE,
    resume=RESUME, freeze_encoder_layers=FREEZE_ENCODER_LAYERS,
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
