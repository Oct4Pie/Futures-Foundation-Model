# ==============================================================================
# MANTIS SSL STAGE 3 — TEMPORAL-NEIGHBORHOOD CONTRASTIVE (regime geometry) (Colab GPU)
# ==============================================================================
#
# Fine-tunes the PROMOTED stage-2 seq2seq encoder with a self-supervised temporal contrastive
# objective (contrastive-ffm-requirements spec). NO labels anywhere:
#
#   * POSITIVES  = windows at short/medium/long temporal offsets (pos_deltas) + augmented views
#                  (noise / channel scaling / time masking / crop-resize)
#   * NEGATIVES  = in-batch windows FAR apart in time (pairs closer than FAR_MIN bars are
#                  excluded from the loss denominator entirely)
#   * WEIGHTING  = data-driven per-window volatility σ_t (mean|Δclose|/mean|close|):
#                  high-vol/chaotic anchors down-weighted — a weight, never a label
#
# GOAL: a smooth "market state geometry" — nearby-in-time / structurally-similar windows
# cluster; different structures separate. THIS STAGE'S GATE = the spec's structural metrics
# (A temporal consistency, B emergent clusters, C multi-scale ordering, D noise robustness,
# E temporal stability), printed per-epoch and as a final PASS/FAIL verdict.
#
# SHIP GATE UNCHANGED: stage-2 seq2seq stays the shipped base. This checkpoint is a CANDIDATE —
# it must beat 54.7%@1/d on the one-shot 2026 WR@3R benchmark before any promotion:
#   S3_CKPT=<this .pt>  python3 colabs/mantis_2026_benchmark.py
#
# SAFETY: writes a DISTINCT checkpoint (mantis_ssl_regime.pt). NEVER overwrites
# mantis_ssl_seq2seq.pt (incumbent) or mantis_ssl_ohlcv.pt (stage-1) — preflight hard-fails.
# ==============================================================================


# ======================================= CELL 1 — SETUP (clone FFM @ main, install) ===========
import os, subprocess
os.chdir('/content')

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

print('Cloning FFM repo (main)...')
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
DATA_DIR  = '/content/drive/MyDrive/Futures Data'
WARM_CKPT = '/content/drive/MyDrive/AI_Models/mantis_ssl_seq2seq.pt'    # stage-2 (warm-start)
OUT_PATH  = '/content/drive/MyDrive/AI_Models/mantis_ssl_regime.pt'     # stage-3 CANDIDATE (NEW file)

# ── CORPUS (same universe as stage 1/2 — the ruler must not drift) ──
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']
TFS     = ['1min', '3min', '5min', '15min']
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1

# ── TEMPORAL CONTRASTIVE (the spec's knobs) ──
SEQ         = 64                      # window length (parity with stage-2 / downstream MV_SEQ)
POS_DELTAS  = (2, 16, 64)             # short / medium / long positive offsets (bars)
FAR_MIN     = 512                     # pairs closer than this are excluded as negatives
TEMPERATURE = 0.1
AUG_NOISE, AUG_SCALE, AUG_TMASK, CROP_MAX = 0.10, 0.20, 0.15, 0.2
VOL_WEIGHT  = 1.0                     # σ_t down-weighting strength (0 = off)
NEW_CHANNELS, PROJ_DIM = 8, 128
FREEZE_ENCODER_LAYERS  = 0            # bump to 4 if the geometry forms but drifts off seq2seq

# ── TRAINING ──
BATCH   = 256                         # 5 windows/anchor stacked -> 1280 encoder rows/step
EPOCHS  = 60
STEPS   = 200
LR      = 1e-4                        # gentle: this is a REFINE of a proven base
WEIGHT_DECAY, PATIENCE = 0.05, 8
CLAMP, GRAD_CLIP = 10.0, 1.0
METRICS_N = 768                       # val sample for the A-E regime metrics
CONTROLS  = ()                        # honest-ruler controls run downstream (WF), not here
PROBE = True
SEED  = 0
RESUME = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
if device.type != 'cuda':
    print('⚠️  No CUDA — SSL is designed for a Colab GPU runtime.')

# ── PRE-FLIGHT (protected checkpoints can never be clobbered) ──
_PROTECTED = {'mantis_ssl_seq2seq.pt', 'mantis_ssl_ohlcv.pt'}
if os.path.basename(OUT_PATH) in _PROTECTED:
    raise SystemExit('❌ OUT_PATH would overwrite a PROTECTED checkpoint — pick a NEW file.')
if os.path.abspath(OUT_PATH) == os.path.abspath(WARM_CKPT):
    raise SystemExit('❌ OUT_PATH == WARM_CKPT — would overwrite the warm-start.')
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f'DATA_DIR does not exist:\n  {DATA_DIR}')
if not os.path.exists(WARM_CKPT):
    raise FileNotFoundError(f'WARM_CKPT (stage-2 seq2seq) not found:\n  {WARM_CKPT}')
found = [f'{tk}_{tf}' for tk in TICKERS for tf in TFS
         if os.path.exists(os.path.join(DATA_DIR, f'{tk}_{tf}.csv'))]
if not found:
    raise FileNotFoundError(f'No {{TICKER}}_{{TF}}.csv files under {DATA_DIR}.')
print(f'✅ PRE-FLIGHT: {len(found)}/{len(TICKERS)*len(TFS)} CSVs | warm-start <- {WARM_CKPT}')
print(f'   deltas={POS_DELTAS} far_min={FAR_MIN} vol_w={VOL_WEIGHT} lr={LR:.1e} '
      f'batch={BATCH} frz={FREEZE_ENCODER_LAYERS}')
print(f'   OUTPUT -> {OUT_PATH}   (mantis_ssl_seq2seq.pt UNTOUCHED)')


# ======================================= CELL 3 — TRAIN ========================================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH, tickers=TICKERS, tfs=TFS,
    pretext='contrastive', backbone_ckpt=WARM_CKPT,
    seq=SEQ, pos_deltas=POS_DELTAS, far_min=FAR_MIN, temperature=TEMPERATURE,
    aug_noise=AUG_NOISE, aug_scale=AUG_SCALE, aug_tmask=AUG_TMASK, crop_max=CROP_MAX,
    vol_weight=VOL_WEIGHT, new_channels=NEW_CHANNELS, proj_dim=PROJ_DIM, metrics_n=METRICS_N,
    batch=BATCH, epochs=EPOCHS, steps_per_epoch=STEPS, lr=LR, weight_decay=WEIGHT_DECAY,
    patience=PATIENCE, clamp=CLAMP, grad_clip=GRAD_CLIP, val_frac=VAL_FRAC,
    holdout_start=HOLDOUT_START, controls=CONTROLS, probe=PROBE,
    resume=RESUME, freeze_encoder_layers=FREEZE_ENCODER_LAYERS,
    device=device.type, seed=SEED)

print('\n' + '=' * 60 + '\n  SSL STAGE 3 (temporal contrastive / regime geometry) VERDICT\n' + '=' * 60)
for k, v in verdict.items():
    if k != 'history':
        print(f'  {k:>22}: {v}')

# ── the spec's A-E REGIME-GEOMETRY GATE (this stage's success definition) ──
from futures_foundation.finetune._ssl_torch import regime_gate
hist = verdict.get('history') or []
extras = [h for h in hist if isinstance(h, dict) and 'smooth' in h]
if extras:
    ok, checks = regime_gate(extras[-1])
    print('-' * 60)
    print(f'  REGIME GEOMETRY GATE: {"PASS — structured market-state geometry" if ok else "FAIL — geometry did not form"}')
    for name, passed in checks.items():
        print(f'    {name:26s} {"✓" if passed else "✗"}')
else:
    print('  (no A-E metrics found in history — inspect the per-epoch log above)')
print('=' * 60)
print(f'\nadapted encoder  -> {OUT_PATH}   (mantis_ssl_seq2seq.pt UNTOUCHED)')
print('\nNext: the SHIP gate is unchanged — one-shot 2026 WR@3R vs the 54.7% stage-2 base:')
print(f'  S3_CKPT={OUT_PATH}  python3 colabs/mantis_2026_benchmark.py   (promote iff it wins)')
