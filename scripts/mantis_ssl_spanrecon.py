# ==============================================================================
# MANTIS SSL — SpanBERT-STYLE SPAN-MASKED RECONSTRUCTION (trend-development refine) (Colab GPU)
# ==============================================================================
#
# Refines the promoted base by RECONSTRUCTING contiguous MASKED SPANS of bars. Unlike single-bar
# masking (interpolate a hole from neighbors — a LOCAL skill), masking a whole run of bars forces
# the encoder to infer the MISSING MOVE from surrounding context: momentum, where the trend was
# heading, how far it should have carried. That's "learn trend patterns over time" as a pure
# RECONSTRUCTION task — the objective family that has WON here twice (reconstruction is load-
# bearing). Crucially, unlike ELECTRA (which failed by giving the encoder no recon gradient),
# THIS objective IS reconstruction, so there is no anchor problem — the trend signal is preserved
# and the span structure is added on top.
#
# ── VERIFY IT'S LEARNING (per-epoch log + probe) ──
#   val_loss   span-reconstruction MSE — should fall below the base's level (the base never did
#              span recon, so any drop = genuinely new structure learned, not re-fitting).
#   std        embedding collapse guard (> 0.01). Watch it does NOT balloon like electra's did
#              (1.0->2.35 = drift): reconstruction should keep it near the base's ~1.0.
#   PROBE      frozen-embedding vs vanilla — mean_core_delta positive AND ideally the trend_eff /
#              fwd_absmove rows improve vs the base's probe (that's the trend-detection bet).
#
# SHIP GATE UNCHANGED (never the pretext loss): 2025 dry-run vs the base at matched operating
# points, then the one-shot 2026 only if it wins. CANDIDATE row: S4_CKPT=<this .pt>.
#
# SAFETY: writes a DISTINCT checkpoint. NEVER overwrites the promoted bases — preflight hard-fails.
# ==============================================================================


# ======================================= CELL 1 — SETUP (clone FFM @ main, install) ===========
import os, subprocess
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
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

# ── PATHS (Drive) — warm from the PROMOTED base (or the RoBERTa extension winner if it's promoted).
DATA_DIR  = os.environ.get('DATA_DIR', '/content/drive/MyDrive/Futures Data')
WARM_CKPT = os.environ.get('WARM_CKPT', '/content/drive/MyDrive/AI_Models/mantis_ssl_ctr_seq2seq.pt')
OUT_PATH  = os.environ.get('OUT_PATH', '/content/drive/MyDrive/AI_Models/mantis_ssl_spanrecon.pt')

# ── CORPUS (same universe as every stage — the ruler must not drift) ──
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']
TFS     = ['1min', '3min', '5min', '15min']
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1

# ── SPAN-RECON knobs ──
SEQ         = 64                      # window parity with the base + downstream MV_SEQ
MASK_RATIO  = float(os.environ.get('MASK_RATIO', '0.25'))  # fraction of bars covered by masked spans
SPAN_MEAN   = float(os.environ.get('SPAN_MEAN', '5'))      # geometric mean span length (bars)
SPAN_MAX    = int(os.environ.get('SPAN_MAX', '12'))        # clip per span
NEW_CHANNELS = 3                      # sweep-winner setting of the base
FREEZE_ENCODER_LAYERS = int(os.environ.get('FREEZE_ENCODER_LAYERS', '2'))  # anti-forgetting (base=frz2)

# ── TRAINING ──
BATCH   = int(os.environ.get('BATCH', '512'))
EPOCHS  = int(os.environ.get('EPOCHS', '120'))             # generous (RoBERTa lesson: don't wall it)
STEPS   = 200
LR      = float(os.environ.get('LR', '1e-4'))              # gentle: a REFINE of a proven base
WEIGHT_DECAY, PATIENCE = 0.05, 8
CONTROLS = ()                          # honest-ruler controls run downstream (WF), not here
PROBE = True
SEED  = 0
RESUME = os.environ.get('RESUME', '0') == '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
if device.type != 'cuda':
    print('⚠️  No CUDA — SSL is designed for a Colab GPU runtime.')

# ── PRE-FLIGHT (protected checkpoints can never be clobbered) ──
_PROTECTED = {'mantis_ssl_seq2seq.pt', 'mantis_ssl_ohlcv.pt', 'mantis_ssl_ctr_seq2seq.pt'}
if os.path.basename(OUT_PATH) in _PROTECTED:
    raise SystemExit('❌ OUT_PATH would overwrite a PROTECTED checkpoint — pick a NEW file.')
if os.path.abspath(OUT_PATH) == os.path.abspath(WARM_CKPT):
    raise SystemExit('❌ OUT_PATH == WARM_CKPT — would overwrite the warm-start.')
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f'DATA_DIR does not exist:\n  {DATA_DIR}')
if not os.path.exists(WARM_CKPT):
    raise FileNotFoundError(f'WARM_CKPT (promoted base) not found:\n  {WARM_CKPT}')
found = [f'{tk}_{tf}' for tk in TICKERS for tf in TFS
         if os.path.exists(os.path.join(DATA_DIR, f'{tk}_{tf}.csv'))]
if not found:
    raise FileNotFoundError(f'No {{TICKER}}_{{TF}}.csv files under {DATA_DIR}.')
print(f'✅ PRE-FLIGHT: {len(found)}/{len(TICKERS)*len(TFS)} CSVs | warm-start <- {WARM_CKPT}')
print(f'   SPAN-RECON: mask={MASK_RATIO} span_mean={SPAN_MEAN} span_max={SPAN_MAX} '
      f'lr={LR:.1e} batch={BATCH} epochs={EPOCHS} frz={FREEZE_ENCODER_LAYERS}')
print(f'   OUTPUT -> {OUT_PATH}   (promoted bases UNTOUCHED)')


# ======================================= CELL 3 — TRAIN ========================================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH, tickers=TICKERS, tfs=TFS,
    pretext='mask', backbone_ckpt=WARM_CKPT,        # pretext='mask' + span_mean>0 = SpanBERT recon
    seq=SEQ, mask_ratio=MASK_RATIO, span_mean=SPAN_MEAN, span_max=SPAN_MAX,
    new_channels=NEW_CHANNELS,
    batch=BATCH, epochs=EPOCHS, steps_per_epoch=STEPS, lr=LR, weight_decay=WEIGHT_DECAY,
    patience=PATIENCE, val_frac=VAL_FRAC, holdout_start=HOLDOUT_START,
    controls=CONTROLS, probe=PROBE, resume=RESUME,
    freeze_encoder_layers=FREEZE_ENCODER_LAYERS, device=device.type, seed=SEED)

print('\n' + '=' * 60 + '\n  SpanBERT span-masked reconstruction VERDICT\n' + '=' * 60)
for k, v in verdict.items():
    if k not in ('history', 'epochs'):
        print(f'  {k:>22}: {v}')
print('-' * 60)
print('  Pretext-level pass = mean_core_delta > 0 + no collapse (std ~1.0, NOT ballooning).')
print('  THE verdict is downstream: 2025 dry-run vs the base at matched operating points.')
print('=' * 60)
print(f'\nadapted encoder  -> {OUT_PATH}   (promoted bases UNTOUCHED)')
print('\nNext — the SHIP gate (unchanged):')
print(f'  S4_CKPT={OUT_PATH}  python3 colabs/mantis_2026_benchmark.py   (2025 dry-run first; '
      'one-shot 2026 only if it beats the base)')
