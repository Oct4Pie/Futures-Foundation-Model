# ==============================================================================
# MANTIS SSL STAGE 4 — ELECTRA-STYLE REPLACED-CANDLE DETECTION (RTD) (Colab GPU)
# ==============================================================================
#
# Refines the PROMOTED base (ctr_seq2seq) with a DISCRIMINATIVE objective. A small, deliberately
# weak conv GENERATOR fills masked bars with plausible fakes; the Mantis encoder (the foundation
# we keep) labels EVERY bar real-vs-replaced. Every bar carries a training signal (vs ~15% for
# generative masking) — the ELECTRA sample-efficiency insight, our highest-leverage axis on
# limited financial data. Non-adversarial: generator trains on recon only (no GAN loop).
#
# ── HOW WE VERIFY IT'S ACTUALLY LEARNING (not cheating) — read the per-epoch log + final block ──
#   rtd_bal_acc  BALANCED accuracy (fake-recall + real-acc)/2 — the honest signal. A lazy
#                all-real predictor scores 0.5 here (it would score 85% RAW accuracy!).
#                LEARNING = rises from ~0.5 and lands in the ~0.60-0.95 band.
#                ~0.50 flat = not learning (generator too strong / encoder blind).
#                >0.97     = fakes trivially detectable => a SHORTCUT TELL or too-weak generator
#                            (lower gen quality bar: raise GEN_WIDTH, or raise MASK_RATIO).
#   fake_recall / real_acc  the two error modes split out (both should be well off 0/1 extremes).
#   gen_mse      generator plausibility — should FALL early then flatten; if it keeps falling fast
#                while rtd_bal_acc drops toward 0.5, the generator is getting too good.
#   std          embedding collapse guard (must stay > 0.01).
#   PROBE        (report-only gate) frozen-embedding probe: mean_core_delta must stay positive —
#                the encoder still encodes regime/vol/structure better than vanilla, i.e. the
#                RTD refine did not destroy the reconstruction lineage it warmed from.
#   Anti-cheat baked into the trainer: OHLC-clamped fakes (no impossible-candle tell), BCE
#   pos_weight (loss can't be gamed by predicting all-real), detached fakes (no adversarial loop).
#
# SHIP GATE UNCHANGED (never the pretext metrics): downstream one-shot 2026 WR@3R vs the current
# base + generality probes (forecast skill / regime separation). This checkpoint is a CANDIDATE:
#   S3_CKPT=<this .pt>  python3 colabs/mantis_2026_benchmark.py   (promote iff it wins)
#
# SAFETY: writes a DISTINCT checkpoint (mantis_ssl_electra.pt). NEVER overwrites the promoted
# bases (ctr_seq2seq / seq2seq / ohlcv) — preflight hard-fails.
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

# ── PATHS (Drive) — warm from the PROMOTED base (the validated ctr_seq2seq lineage) ──
DATA_DIR  = os.environ.get('DATA_DIR', '/content/drive/MyDrive/Futures Data')
WARM_CKPT = os.environ.get('WARM_CKPT', '/content/drive/MyDrive/AI_Models/mantis_ssl_ctr_seq2seq.pt')
OUT_PATH  = os.environ.get('OUT_PATH', '/content/drive/MyDrive/AI_Models/mantis_ssl_electra.pt')

# ── CORPUS (same universe as every stage — the ruler must not drift) ──
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']
TFS     = ['1min', '3min', '5min', '15min']
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1

# ── ELECTRA / RTD knobs ──
SEQ         = 64                      # window parity with the base + downstream MV_SEQ
MASK_RATIO  = float(os.environ.get('MASK_RATIO', '0.15'))   # fraction of bars replaced
RTD_WEIGHT  = float(os.environ.get('RTD_WEIGHT', '5.0'))    # bce weight vs recon
GEN_WIDTH   = int(os.environ.get('GEN_WIDTH', '48'))        # generator size — the strength knob
NEW_CHANNELS = 3                      # overcomplete adapter — sweep-winner setting of the base
FREEZE_ENCODER_LAYERS = int(os.environ.get('FREEZE_ENCODER_LAYERS', '2'))  # anti-forgetting (base=frz2)

# ── TRAINING ──
BATCH   = int(os.environ.get('BATCH', '512'))
EPOCHS  = int(os.environ.get('EPOCHS', '60'))
STEPS   = 200
LR      = float(os.environ.get('LR', '1e-4'))               # gentle: a REFINE of a proven base
WEIGHT_DECAY, PATIENCE = 0.05, 8
CONTROLS = ()                          # honest-ruler controls run downstream (WF), not here
PROBE = True                           # frozen-embedding probe = the representation gate
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
print(f'   mask={MASK_RATIO} rtd_w={RTD_WEIGHT} gen_width={GEN_WIDTH} lr={LR:.1e} '
      f'batch={BATCH} frz={FREEZE_ENCODER_LAYERS}')
print(f'   OUTPUT -> {OUT_PATH}   (promoted bases UNTOUCHED)')


# ======================================= CELL 3 — TRAIN ========================================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH, tickers=TICKERS, tfs=TFS,
    pretext='electra', backbone_ckpt=WARM_CKPT,
    seq=SEQ, mask_ratio=MASK_RATIO, rtd_weight=RTD_WEIGHT, gen_width=GEN_WIDTH,
    new_channels=NEW_CHANNELS,
    batch=BATCH, epochs=EPOCHS, steps_per_epoch=STEPS, lr=LR, weight_decay=WEIGHT_DECAY,
    patience=PATIENCE, val_frac=VAL_FRAC, holdout_start=HOLDOUT_START,
    controls=CONTROLS, probe=PROBE, resume=RESUME,
    freeze_encoder_layers=FREEZE_ENCODER_LAYERS, device=device.type, seed=SEED)

print('\n' + '=' * 60 + '\n  SSL STAGE 4 (ELECTRA replaced-candle detection) VERDICT\n' + '=' * 60)
for k, v in verdict.items():
    if k not in ('history', 'epochs'):
        print(f'  {k:>22}: {v}')

# ── LEARNING-VERIFICATION BLOCK — is the discriminator actually learning (and not cheating)? ──
hist = [h for h in (verdict.get('epochs') or []) if isinstance(h, dict) and 'rtd_bal_acc' in h]
print('-' * 60)
if hist:
    first, best = hist[0], max(hist, key=lambda h: h['rtd_bal_acc'])
    last = hist[-1]
    ba0, ba_best, ba_last = first['rtd_bal_acc'], best['rtd_bal_acc'], last['rtd_bal_acc']
    checks = {
        'learning (bal_acc rose >= +0.03 off start)': ba_best >= ba0 + 0.03,
        'non-trivial (bal_acc <= 0.97, no shortcut tell)': ba_best <= 0.97,
        'above chance (bal_acc >= 0.55)': ba_best >= 0.55,
        'both error modes off extremes (recall/realacc in 0.2..0.995)':
            0.2 <= last['fake_recall'] <= 0.995 and 0.2 <= last['real_acc'] <= 0.995,
        'no collapse (std > 0.01)': last.get('std', 1.0) > 0.01,
    }
    print(f'  RTD LEARNING CHECK: start bal_acc={ba0:.3f} -> best={ba_best:.3f} last={ba_last:.3f}'
          f' | fake_recall={last["fake_recall"]:.3f} real_acc={last["real_acc"]:.3f}'
          f' gen_mse={last.get("gen_mse", float("nan")):.4f}')
    for name, passed in checks.items():
        print(f'    {name:58s} {"✓" if passed else "✗"}')
    if all(checks.values()):
        print('  => DISCRIMINATOR IS LEARNING in the healthy band.')
    elif ba_best > 0.97:
        print('  => TASK TOO EASY — likely a shortcut tell or too-weak generator: raise GEN_WIDTH'
              ' and/or MASK_RATIO, re-run.')
    elif ba_best < 0.55:
        print('  => NOT LEARNING — generator too strong or encoder blind: lower GEN_WIDTH, raise'
              ' RTD_WEIGHT, re-run.')
    else:
        print('  => MIXED — inspect the per-epoch log before spending the downstream benchmark.')
else:
    print('  (no rtd_bal_acc in history — inspect the per-epoch log above)')
print('=' * 60)
print(f'\nadapted encoder  -> {OUT_PATH}   (promoted bases UNTOUCHED)')
print('\nNext: the SHIP gate is unchanged — one-shot 2026 WR@3R vs the current base'
      ' + generality probes (forecast/regime):')
print(f'  S3_CKPT={OUT_PATH}  python3 colabs/mantis_2026_benchmark.py   (promote iff it wins)')
