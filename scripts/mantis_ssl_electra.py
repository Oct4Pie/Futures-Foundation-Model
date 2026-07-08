# ==============================================================================
# MANTIS SSL STAGE 4 — BREAK-HOLD DISCRIMINATIVE PRETEXT (the rewritten electra slot, Colab GPU)
# ==============================================================================
#
# ██ THE IDEA — make FAKEOUT-DETECTION the objective, not a hope ██
#   Every prior objective (reconstruct/forecast candles) was INDIRECT — we hoped the encoder would
#   incidentally learn "real break vs fakeout." This makes it the TASK. At each window's causal
#   ANCHOR (last bar), a structural break (close through the swing high/low of the prior
#   BREAK_LOOKBACK bars) is labeled HOLD or FAIL over the next HOLD_K bars:
#     HOLD = price extends >= HOLD_THETA*ATR in the break direction BEFORE retracing the broken level
#     FAIL = it falls back through the level (trap) OR stalls (dead bounce) within HOLD_K bars.
#   That hold-vs-fail label is the atomic unit of a fakeout — self-supervised from raw OHLCV, millions
#   of examples, NO strategy outcome, NO leak (the HOLD_K future bars are RESERVED, never encoded).
#   This is a GENERIC FOUNDATION objective: it sharpens the SHIPPED embedding so EVERY downstream head
#   (pivot, kalman, any strategy) inherits an encoder that knows real breaks from fakeouts.
#
#   Discriminative like electra (encoder is the keeper, a binary head reads it out) but GENERATOR-FREE
#   — the "fake" is a REAL failed break, not a synthesized candle (no GAN, no OHLC-clamp cheat). The
#   one electra-v2 piece that worked MECHANICALLY is KEPT: the encoder-side RECON anchor.
#     loss = RECON_WEIGHT * enc_recon(clean window) + HOLD_WEIGHT * BCE(hold | is_break)
#
# ── HOW WE VERIFY IT'S LEARNING (read the per-epoch log + final block) ──
#   hold_bal_acc  BALANCED accuracy over break windows (hold_recall + fail_recall)/2 — the honest
#                 signal (a lazy majority predictor scores 0.5). LEARNING = rises off ~0.5.
#                 >~0.55 = there IS fakeout signal in OHLCV the indirect objectives missed.
#                 ~0.50 flat = the discriminator is NOT in price -> it's order flow (a CLEAN verdict).
#   hold_recall / fail_recall  the two error modes (both should be off the 0/1 extremes).
#   break_rate    fraction of windows that are breaks (label density); hold_rate = of those, % held.
#   enc_recon     the anchor — should DROP (encoder rebuilding the clean window).
#   std           embedding-collapse guard (must stay > 0.01, and NOT drift to 2+).
#   PROBE         report-only gate: mean_core_delta stays positive (didn't destroy the base lineage).
#
# SHIP GATE UNCHANGED (never the pretext metrics): downstream one-shot 2026 WR@3R vs the current base
#   + generality probes. This checkpoint is a CANDIDATE:
#     S3_CKPT=<this .pt>  python3 colabs/mantis_2026_benchmark.py   (promote iff it wins)
#
# SAFETY: writes a DISTINCT checkpoint (mantis_ssl_breakhold.pt). NEVER overwrites the promoted bases
#   (ctr_seq2seq / seq2seq / ohlcv) — preflight hard-fails.
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
OUT_PATH  = os.environ.get('OUT_PATH', '/content/drive/MyDrive/AI_Models/mantis_ssl_breakhold.pt')

# ── CORPUS (same universe as every stage — the ruler must not drift) ──
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']
TFS     = ['1min', '3min', '5min', '15min']
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1

# ── BREAK-HOLD knobs ──
SEQ           = 64                     # window parity with the base + downstream MV_SEQ
HOLD_K        = int(os.environ.get('HOLD_K', '12'))          # future bars the label races over
BREAK_LOOKBACK = int(os.environ.get('BREAK_LOOKBACK', '20')) # swing window the break must clear
HOLD_THETA    = float(os.environ.get('HOLD_THETA', '1.0'))  # hold = extend >= theta*ATR before retrace
HOLD_WEIGHT   = float(os.environ.get('HOLD_WEIGHT', '5.0'))  # BCE weight (0 = denoising-AE ablation)
RECON_WEIGHT  = float(os.environ.get('RECON_WEIGHT', '1.0')) # encoder-recon anchor (0 = pure discrim)
NEW_CHANNELS  = 3                     # overcomplete adapter — sweep-winner setting of the base
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
print(f'   break-hold [{"v-anchored" if RECON_WEIGHT > 0 else "pure-discrim(drift risk)"}] '
      f'| hold_k={HOLD_K} lookback={BREAK_LOOKBACK} theta={HOLD_THETA} '
      f'hold_w={HOLD_WEIGHT} recon_w={RECON_WEIGHT} lr={LR:.1e} batch={BATCH} frz={FREEZE_ENCODER_LAYERS}')
print(f'   OUTPUT -> {OUT_PATH}   (promoted bases UNTOUCHED)')


# ======================================= CELL 3 — TRAIN ========================================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH, tickers=TICKERS, tfs=TFS,
    pretext='electra', backbone_ckpt=WARM_CKPT,
    seq=SEQ, hold_k=HOLD_K, break_lookback=BREAK_LOOKBACK, hold_theta=HOLD_THETA,
    hold_weight=HOLD_WEIGHT, recon_weight=RECON_WEIGHT, new_channels=NEW_CHANNELS,
    batch=BATCH, epochs=EPOCHS, steps_per_epoch=STEPS, lr=LR, weight_decay=WEIGHT_DECAY,
    patience=PATIENCE, val_frac=VAL_FRAC, holdout_start=HOLDOUT_START,
    controls=CONTROLS, probe=PROBE, resume=RESUME,
    freeze_encoder_layers=FREEZE_ENCODER_LAYERS, device=device.type, seed=SEED)

print('\n' + '=' * 60 + '\n  SSL STAGE 4 (BREAK-HOLD discriminative) VERDICT\n' + '=' * 60)
for k, v in verdict.items():
    if k not in ('history', 'epochs'):
        print(f'  {k:>22}: {v}')

# ── LEARNING-VERIFICATION BLOCK — is the discriminator actually learning (and is fakeout signal in OHLCV)? ──
hist = [h for h in (verdict.get('epochs') or []) if isinstance(h, dict) and 'hold_bal_acc' in h]
print('-' * 60)
if hist:
    first, best = hist[0], max(hist, key=lambda h: h['hold_bal_acc'])
    last = hist[-1]
    ba0, ba_best, ba_last = first['hold_bal_acc'], best['hold_bal_acc'], last['hold_bal_acc']
    er0, er_last = first.get('enc_recon', float('nan')), last.get('enc_recon', float('nan'))
    anchored = RECON_WEIGHT == 0 or (er_last == er_last and er0 == er0 and er_last <= er0)  # NaN-safe
    std_ok = last.get('std', 1.0) <= 1.6                   # anchored stays ~1 (drift guard)
    checks = {
        'learning (bal_acc rose >= +0.03 off start)': ba_best >= ba0 + 0.03,
        'fakeout signal in OHLCV (bal_acc >= 0.55)': ba_best >= 0.55,
        'not a shortcut (bal_acc <= 0.97)': ba_best <= 0.97,
        'both error modes off extremes (recall in 0.2..0.995)':
            0.2 <= last['hold_recall'] <= 0.995 and 0.2 <= last['fail_recall'] <= 0.995,
        'no collapse (std > 0.01)': last.get('std', 1.0) > 0.01,
        'ANCHOR: enc_recon dropped (encoder rebuilding clean window)': anchored,
        'NO DRIFT: emb_std <= 1.6': std_ok,
    }
    print(f'  BREAK-HOLD CHECK: start bal_acc={ba0:.3f} -> best={ba_best:.3f} last={ba_last:.3f}'
          f' | hold_recall={last["hold_recall"]:.3f} fail_recall={last["fail_recall"]:.3f}'
          f' | break_rate={last.get("break_rate", float("nan")):.3f}'
          f' hold_rate={last.get("hold_rate", float("nan")):.3f}'
          f' | enc_recon {er0:.4f}->{er_last:.4f} emb_std={last.get("std", float("nan")):.3f}')
    for name, passed in checks.items():
        print(f'    {name:58s} {"✓" if passed else "✗"}')
    if all(checks.values()):
        print('  => DISCRIMINATOR IS LEARNING — real fakeout signal exists in OHLCV. Spend the benchmark.')
    elif ba_best < 0.55:
        print('  => bal_acc ~0.5: the fakeout discriminator is likely NOT in price (order-flow) —'
              ' a CLEAN, valuable negative. Inspect break_rate/hold_rate before concluding.')
    elif ba_best > 0.97:
        print('  => TASK TOO EASY — a shortcut tell (check break detection). Inspect the log.')
    else:
        print('  => MIXED — inspect the per-epoch log before spending the downstream benchmark.')
else:
    print('  (no hold_bal_acc in history — inspect the per-epoch log above)')
print('=' * 60)
print(f'\nadapted encoder  -> {OUT_PATH}   (promoted bases UNTOUCHED)')
print('\nNext: the SHIP gate is unchanged — one-shot 2026 WR@3R vs the current base'
      ' + generality probes:')
print(f'  S3_CKPT={OUT_PATH}  python3 colabs/mantis_2026_benchmark.py   (promote iff it wins)')
