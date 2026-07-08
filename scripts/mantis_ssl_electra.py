# ==============================================================================
# MANTIS SSL STAGE 4 — TURN-ELECTRA: REPLACED-TURN DETECTION (Colab GPU)
# ==============================================================================
#
# ██ THE IDEA — teach the encoder REAL TURN vs FAKE TURN, as pure SSL ██
#   The pivot strategy enters AT the swing; its live losers are FAKE turns (relief bounces in a
#   grinding trend that look like reversals and die). Prior discriminative refines missed the event:
#   replaced-candle ELECTRA corrupted UNIFORM bars; break-hold anchored on the BREAK bar (after the
#   entry). TURN-ELECTRA keeps the objective 100% self-supervised and changes WHERE the corruption
#   lands (the salient-span insight): span-mask the regions AROUND DETECTED TURNS (local swing
#   highs/lows — the same structural event the pivot trades), let a deliberately-weak generator fill
#   each masked turn with a PLAUSIBLE alternative development — functionally a SYNTHETIC FAKE TURN —
#   and train the encoder to label every bar real-vs-replaced. To win, the encoder must learn how
#   GENUINE turns develop vs plausible imposters: the fakeout-vs-real skill, zero labels, generic.
#     loss = gen_recon(masked) + RTD_WEIGHT*bce(all bars) + RECON_WEIGHT*enc_recon(clean window)
#   ABLATION: TURN_BIAS=0 -> uniform span-ELECTRA (does TURN placement earn the lift?).
#
# ── HOW WE VERIFY IT'S LEARNING (read the per-epoch log + final block) ──
#   rtd_bal_acc  BALANCED accuracy (fake-recall + real-acc)/2 — a lazy all-real predictor scores
#                0.5. LEARNING = rises off ~0.5 into the ~0.60-0.95 band. >0.97 = shortcut tell or
#                too-weak generator (raise GEN_WIDTH / MASK_RATIO); ~0.50 flat = not learning.
#   turn_cov     fraction of masked bars within ±TURN_W of a detected turn — the corruption is only
#                doing its job if this is HIGH (~0.8+ at TURN_BIAS=0.85). Low = placement broken.
#   fake_recall / real_acc  the two error modes (both off the 0/1 extremes).
#   gen_mse      generator plausibility — falls early then flattens.
#   enc_recon    the anchor — should DROP (encoder rebuilding the clean window).
#   std          embedding-collapse guard (> 0.01, and NOT drifting toward 2).
#   PROBE        report-only gate: mean_core_delta stays positive (didn't destroy the base lineage).
#
# SHIP GATE (per the user's rule — judge ONLY on the metric specific to what this tests): fakeout
#   discrimination among COUNTER-TREND / turn pivots (the produce alignment table WR@3R-by-score),
#   NOT the aggregate (that's the forecasting objective's home turf). Pipeline: this run -> pivot WF
#   (2026 untouched) -> produce dry-run (HOLDOUT_START=2025-01-01) -> read the counter-trend table
#   vs the base's. rtd_bal_acc / probe are tickets, never verdicts.
#
# SAFETY: writes a DISTINCT checkpoint (mantis_ssl_turnelectra.pt). NEVER overwrites the promoted
#   bases (ctr_seq2seq / seq2seq / ohlcv) — preflight hard-fails.
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
OUT_PATH  = os.environ.get('OUT_PATH', '/content/drive/MyDrive/AI_Models/mantis_ssl_turnelectra.pt')

# ── CORPUS (same universe as every stage — the ruler must not drift) ──
TICKERS = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'CL', 'ZB', 'ZN']
TFS     = ['1min', '3min', '5min', '15min']
HOLDOUT_START = '2026-01-01'          # EXCLUDED from SSL (downstream OOS stays clean)
VAL_FRAC      = 0.1

# ── TURN-ELECTRA knobs ──
SEQ         = 64                      # window parity with the base + downstream MV_SEQ
MASK_RATIO  = float(os.environ.get('MASK_RATIO', '0.2'))    # fraction of bars replaced
SPAN_MEAN   = float(os.environ.get('SPAN_MEAN', '4'))       # geometric mean span length (bars)
SPAN_MAX    = int(os.environ.get('SPAN_MAX', '10'))
TURN_W      = int(os.environ.get('TURN_W', '3'))            # swing neighborhood (±bars)
TURN_BIAS   = float(os.environ.get('TURN_BIAS', '0.85'))    # P(span centered on a turn); 0 = uniform ablation
RTD_WEIGHT  = float(os.environ.get('RTD_WEIGHT', '5.0'))    # bce weight (0 = denoising-AE ablation)
RECON_WEIGHT = float(os.environ.get('RECON_WEIGHT', '1.0')) # encoder-recon anchor (0 = pure discrim)
GEN_WIDTH   = int(os.environ.get('GEN_WIDTH', '48'))        # generator size — the strength knob
NEW_CHANNELS = 3                      # overcomplete adapter — sweep-winner setting of the base
FREEZE_ENCODER_LAYERS = int(os.environ.get('FREEZE_ENCODER_LAYERS', '2'))  # anti-forgetting (base=frz2)

# ── TRAINING ──
BATCH   = int(os.environ.get('BATCH', '512'))
EPOCHS  = int(os.environ.get('EPOCHS', '60'))               # cosine LR anneals over EXACTLY this many
STEPS   = 200
LR      = float(os.environ.get('LR', '1e-4'))               # gentle: a REFINE of a proven base
WEIGHT_DECAY = 0.05
PATIENCE = int(os.environ.get('PATIENCE', str(max(8, EPOCHS // 6))))   # scales with EPOCHS
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
print(f'   turn-electra [{"anchored" if RECON_WEIGHT > 0 else "pure-discrim(drift risk)"}] '
      f'| mask={MASK_RATIO} span={SPAN_MEAN}/{SPAN_MAX} turn_w={TURN_W} turn_bias={TURN_BIAS} '
      f'rtd_w={RTD_WEIGHT} recon_w={RECON_WEIGHT} gen_width={GEN_WIDTH} '
      f'lr={LR:.1e} batch={BATCH} frz={FREEZE_ENCODER_LAYERS} patience={PATIENCE}')
print(f'   OUTPUT -> {OUT_PATH}   (promoted bases UNTOUCHED)')


# ======================================= CELL 3 — TRAIN ========================================
verdict = ssl.loop_ssl(
    data_dir=DATA_DIR, out_path=OUT_PATH, tickers=TICKERS, tfs=TFS,
    pretext='electra', backbone_ckpt=WARM_CKPT,
    seq=SEQ, mask_ratio=MASK_RATIO, span_mean=SPAN_MEAN, span_max=SPAN_MAX,
    turn_w=TURN_W, turn_bias=TURN_BIAS, rtd_weight=RTD_WEIGHT, recon_weight=RECON_WEIGHT,
    gen_width=GEN_WIDTH, new_channels=NEW_CHANNELS,
    batch=BATCH, epochs=EPOCHS, steps_per_epoch=STEPS, lr=LR, weight_decay=WEIGHT_DECAY,
    patience=PATIENCE, val_frac=VAL_FRAC, holdout_start=HOLDOUT_START,
    controls=CONTROLS, probe=PROBE, resume=RESUME,
    freeze_encoder_layers=FREEZE_ENCODER_LAYERS, device=device.type, seed=SEED)

print('\n' + '=' * 60 + '\n  SSL STAGE 4 (TURN-ELECTRA replaced-turn detection) VERDICT\n' + '=' * 60)
for k, v in verdict.items():
    if k not in ('history', 'epochs'):
        print(f'  {k:>22}: {v}')

# ── LEARNING-VERIFICATION BLOCK — is it learning real-vs-fake TURNS (and not cheating)? ──
hist = [h for h in (verdict.get('epochs') or []) if isinstance(h, dict) and 'rtd_bal_acc' in h]
print('-' * 60)
if hist:
    first, best = hist[0], max(hist, key=lambda h: h['rtd_bal_acc'])
    last = hist[-1]
    ba0, ba_best, ba_last = first['rtd_bal_acc'], best['rtd_bal_acc'], last['rtd_bal_acc']
    er0, er_last = first.get('enc_recon', float('nan')), last.get('enc_recon', float('nan'))
    anchored = RECON_WEIGHT == 0 or (er_last == er_last and er0 == er0 and er_last <= er0)  # NaN-safe
    tc = last.get('turn_cov', float('nan'))
    checks = {
        'learning (bal_acc rose >= +0.03 off start)': ba_best >= ba0 + 0.03,
        'above chance (bal_acc >= 0.55)': ba_best >= 0.55,
        'not a shortcut (bal_acc <= 0.97)': ba_best <= 0.97,
        'both error modes off extremes (recall/realacc in 0.2..0.995)':
            0.2 <= last['fake_recall'] <= 0.995 and 0.2 <= last['real_acc'] <= 0.995,
        'corruption IS turn-focused (turn_cov >= 0.6)': tc == tc and tc >= 0.6,
        'no collapse (std > 0.01)': last.get('std', 1.0) > 0.01,
        'ANCHOR: enc_recon dropped (encoder rebuilding clean window)': anchored,
        'NO DRIFT: emb_std <= 1.6': last.get('std', 1.0) <= 1.6,
    }
    print(f'  TURN-ELECTRA CHECK: start bal_acc={ba0:.3f} -> best={ba_best:.3f} last={ba_last:.3f}'
          f' | fake_recall={last["fake_recall"]:.3f} real_acc={last["real_acc"]:.3f}'
          f' | turn_cov={tc:.2f} gen_mse={last.get("gen_mse", float("nan")):.4f}'
          f' | enc_recon {er0:.4f}->{er_last:.4f} emb_std={last.get("std", float("nan")):.3f}')
    for name, passed in checks.items():
        print(f'    {name:58s} {"✓" if passed else "✗"}')
    if all(checks.values()):
        print('  => LEARNING real-vs-fake TURNS in the healthy band. Next: pivot WF -> produce'
              ' dry-run -> the COUNTER-TREND table (the ONLY valid verdict for this pretext).')
    elif ba_best > 0.97:
        print('  => TASK TOO EASY — shortcut tell or too-weak generator: raise GEN_WIDTH and/or'
              ' MASK_RATIO, re-run.')
    elif ba_best < 0.55:
        print('  => NOT LEARNING — generator too strong or encoder blind: lower GEN_WIDTH, raise'
              ' RTD_WEIGHT, re-run.')
    else:
        print('  => MIXED — inspect the per-epoch log before spending downstream runs.')
else:
    print('  (no rtd_bal_acc in history — inspect the per-epoch log above)')
print('=' * 60)
print(f'\nadapted encoder  -> {OUT_PATH}   (promoted bases UNTOUCHED)')
print('\nNext (2026 stays reserved): pivot WF with BACKBONE_CKPT={} ,'.format(OUT_PATH))
print('then MODE=produce HOLDOUT_START=2025-01-01 -> read the counter-trend alignment table'
      ' vs the base (fakeout discrimination among counter-trend pivots = the verdict).')
