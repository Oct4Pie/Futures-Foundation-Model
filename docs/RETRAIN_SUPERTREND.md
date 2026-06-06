# SuperTrend Chronos — Quarterly Retrain Runbook

Repeatable procedure to refresh data and rebuild the live SuperTrend Chronos+XGBoost
bundle. **Same scripts every time — no code changes.** Cadence: **every ~3 months.**

- Last run: **2026-06-05** (data through 2026-06-04). Next due: **~2026-09**.
- Architecture is settled: vanilla **frozen `amazon/chronos-bolt-tiny`** embed + XGBoost
  selection head. Do NOT fine-tune the backbone (tested, no lift). Protect the live model.
- All commands run from the repo root: `cd /Users/johnmcruz/Projects/Futures-Foundation-Model`

---

## 0. Pre-flight — verify the backbone (every time)
The fine-tune wiring gap is real: an unset env var silently uses vanilla (which is what we
WANT here). Confirm it's unset so we ship vanilla bolt-tiny:
```bash
echo "CHRONOS_FT_CKPT=${CHRONOS_FT_CKPT:-<UNSET = vanilla bolt-tiny ✓>}"
# if it's SET, unset it for this whole procedure:
unset CHRONOS_FT_CKPT
```

## 1. Append new DataBento data
New files arrive in `databento/` as `*.dbn.zst` (multi-ticker) or `*.csv.zst` (single).
Preview first (NO writes), then commit (auto-backs-up to `data/backup_<stamp>/`):
```bash
# PREVIEW — shows seam + continuity report, writes only to /tmp
python3 databento/append_update.py <NEW_FILE.dbn.zst> --tickers NQ,ES,GC,RTY,YM --tfs 3min,5min
# COMMIT — writes data/{TK}_{3min,5min}.csv after you eyeball the preview
python3 databento/append_update.py <NEW_FILE.dbn.zst> --tickers NQ,ES,GC,RTY,YM --tfs 3min,5min --commit
```
- Omit `--tickers` to take every instrument in the file.
- Splices at the new file's start (overlap rebuilt bit-for-bit → continuous, no dup seam).
- Weekend gaps are real market closure — expected, left as-is.
- 2026-06: ES/NQ/GC/RTY/YM updated; SI was not refreshed that round (fine — optional).

## 2. Regenerate the FFM feature parquet
`prepare_data` globs ALL `*.csv` and splits the name on `_`, so 3min + 5min in one dir
COLLIDE. Isolate the 3min files into a clean dir first (COPY, not symlink — symlinks fail
in /tmp). Then build into a NEW dir, verify, and only then replace the canonical one.
```bash
# isolate 3min CSVs (real copies)
RAW=/tmp/ffm_raw_3min; rm -rf $RAW; mkdir -p $RAW
for tk in ES NQ RTY YM GC SI; do cp data/${tk}_3min.csv $RAW/${tk}_3min.csv; done

# generate into a NEW dir (atr_period=20, force=True — same as colabs/prepare_data_3min.py)
python3 - <<'PY'
from futures_foundation import prepare_data
print(prepare_data('/tmp/ffm_raw_3min',
                   'temp/3min_FFM_Prepared_v2_causal_NEW',
                   atr_period=20, force=True))
PY

# VERIFY: 6 tickers, 83 cols, dates extend; CAUSALITY GATE must be 0 mismatches
python3 - <<'PY'
import pandas as pd
from futures_foundation import derive_features, get_model_feature_columns
NEW='temp/3min_FFM_Prepared_v2_causal_NEW'
for tk in ['ES','NQ','RTY','YM','GC','SI']:
    d=pd.read_parquet(f'{NEW}/{tk}_features.parquet', columns=['_datetime'])
    print(tk, len(d), '->', d['_datetime'].max())
df=pd.read_csv('data/NQ_3min.csv').tail(1600).reset_index(drop=True)
batch=derive_features(df,'NQ',atr_period=20); cols=get_model_feature_columns(); bad=0
for i in range(900,len(df),151):
    c=derive_features(df.iloc[:i+1].copy(),'NQ',atr_period=20)
    for col in cols:
        b,v=batch[col].iloc[i],c[col].iloc[i]
        if pd.isna(b) and pd.isna(v): continue
        if abs(float(b)-float(v))>1e-6: bad+=1
print('CAUSALITY mismatches:', bad, '(MUST be 0)')
PY

# swap in (temp/ is regenerable; delete old to avoid confusion)
rm -rf temp/3min_FFM_Prepared_v2_causal
mv temp/3min_FFM_Prepared_v2_causal_NEW temp/3min_FFM_Prepared_v2_causal
rm -rf /tmp/ffm_raw_3min
```

## 3. Walk-forward RE-CERT (must hold before producing)
```bash
python3 -c "import importlib.util as u; s=u.spec_from_file_location('st','colabs/supertrend_chronos.py'); m=u.module_from_spec(s); s.loader.exec_module(m); m.run(seeds=(0,1,2))" 2>&1 | tee /tmp/st_recert.log
```
PASS criterion (honest ruler): **REAL meanR − SHUFFLE ≥ ~0.4R** and beats RANDOM/NAIVE on
≥ most tickers. Historical baseline ≈ **+0.49R**; 2026-06-05 re-cert = **+0.47R** (held).
- The pre-registered verdict prints ❌ FAIL **only** on the risk-head check (MAE > 2.5,
  r ≈ 0) — that is the KNOWN, pre-existing fat-tail weakness, identical to live. The
  signal head passing + edge holding = OK to produce. (Bot uses fixed TP / ignores risk head.)
- If the SIGNAL edge collapses (REAL−SHUFFLE well below ~0.3R), STOP — investigate, do not ship.

## 4. Produce + name the bundle
```bash
python3 colabs/supertrend_chronos_produce.py            # vanilla bolt-tiny, 1mo holdout
# (or --holdout-months 0 to train on every bar)

# back up the current live bundle, then rename the new one
cp -p supertrend_chronos.joblib supertrend_chronos.joblib.bak_$(date +%Y%m%d) 2>/dev/null || true
mv chronos_supertrendchronos_production_*.joblib supertrend_chronos.joblib

# verify the bundle
python3 - <<'PY'
import joblib
b=joblib.load('supertrend_chronos.joblib')
print('ckpt:', b['chronos_ckpt'], '| feat_dim:', b['feat_dim'], '| n_classes:', b['n_classes'])
print('train_span:', b['training_metadata']['train_span'])
print('heads:', type(b['signal_head']).__name__, type(b['risk_head']).__name__)
PY
```
EXPECT: `ckpt=amazon/chronos-bolt-tiny`, `feat_dim=334` (78 handcraft + 256 embed),
`n_classes=2`, `XGBHead` + `XGBRiskHead`. A thin recent holdout month is normal, not a blocker.

## 5. Hand off
- `supertrend_chronos.joblib` is **gitignored** (artifacts never committed).
- It's a **drop-in for algoTrader** (identical 334-d contract). algoTrader does joblib→ONNX
  export on its side. Copy the bundle over per your deploy process.

---

### Notes / gotchas (learned 2026-06-05)
- `databento/append_update.py` and `databento/build_continuous.py` are the only data tools.
- prepare_data 3min/5min collision + symlink-in-/tmp failure → see step 2 (copy 3min only).
- Backups pile up in `data/backup_<stamp>/` and `*.bak_*` — prune occasionally.
- SI optional each round; update only if a fresh SI DataBento file is pulled.
- See memory `project_live_production_model.md` for the why; this doc is the how.
