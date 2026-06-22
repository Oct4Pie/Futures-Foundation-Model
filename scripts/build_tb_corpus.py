"""Strategy-free triple-barrier DIRECTION corpus for bolt FT.
Symmetric k=1.0 ATR barriers, H=60, 3-class (0=UP,1=DOWN,2=NEITHER). Context =
128-bar RAW log-close (bolt instance-norms internally — matches the extractor)."""
import sys, os, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from futures_foundation.chronos._primitives import compute_atr
TICKERS = ['ES','NQ','RTY','YM','GC','SI','CL','ZB','ZN']
TFS = ['1min','3min','5min','15min']
CTX, ATR_P, K, H, STRIDE = 128, 20, 1.0, 60, 16
Xs, ys = [], []
t0 = time.time()
for tk in TICKERS:
    for tf in TFS:
        p = f'data/{tk}_{tf}.csv'
        if not os.path.exists(p):
            print(f'  skip {tk}_{tf} (no file)', flush=True); continue
        df = pd.read_csv(p, usecols=['high','low','close'])
        h,l,c = (df[x].to_numpy(float) for x in ('high','low','close'))
        atr = compute_atr(h,l,c,ATR_P); lp = np.log(c); n = len(c); cnt = 0
        for i in range(max(CTX-1, ATR_P+1), n-H-1, STRIDE):
            a = atr[i]
            if not np.isfinite(a) or a <= 0: continue
            ub, lb = c[i]+K*a, c[i]-K*a
            hs, ls = h[i+1:i+H+1], l[i+1:i+H+1]
            uh = np.argmax(hs >= ub) if (hs >= ub).any() else 10**9
            dh = np.argmax(ls <= lb) if (ls <= lb).any() else 10**9
            y = 2 if uh == dh else (0 if uh < dh else 1)
            Xs.append(lp[i-CTX+1:i+1].astype(np.float32)); ys.append(y); cnt += 1
        print(f'  {tk}_{tf}: {cnt:,} windows', flush=True)
X = np.asarray(Xs, np.float32); y = np.asarray(ys, np.int8)
os.makedirs('temp/tb_corpus', exist_ok=True)
np.save('temp/tb_corpus/X.npy', X); np.save('temp/tb_corpus/y.npy', y)
print(f'\nCORPUS: X={X.shape}  y dist (UP/DOWN/NEITHER)={np.bincount(y, minlength=3)}  ({time.time()-t0:.0f}s)', flush=True)
