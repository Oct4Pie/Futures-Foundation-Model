"""VALIDATE that a foundation checkpoint encodes a CORRECT market-state representation —
interpretable proof beyond the training-time A-E numbers. Answers "does the embedding actually
capture market structure?" with two direct, visual-in-numbers tests, and A/Bs the candidate
against seq2seq so "correct" is measured as a LIFT, not an absolute.

  1. NEAREST-NEIGHBOR RETRIEVAL (temporal/structural consistency, made concrete)
     For random query windows, is the embedding-nearest OTHER window structurally similar? We
     score similarity by an INDEPENDENT descriptor the encoder never sees as a label — per-window
     (trend slope, volatility, range) — and compare the descriptor-distance of embedding-neighbors
     vs random pairs. A correct representation -> neighbors are far more similar than random.

  2. KNOWN-REGIME SEPARATION (does it place different market states in different regions?)
     Bucket windows by a SIMPLE, transparent regime label computed from raw price (NOT used in
     training): TREND-UP / TREND-DOWN / CHOP / HIGH-VOL. A correct representation -> within-regime
     embedding distance << between-regime (report the ratio + a linear-probe accuracy on the
     frozen embedding as the summary number). This is the honest "usefulness-adjacent" check: the
     probe is REPORT-ONLY (we banked that probe != trading value), but regime separability is the
     literal thing a regime representation should have.

Label-free training; these labels exist ONLY to GRADE the geometry after the fact. Runs on the
frozen encoder (embed_windows), CPU-fine. A/B: pass --ckpt <candidate> --base <seq2seq>.

    python3 scripts/validate_regime_representation.py \
        --ckpt checkpoints/mantis_ssl_regime.pt --base checkpoints/mantis_ssl_seq2seq.pt \
        --data data --tickers ES,NQ,GC --tfs 3min --n 3000
"""
import argparse
import os

import numpy as np
import pandas as pd

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

SEQ = 64


def _windows(data_dir, tickers, tfs, n, seed=0):
    """Raw OHLCV windows [N,5,SEQ] + a per-window transparent descriptor for grading."""
    rng = np.random.default_rng(seed)
    W, D = [], []
    per = max(1, n // (len(tickers) * len(tfs)))
    for tk in tickers:
        for tf in tfs:
            p = os.path.join(data_dir, f'{tk}_{tf}.csv')
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p, usecols=['open', 'high', 'low', 'close', 'volume'])
            o, h, l, c, v = (df[k].to_numpy(float) for k in
                             ('open', 'high', 'low', 'close', 'volume'))
            T = len(c)
            if T < SEQ + 2:
                continue
            starts = rng.integers(0, T - SEQ, size=min(per, T - SEQ))
            for s in starts:
                sl = slice(s, s + SEQ)
                W.append(np.stack([o[sl], h[sl], l[sl], c[sl], v[sl]]).astype(np.float32))
                cs = c[sl]
                cn = (cs - cs.mean()) / (cs.std() + 1e-9)                 # scale-free close
                slope = np.polyfit(np.arange(SEQ), cn, 1)[0]             # trend direction/strength
                vol = np.abs(np.diff(cs)).mean() / (np.abs(cs).mean() + 1e-9)   # noisiness
                rng_atr = (h[sl] - l[sl]).mean() / (np.abs(cs).mean() + 1e-9)   # range
                D.append([slope, vol, rng_atr])
    return np.stack(W), np.asarray(D, np.float32)


def _regime_label(desc):
    """Transparent 4-way regime from the descriptor (post-hoc grading label, NOT training):
    HIGH-VOL if noisiness in the top tercile; else TREND-UP / TREND-DOWN by slope sign beyond a
    deadband; else CHOP. Simple and inspectable on purpose."""
    slope, vol = desc[:, 0], desc[:, 1]
    vhi = np.quantile(vol, 2 / 3)
    sdz = np.quantile(np.abs(slope), 1 / 3)                              # slope deadband
    lab = np.full(len(desc), 'chop', dtype=object)
    lab[slope > sdz] = 'trend_up'
    lab[slope < -sdz] = 'trend_dn'
    lab[vol >= vhi] = 'high_vol'                                         # vol dominates
    return lab


def _embed(W, ckpt, *, model_id, model_version, device='cpu'):
    from futures_foundation.finetune.pretext._torch.common import embed_windows
    z = embed_windows(
        W, ckpt=ckpt, model_id=model_id, model_version=model_version, device=device,
    )
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)         # cosine geometry


def _nn_retrieval(z, desc, seed=0):
    """Mean descriptor-distance of each window's EMBEDDING-nearest neighbor vs a RANDOM window.
    ratio = nn_dist / rand_dist; << 1 means embedding neighbors are genuinely similar market
    states. desc standardized so all three descriptor dims count equally."""
    rng = np.random.default_rng(seed)
    ds = (desc - desc.mean(0)) / (desc.std(0) + 1e-9)
    sim = z @ z.T
    np.fill_diagonal(sim, -1e9)
    nn = sim.argmax(1)
    nn_d = np.linalg.norm(ds - ds[nn], axis=1).mean()
    rand = rng.permutation(len(z))
    rand_d = np.linalg.norm(ds - ds[rand], axis=1).mean()
    return float(nn_d), float(rand_d), float(nn_d / (rand_d + 1e-9))


def _regime_separation(z, lab):
    """within-regime vs between-regime mean cosine distance + a report-only linear-probe accuracy
    (5-fold) on the FROZEN embedding. Separation ratio << 1 and probe >> chance => the geometry
    places regimes in distinct regions."""
    d = 1.0 - z @ z.T
    labs = np.array(lab)
    same = labs[:, None] == labs[None, :]
    eye = np.eye(len(z), dtype=bool)
    within = d[same & ~eye].mean()
    between = d[~same].mean()
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        acc = float(cross_val_score(LogisticRegression(max_iter=500), z, labs, cv=5).mean())
    except Exception:
        acc = float('nan')
    chance = float(pd.Series(labs).value_counts(normalize=True).max())
    return float(within), float(between), float(within / (between + 1e-9)), acc, chance


def _run(name, ckpt, W, desc, lab, *, model_id, model_version, device):
    z = _embed(
        W, ckpt, model_id=model_id, model_version=model_version, device=device,
    )
    nn_d, rand_d, nn_ratio = _nn_retrieval(z, desc)
    wi, bw, sep_ratio, acc, chance = _regime_separation(z, lab)
    print(f"\n=== {name} ===  ({ckpt})")
    print(f"  1. NN retrieval : neighbor descriptor-dist {nn_d:.3f} vs random {rand_d:.3f} "
          f"-> ratio {nn_ratio:.3f}  (lower=better; <1 = neighbors are similar states)")
    print(f"  2. regime sep   : within {wi:.3f} vs between {bw:.3f} -> ratio {sep_ratio:.3f} "
          f"(lower=better)")
    print(f"     linear probe : {acc:.3f} vs chance {chance:.3f}  (report-only; regime separability)")
    return {'nn_ratio': nn_ratio, 'sep_ratio': sep_ratio, 'probe': acc, 'chance': chance}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='candidate checkpoint (e.g. regime.pt)')
    ap.add_argument('--base', default=None, help='baseline to A/B against (e.g. seq2seq.pt)')
    ap.add_argument('--data', default='data')
    ap.add_argument('--tickers', default='ES,NQ,GC')
    ap.add_argument('--tfs', default='3min')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--model-version', choices=('v1', 'v2'), required=True)
    ap.add_argument('--candidate-admission-report', required=True)
    ap.add_argument('--base-admission-report')
    a = ap.parse_args()

    from futures_foundation.finetune.native_contracts import verify_admission_report
    arm_key = 'mantis_v2' if a.model_version == 'v2' else 'mantis_v1'
    model_id = 'paris-noah/MantisV2' if a.model_version == 'v2' else 'paris-noah/Mantis-8M'
    verify_admission_report(
        a.candidate_admission_report, arm_key=arm_key, track='C',
        route='historical_custom_representation_extraction', require_training=False,
        required_artifacts={'checkpoint': a.ckpt},
    )
    if a.base:
        if not a.base_admission_report:
            ap.error('--base requires --base-admission-report')
        verify_admission_report(
            a.base_admission_report, arm_key=arm_key, track='C',
            route='historical_custom_representation_extraction', require_training=False,
            required_artifacts={'checkpoint': a.base},
        )
    elif a.base_admission_report:
        ap.error('--base-admission-report requires --base')

    W, desc = _windows(a.data, a.tickers.split(','), a.tfs.split(','), a.n)
    lab = _regime_label(desc)
    print(f"windows={len(W)}  regimes: " +
          ', '.join(f'{k}={int(v)}' for k, v in pd.Series(lab).value_counts().items()))

    cand = _run(
        'CANDIDATE', a.ckpt, W, desc, lab,
        model_id=model_id, model_version=a.model_version, device=a.device,
    )
    if a.base:
        base = _run(
            'BASELINE', a.base, W, desc, lab,
            model_id=model_id, model_version=a.model_version, device=a.device,
        )
        print("\n" + "=" * 60 + "\n  A/B VERDICT (candidate vs baseline)\n" + "=" * 60)
        dnn = base['nn_ratio'] - cand['nn_ratio']            # + = candidate neighbors more similar
        dsep = base['sep_ratio'] - cand['sep_ratio']         # + = candidate separates regimes better
        dprobe = cand['probe'] - base['probe']               # + = candidate more regime-separable
        print(f"  NN retrieval ratio : {cand['nn_ratio']:.3f} vs {base['nn_ratio']:.3f}  "
              f"({'BETTER' if dnn > 0.005 else 'worse' if dnn < -0.005 else 'tie'} {dnn:+.3f})")
        print(f"  regime sep ratio   : {cand['sep_ratio']:.3f} vs {base['sep_ratio']:.3f}  "
              f"({'BETTER' if dsep > 0.005 else 'worse' if dsep < -0.005 else 'tie'} {dsep:+.3f})")
        print(f"  regime probe acc   : {cand['probe']:.3f} vs {base['probe']:.3f}  "
              f"({'BETTER' if dprobe > 0.005 else 'worse' if dprobe < -0.005 else 'tie'} {dprobe:+.3f})")
        print("=" * 60)
        print("READ: 'BETTER' on NN + regime sep => stage-3 improved the market representation as a"
              "\n      representation (independent of the WR@3R trading benchmark).")


if __name__ == '__main__':
    main()
