"""Analyze R_F vs HT-SR: (1) orthogonality across models, (2) incremental
predictive value of R_F over HT-SR for the functional target (angular distance)."""
import os, json
import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, 'results')
ht = json.load(open(os.path.join(R, 'exp_htsr_shootout.json')))
fwd = json.load(open(os.path.join(R, 'exp_metrics_fwd.json')))

MODELS = ['gpt2', 'pythia-410m', 'pythia-1b', 'tinyllama', 'qwen2.5-0.5b',
          'qwen2.5-1.5b', 'phi-2']
HT_KEYS = ['alpha_mean', 'alpha_M', 'sr_M', 'me_M']

print("="*78)
print("(1) Is R_F a re-skin of HT-SR?  Spearman(R_F, metric) per model")
print("="*78)
print(f"{'metric':<11}" + "".join(f"{m.split('-')[0][:7]:>8}" for m in MODELS)
      + f"{'mean|r|':>8}{'sign':>6}")
for k in HT_KEYS:
    rhos = []
    for m in MODELS:
        rho = ht[m]['rho_vs_rf'][k]['rho']
        rhos.append(rho)
    arr = np.array(rhos)
    consistent = 'same' if (arr > 0).all() or (arr < 0).all() else 'FLIP'
    print(f"{k:<11}" + "".join(f"{r:+8.2f}" for r in rhos)
          + f"{np.abs(arr).mean():>8.2f}{consistent:>6}")

print("\n  -> If mean|r| is low AND sign FLIPs, R_F is NOT explained by that HT-SR metric.\n")


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z (1 or more cols).
    Rank-transform, regress out z from rank(x) and rank(y), correlate residuals."""
    rx = stats.rankdata(x); ry = stats.rankdata(y)
    Z = np.column_stack([stats.rankdata(zi) for zi in z])
    Z = np.column_stack([np.ones(len(rx)), Z])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    ex = rx - Z @ bx; ey = ry - Z @ by
    r, p = stats.pearsonr(ex, ey)
    return r, p


print("="*78)
print("(2) Incremental value: does R_F predict angular distance BEYOND HT-SR?")
print("="*78)
for m in ['qwen2.5-0.5b', 'tinyllama']:
    rows = ht[m]['layers']
    rf = np.array([r['rf'] for r in rows])
    ang = np.array(fwd[m]['angular'])
    htm = {k: np.array([r[k] for r in rows], dtype=float) for k in HT_KEYS}
    # align lengths (angular has n_layers entries)
    n = min(len(rf), len(ang)); rf, ang = rf[:n], ang[:n]
    for k in HT_KEYS:
        htm[k] = htm[k][:n]
    print(f"\n--- {m} (n={n}) ---  target = angular distance (1-cos)")
    print(f"  R_F vs angular            : rho={stats.spearmanr(rf,ang)[0]:+.3f}")
    for k in HT_KEYS:
        v = htm[k]; mask = np.isfinite(v)
        rho = stats.spearmanr(v[mask], ang[mask])[0]
        print(f"  {k:<9} vs angular        : rho={rho:+.3f}")
    # partial: R_F vs angular controlling for ALL HT-SR metrics jointly
    Z = [htm[k] for k in HT_KEYS]
    # replace any nan
    ok = np.all(np.isfinite(np.column_stack(Z)), axis=1)
    pr, pp = partial_spearman(rf[ok], ang[ok], [z[ok] for z in Z])
    print(f"  R_F vs angular | ALL HT-SR : partial r={pr:+.3f} (p={pp:.1e})   <-- R_F unique signal")
    # reverse: best HT-SR vs angular controlling for R_F
    for k in HT_KEYS:
        v = htm[k]; ok2 = np.isfinite(v)
        pr2, pp2 = partial_spearman(v[ok2], ang[ok2], [rf[ok2]])
        print(f"  {k:<9} vs angular | R_F  : partial r={pr2:+.3f} (p={pp2:.1e})")
