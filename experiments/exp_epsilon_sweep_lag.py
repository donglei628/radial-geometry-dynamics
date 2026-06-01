"""
P4 causality: does the loss--geometry lag grow as beta*ln(1/eps)?  (synthetic)

CORRECTION to the spec's M*=sI teacher: with a radial teacher the loss and the
radial geometry are driven by the SAME thing, so they ignite together (Delta~0) --
no lag. The paper's mechanism ("loss's fast drop does not depend on radial
alignment") requires a teacher whose loss is dominated by a NON-radial structure,
with the radial/trace signal carried by a separate, slower mode.

Teacher 'split' (block-diagonal, decoupled timescales):
   M* = blkdiag( B [p x p, random, spectral norm L, ~0 trace -> drives LOSS, fast],
                 c * I [(m-p) x (m-p)  -> drives R_F via trace, slow] ),  L > c.
Then t_loss ~ (1/(eta L)) ln(1/eps),  t_rad ~ (1/(eta c)) ln(1/eps),
so Delta = t_rad - t_loss = beta ln(1/eps), beta = (1/c - 1/L)/eta > 0.
Control 'radial' (M*=sI) should give Delta ~ 0.
"""
import os, json, math
import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(__file__), 'results', 'exp_epsilon_sweep_lag.json')


def rf_of(M):
    m = M.shape[0]
    trM = np.trace(M); trM2 = np.trace(M @ M); fro = (M ** 2).sum()
    return (trM**2 + trM2 + fro) / ((m + 2) * fro) if fro > 1e-30 else 0.0


def make_teacher(m, kind, L=4.0, c=0.5, p=None, seed=0):
    rng = np.random.default_rng(seed)
    if kind == 'radial':
        return c * np.eye(m)                      # M*=cI control (degenerate)
    p = p if p is not None else m // 2
    B = rng.standard_normal((p, p))
    B = B / np.linalg.norm(B, 2) * L              # spectral norm L, trace ~ 0
    Ms = np.zeros((m, m)); Ms[:p, :p] = B; Ms[p:, p:] = c * np.eye(m - p)
    return Ms


def train(m, k, eps, Mstar, eta=2e-3, steps=40000, seed=0, sample=2):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((k, m)) * eps
    W2 = rng.standard_normal((m, k)) * eps
    L0 = 0.5 * ((W2 @ W1 - Mstar) ** 2).sum()
    floor = 1.0 / m
    losses, rfs, steps_rec = [], [], []
    for t in range(steps):
        M = W2 @ W1
        G = M - Mstar
        W1 -= eta * (W2.T @ G); W2 -= eta * (G @ W1.T)
        if t % sample == 0:
            losses.append(0.5 * (G ** 2).sum()); rfs.append(rf_of(M)); steps_rec.append(t)
    M = W2 @ W1
    return np.array(steps_rec), np.array(losses), np.array(rfs), L0, rf_of(M)


def t_loss_of(steps, losses, L0, frac=0.99):
    Linf = losses[-1]
    target = L0 - frac * (L0 - Linf)
    idx = np.where(losses <= target)[0]
    return steps[idx[0]] if len(idx) else steps[-1]


def t_rad_of(steps, rfs, floor, frac=0.5):
    rinf = rfs[-1]
    if rinf - floor < 1e-6:
        return steps[-1]
    target = floor + frac * (rinf - floor)
    idx = np.where(rfs >= target)[0]
    return steps[idx[0]] if len(idx) else steps[-1]


def run(kind, m=64, k=64, eta=2e-3, L=4.0, c=0.5, seeds=5):
    floor = 1.0 / m
    Mstar = make_teacher(m, kind, L=L, c=c, seed=0)
    print(f"\n=== teacher='{kind}'  R_F(M*)={rf_of(Mstar):.3f}  "
          f"tr(M*)/||M*||={np.trace(Mstar)/np.linalg.norm(Mstar):.2f} ===")
    epss = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
    rows = []
    print(f"  {'eps':>9}{'ln(1/eps)':>10}{'t_loss':>9}{'t_rad':>9}{'Delta':>9}")
    for eps in epss:
        tl, tr, dl = [], [], []
        for sd in range(seeds):
            st, ls, rf, L0, rff = train(m, k, eps, Mstar, eta=eta, seed=sd)
            a = t_loss_of(st, ls, L0); b = t_rad_of(st, rf, floor)
            tl.append(a); tr.append(b); dl.append(b - a)
        rows.append({'eps': eps, 'ln_inv': math.log(1/eps),
                     't_loss': float(np.mean(tl)), 't_rad': float(np.mean(tr)),
                     'delta': float(np.mean(dl)), 'delta_se': float(np.std(dl)/np.sqrt(seeds))})
        print(f"  {eps:>9.4f}{math.log(1/eps):>10.2f}{np.mean(tl):>9.0f}{np.mean(tr):>9.0f}{np.mean(dl):>9.0f}")
    x = np.array([r['ln_inv'] for r in rows]); d = np.array([r['delta'] for r in rows])
    tl = np.array([r['t_loss'] for r in rows]); trd = np.array([r['t_rad'] for r in rows])
    beta, b0 = np.polyfit(x, d, 1); r, p = stats.pearsonr(x, d)
    sl_loss = np.polyfit(x, tl, 1)[0]; sl_rad = np.polyfit(x, trd, 1)[0]
    print(f"  Delta = {beta:.0f}*ln(1/eps) + {b0:.0f}   (Pearson r={r:.3f}, p={p:.1e})")
    print(f"  slope t_rad={sl_rad:.0f}  slope t_loss={sl_loss:.0f}  (theory 1/(eta c)={1/(eta*c):.0f}, 1/(eta L)={1/(eta*L):.0f})")
    print(f"  counterfactual: Delta(max eps)={d[-1]:.0f} vs Delta(min eps)={d[0]:.0f}")
    return {'kind': kind, 'rf_Mstar': rf_of(Mstar), 'rows': rows,
            'beta': float(beta), 'pearson': float(r), 'p': float(p),
            'slope_t_rad': float(sl_rad), 'slope_t_loss': float(sl_loss),
            'theory_slope_rad': 1/(eta*c), 'theory_slope_loss': 1/(eta*L)}


def sweep(kind, m=64, k=64, eta=2e-3, L=4.0, c=0.5, seeds=5,
          loss_fracs=(0.80, 0.90, 0.95, 0.99)):
    """Full sweep: cache trajectories, report Delta vs ln(1/eps) at several loss thresholds."""
    floor = 1.0 / m
    Mstar = make_teacher(m, kind, L=L, c=c, seed=0)
    rfMs = rf_of(Mstar)
    epss = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
    traj = {(eps, sd): train(m, k, eps, Mstar, eta=eta, seed=sd)
            for eps in epss for sd in range(seeds)}
    print(f"\n=== teacher='{kind}'  R_F(M*)={rfMs:.3f} ===")
    out = {'kind': kind, 'rf_Mstar': rfMs, 'L': L, 'c': c, 'eta': eta,
           'ln_inv': [math.log(1/e) for e in epss], 'thresholds': {}}
    for fl in loss_fracs:
        x, D, TL, TR = [], [], [], []
        for eps in epss:
            tl = [t_loss_of(*traj[(eps, sd)][:2], traj[(eps, sd)][3], frac=fl) for sd in range(seeds)]
            tr = [t_rad_of(traj[(eps, sd)][0], traj[(eps, sd)][2], floor, frac=0.5) for sd in range(seeds)]
            x.append(math.log(1/eps)); D.append(float(np.mean(np.array(tr)-np.array(tl))))
            TL.append(float(np.mean(tl))); TR.append(float(np.mean(tr)))
        x = np.array(x)
        beta, b0 = np.polyfit(x, D, 1); r, p = stats.pearsonr(x, D)
        out['thresholds'][f'{fl}'] = {'delta': D, 't_loss': TL, 't_rad': TR,
                                      'beta': float(beta), 'r': float(r), 'p': float(p),
                                      'slope_t_loss': float(np.polyfit(x, TL, 1)[0]),
                                      'slope_t_rad': float(np.polyfit(x, TR, 1)[0])}
        print(f"  loss_thr={fl:.2f}: Delta={beta:>6.0f}*ln(1/eps)+{b0:.0f}  r={r:+.3f}  "
              f"Delta[bigEps..smallEps]={D[-1]:.0f}..{D[0]:.0f}")
    return out


def main():
    res = {}
    res['split'] = sweep('split', L=4.0, c=0.5)   # non-radial-dominant -> lag at bulk thresholds
    res['radial'] = sweep('radial', c=4.0)        # M*=sI control -> no lag (R_F leads)
    json.dump(res, open(OUT, 'w'), indent=2)
    print(f"\nSaved {OUT}")


if __name__ == '__main__':
    main()
