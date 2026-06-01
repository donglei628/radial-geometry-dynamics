"""
Saturation value of R_F: closing the trajectory (floor -> ignite -> saturation).

CLEAN result (symmetric-PSD / diagonal converged operator): as the modes converge,
  R_F -> R_F_sat = (1 + a2_sat + b_sat)/(m+2),  a2_sat = (sum s)^2 / (sum s^2)  [participation ratio]
For a positive diagonal teacher b_sat = 1, so  R_F_sat = (2 + PR)/(m+2),  PR = participation ratio.
  - all-equal spectrum (rank m): PR=m  -> R_F_sat=1   (recovers the radial-teacher limit)
  - top-k flat, rest 0:          PR=k  -> R_F_sat=(2+k)/(m+2)   (k=m/2 -> ~0.5)
This pins the END of the trajectory as a spectral functional (synthetic verification here;
the real-network caveat -- non-normality -- is handled separately).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)


def rf_of(M):
    m = M.shape[0]
    trM = np.trace(M); trM2 = np.trace(M @ M); fro = (M ** 2).sum()
    return (trM ** 2 + trM2 + fro) / ((m + 2) * fro) if fro > 1e-30 else 0.0


def participation(s):
    s = np.asarray(s, float)
    return (s.sum() ** 2) / (s ** 2).sum()


def teacher(m, kind, k=None, s=4.0, seed=0):
    rng = np.random.default_rng(seed)
    d = np.zeros(m)
    if kind == 'equal':
        d[:] = s
    elif kind == 'topk':
        d[:k] = s
    elif kind == 'powerlaw':
        d = s * (np.arange(1, m + 1) ** -1.0)   # 1/i decay
    elif kind == 'gauss':
        d = np.abs(rng.normal(1.0, 1.0, m)) * s
    return np.diag(d), d


def train(Mstar, m=64, k=64, eps=0.01, eta=2e-3, steps=30000, seed=0, sample=50):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((k, m)) * eps
    W2 = rng.standard_normal((m, k)) * eps
    traj_t, traj_rf = [], []
    for t in range(steps):
        M = W2 @ W1
        G = M - Mstar
        W1 -= eta * (W2.T @ G); W2 -= eta * (G @ W1.T)
        if t % sample == 0:
            traj_t.append(t); traj_rf.append(rf_of(M))
    return np.array(traj_t), np.array(traj_rf), rf_of(W2 @ W1)


def main():
    m = 64
    floor = 1.0 / m
    specs = [('equal', None, 'all-equal (rank m)'),
             ('topk', m // 2, 'top-m/2 flat'),
             ('topk', m // 4, 'top-m/4 flat'),
             ('topk', m // 8, 'top-m/8 flat'),
             ('powerlaw', None, 'power-law 1/i'),
             ('gauss', None, '|N(1,1)| spectrum')]
    print(f"m={m}, floor 1/m={floor:.4f}")
    print(f"{'spectrum':<20}{'PR':>8}{'R_F_sat pred':>14}{'R_F_sat sim':>13}{'rel err':>9}")
    rows = []
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(specs)))
    preds, sims = [], []
    for (kind, k, lab), col in zip(specs, colors):
        Mstar, d = teacher(m, kind, k=k)
        PR = participation(d)
        pred = rf_of(Mstar)                      # = (1+a2+b)/(m+2) of the teacher (exact)
        pred_pr = (2 + PR) / (m + 2)             # simplified b=1 form
        tt, rf, sim = train(Mstar)
        rel = abs(sim - pred) / pred
        rows.append({'spectrum': lab, 'PR': float(PR), 'pred': float(pred),
                     'pred_PR_b1': float(pred_pr), 'sim': float(sim), 'rel_err': float(rel)})
        preds.append(pred); sims.append(sim)
        print(f"{lab:<20}{PR:>8.1f}{pred:>14.4f}{sim:>13.4f}{rel:>9.4f}")
        ax[0].plot(np.maximum(tt, 1), rf, '-', color=col, lw=1.5, label=lab)
        ax[0].axhline(pred, color=col, ls=':', lw=0.8)
    ax[0].axhline(floor, color='k', ls='--', lw=1, label='floor $1/m$')
    ax[0].set_xscale('log'); ax[0].set_xlabel('gradient-flow step'); ax[0].set_ylabel(r'$R_F$')
    ax[0].set_title('(a) Trajectory saturates at the predicted value')
    ax[0].legend(fontsize=8.5, loc='upper left')
    # (b) predicted vs simulated saturation
    preds, sims = np.array(preds), np.array(sims)
    ax[1].plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax[1].scatter(preds, sims, c=colors, s=60, zorder=3)
    ax[1].set_xlabel(r'predicted $R_F^{\rm sat}=(2+\mathrm{PR})/(m+2)$')
    ax[1].set_ylabel(r'simulated $R_F^{\rm sat}$')
    ax[1].set_title('(b) Saturation = spectral participation ratio')
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig_saturation.pdf')
    plt.savefig(out, bbox_inches='tight'); plt.savefig(out.replace('.pdf', '.png'), dpi=130, bbox_inches='tight')
    print(f"\nmax rel err = {max(r['rel_err'] for r in rows):.4f}")
    json.dump({'m': m, 'rows': rows}, open(os.path.join(HERE, 'exp_saturation.json'), 'w'), indent=2)
    print('saved', out)


if __name__ == '__main__':
    main()
