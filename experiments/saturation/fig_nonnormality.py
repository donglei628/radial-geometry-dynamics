"""Figure: R_F is governed by trace-coherence (not rank). R_F = PR * tau^2 / m."""
import os, json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
d = json.load(open(os.path.join(HERE, 'exp_nonnormality.json')))
models = ['gpt2', 'pythia-410m', 'pythia-1b', 'tinyllama', 'qwen2.5-0.5b', 'qwen2.5-1.5b', 'phi-2']
COL = dict(zip(models, plt.cm.tab10(np.linspace(0, 1, len(models)))))


def derive(R):
    m = R[0]['m']
    rf = np.array([r['rf'] for r in R]); PR = np.array([r['PR'] for r in R])
    a2 = np.array([r['a2'] for r in R]); trM = np.array([r['tr_M'] for r in R])
    fro2 = trM**2 / np.maximum(a2, 1e-12); nuc = np.sqrt(PR * fro2); tau = trM / nuc
    pos = np.array([r['pos'] for r in R])
    return m, rf, PR / m, np.abs(tau), pos


plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 2, figsize=(11, 3.7))

# (a) R_F vs |tau| (coherence) -- the dominant driver
allt, allrf, allpr = [], [], []
for k in models:
    m, rf, prm, tau, pos = derive(d[k])
    ax[0].scatter(tau, rf, s=22, color=COL[k], alpha=0.8, label=k)
    allt += list(tau); allrf += list(rf); allpr += list(prm)
allt, allrf, allpr = np.array(allt), np.array(allrf), np.array(allpr)
# overlay R_F = tau^2 * mean(PR/m) guide
xs = np.linspace(0, max(allt), 100)
ax[0].plot(xs, xs**2 * np.median(allpr), 'k--', lw=1,
           label=r'$R_F=\tau^2\cdot\overline{\mathrm{PR}/m}$')
ax[0].set_xlabel(r'trace coherence $|\tau|=|\mathrm{tr}M|/\|M\|_*$')
ax[0].set_ylabel(r'$R_F$')
ax[0].set_title(f'(a) Coherence drives $R_F$  ($\\rho={stats.spearmanr(allt,allrf)[0]:.2f}$)')
ax[0].legend(fontsize=8.5, loc='upper left', ncol=2)

# (b) rank PR/m is high & stable; R_F variation is NOT rank
ax[1].scatter(allpr, allrf, s=22, c='#888', alpha=0.7)
ax[1].set_xlabel(r'rank participation $\mathrm{PR}/m$')
ax[1].set_ylabel(r'$R_F$')
ax[1].set_title(f'(b) Rank does not explain $R_F$  ($\\rho={stats.spearmanr(allpr,allrf)[0]:.2f}$)')
ax[1].set_xlim(0, 1)
ax[1].grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(HERE, 'fig_nonnormality.pdf')
plt.savefig(out, bbox_inches='tight'); plt.savefig(out.replace('.pdf', '.png'), dpi=130, bbox_inches='tight')
print('saved', out)
print(f'pooled rho(|tau|,R_F)={stats.spearmanr(allt,allrf)[0]:.3f}  rho(PR/m,R_F)={stats.spearmanr(allpr,allrf)[0]:.3f}')
print(f'PR/m range: {allpr.min():.2f}-{allpr.max():.2f} (rank high & stable); |tau| range: {allt.min():.2f}-{allt.max():.2f}')
