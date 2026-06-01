"""Figure: R_F ignition is coherence-building, not rank-change (Pythia-160m)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
d = json.load(open(os.path.join(HERE, 'exp_nonnormality_training.json')))
steps = sorted(int(k) for k in d)
tau = [d[str(s)]['mean_abs_tau'] for s in steps]
pr = [d[str(s)]['mean_PR_over_m'] for s in steps]
rfm = [d[str(s)]['rf_mean'] for s in steps]
x = [max(s, 0.5) for s in steps]

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))

# (a) coherence builds, rank flat
ax[0].plot(x, tau, 'o-', color='#C44E52', ms=5, label=r'coherence $|\tau|=|\mathrm{tr}M|/\|M\|_*$')
ax[0].plot(x, pr, 's-', color='#4C72B0', ms=5, label=r'rank $\mathrm{PR}/m$')
ax[0].plot(x, rfm, '^-', color='#55A868', ms=5, label=r'$R_F$ mean')
ax[0].axvspan(1000, 8000, color='orange', alpha=0.12)
ax[0].set_xscale('log'); ax[0].set_xlabel('training step'); ax[0].set_ylabel('value')
ax[0].set_title('(a) Ignition builds coherence, not rank')
ax[0].legend(fontsize=10, loc='center left')

# (b) R_F vs tau^2 over training (R_F = PR * tau^2 / m, PR~const)
tau2 = np.array(tau) ** 2
ax[1].plot(tau2, rfm, 'o-', color='#8172B3', ms=5)
for i, s in enumerate(steps):
    if s in (512, 1000, 4000, 64000):
        ax[1].annotate(f'{s}', (tau2[i], rfm[i]), fontsize=8,
                       textcoords='offset points', xytext=(4, 2))
ax[1].set_xlabel(r'coherence$^2$  $\tau^2$')
ax[1].set_ylabel(r'$R_F$ mean')
ax[1].set_title(r'(b) $R_F\propto\tau^2$ along training (rank $\approx$ const)')
ax[1].grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(HERE, 'fig_nonnormality_training.pdf')
plt.savefig(out, bbox_inches='tight'); plt.savefig(out.replace('.pdf', '.png'), dpi=130, bbox_inches='tight')
print('saved', out)
