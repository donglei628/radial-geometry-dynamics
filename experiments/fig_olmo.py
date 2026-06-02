"""Figure: OLMo-1B (SwiGLU) replicates the R_F floor + ignition + differentiation."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), 'results')
FIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
d = json.load(open(os.path.join(R, 'exp_olmo_dynamics.json')))
ck = d['ckpts']; steps = sorted(int(s) for s in ck)
std = [ck[str(s)]['rf_std'] for s in steps]
mean = [ck[str(s)]['rf_mean'] for s in steps]
ppl = [ck[str(s)]['val_ppl'] for s in steps]
floor = d['floor']['rf_mean']; m = d['floor']['m']

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))

# (a) R_F mean & std vs step, with floor (re-init) and the trajectory (public ckpts >= step 1000)
ax[0].axhline(1.0/m, color='k', ls='--', lw=1, label=f'theory floor $1/m={1/m:.2e}$')
ax[0].scatter([300], [d['floor']['rf_mean']], color='k', marker='*', s=110, zorder=5,
              label=f're-init architecture = {d["floor"]["rf_mean"]:.2e}')
ax[0].plot(steps, mean, 'o-', color='#C44E52', ms=4, label=r'$R_F$ mean (ckpts $\geq$ 1k)')
ax[0].plot(steps, std, 's-', color='#8172B3', ms=4, label=r'$R_F$ across-layer std')
ax[0].axvspan(20000, 740000, color='gray', alpha=0.08)
ax[0].annotate('non-monotone\nlate phase', (1.2e5, 0.05), fontsize=8, color='gray')
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('training step'); ax[0].set_ylabel(r'$R_F$')
ax[0].set_title('(a) Floor (re-init) + post-ignition rise', fontsize=11)
ax[0].legend(fontsize=9, loc='lower right')

# (b) R_F std vs PPL: geometry forms as loss falls
ax[1].plot(ppl, std, 'o-', color='#55A868', ms=5)
for i, s in enumerate(steps):
    if s in (1000, 5000, 20000, 738000):
        ax[1].annotate(f'step {s}', (ppl[i], std[i]), fontsize=8,
                       textcoords='offset points', xytext=(4, 3))
ax[1].set_xlabel('validation PPL'); ax[1].set_ylabel(r'$R_F$ across-layer std')
ax[1].set_title('(b) Bulk differentiation; non-monotone late', fontsize=11)
ax[1].invert_xaxis(); ax[1].grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(FIG, 'fig_olmo.pdf')
plt.savefig(out, bbox_inches='tight'); plt.savefig(out.replace('.pdf', '.png'), dpi=130, bbox_inches='tight')
print('saved', out)
print(f'amplification max_std/floor_std = {max(std)/d["floor"]["rf_std"]:.0f}x')
