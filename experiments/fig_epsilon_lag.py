"""Figure: loss-geometry lag grows as beta*ln(1/eps) (synthetic P4 causality)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), 'results')
FIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
d = json.load(open(os.path.join(R, 'exp_epsilon_sweep_lag.json')))

sp, ra = d['split'], d['radial']
x = np.array(sp['ln_inv'])
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))

# (a) Delta vs ln(1/eps): split (bulk thresholds) vs radial control
for thr, col, mk in [('0.8', '#C44E52', 'o'), ('0.9', '#dd8452', 's')]:
    t = sp['thresholds'][thr]
    ax[0].plot(x, t['delta'], mk + '-', color=col, ms=5,
               label=f"non-radial teacher, loss {int(float(thr)*100)}%  ($\\beta$={t['beta']:.0f}, r={t['r']:.3f})")
tc = ra['thresholds']['0.9']
ax[0].plot(x, tc['delta'], '^--', color='#4C72B0', ms=5,
           label=f"degenerate radial control $M^*\\!=\\!sI$: R$_F$ leads")
ax[0].axhline(0, color='gray', lw=0.6, ls=':')
ax[0].set_xlabel(r'$\ln(1/\varepsilon)$'); ax[0].set_ylabel(r'lag $\Delta=t_{\rm rad}-t_{\rm loss}$ (steps)')
ax[0].set_title('(a) Lag grows as $\\beta\\,\\ln(1/\\varepsilon)$, $\\beta>0$')
ax[0].legend(fontsize=9, loc='upper left'); ax[0].grid(alpha=0.3)

# (b) the two timescales (split, bulk 90%): diverging slopes
t = sp['thresholds']['0.9']
ax[1].plot(x, t['t_loss'], 's-', color='#55A868', ms=5,
           label=f"$t_{{\\rm loss}}$ (bulk, non-radial)  slope {t['slope_t_loss']:.0f}")
ax[1].plot(x, t['t_rad'], 'o-', color='#C44E52', ms=5,
           label=f"$t_{{\\rm rad}}$ (radial ignition)  slope {t['slope_t_rad']:.0f}")
ax[1].set_xlabel(r'$\ln(1/\varepsilon)$'); ax[1].set_ylabel('step')
ax[1].set_title('(b) Radial mode has the slower $\\varepsilon$-scaling')
ax[1].legend(fontsize=9, loc='upper left'); ax[1].grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(FIG, 'fig_epsilon_lag.pdf')
plt.savefig(out, bbox_inches='tight'); plt.savefig(out.replace('.pdf', '.png'), dpi=130, bbox_inches='tight')
print('saved', out)
