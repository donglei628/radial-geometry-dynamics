"""Figure for §4: R_F training dynamics on Pythia (4 sizes)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), 'results')
FIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(FIG, exist_ok=True)

d = json.load(open(os.path.join(R, 'exp_batch_training_dynamics.json')))
loss = json.load(open(os.path.join(R, 'exp_4_05_loss.json')))
A = d['analysis']
MODELS = ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b']
LAB = {'pythia-70m': '70M (m=512)', 'pythia-160m': '160M (m=768)',
       'pythia-410m': '410M (m=1024)', 'pythia-1b': '1B (m=2048)'}
COL = {'pythia-70m': '#4C72B0', 'pythia-160m': '#55A868',
       'pythia-410m': '#C44E52', 'pythia-1b': '#8172B3'}
steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000,
         16000, 32000, 64000, 128000, 143000]

plt.rcParams.update({'font.size': 13, 'axes.grid': True, 'grid.alpha': 0.3})
fig, ax = plt.subplots(2, 2, figsize=(11, 8)); ax = ax.ravel()

# (a) ignition curves: R_F std vs step
for mk in MODELS:
    ck = d[mk]['checkpoints']
    xs, ys = [], []
    for s in steps:
        if str(s) in ck:
            xs.append(max(s, 0.5)); ys.append(np.std(ck[str(s)]['rf_per_layer']))
    ax[0].plot(xs, ys, '-o', ms=3, color=COL[mk], label=LAB[mk])
ax[0].set_xscale('log'); ax[0].set_xlabel('training step'); ax[0].set_ylabel(r'$R_F$ across-layer std')
ax[0].set_title('(a) Ignition: frozen $\\to$ ignite $\\to$ saturate')
ax[0].axvspan(0.5, 64, color='gray', alpha=0.12)
ax[0].axvspan(1000, 4000, color='orange', alpha=0.12)
ax[0].legend(fontsize=10, loc='upper left')

# (b) init R_F = 1/m (floor), 4 sizes
ms = [d[mk]['m'] for mk in MODELS]
init = [A[mk]['init_rf_mean'] for mk in MODELS]
inv = [1.0 / m for m in ms]
ax[1].plot(ms, init, 'o', ms=8, color='#C44E52', label='measured init $R_F$', zorder=3)
mm = np.linspace(min(ms) * 0.9, max(ms) * 1.05, 100)
ax[1].plot(mm, 1 / mm, 'k--', lw=1, label=r'theory $1/m$')
ax[1].set_xscale('log'); ax[1].set_yscale('log')
ax[1].set_xlabel('width $m$'); ax[1].set_ylabel(r'init $R_F$ (layer mean)')
ax[1].set_title('(b) Floor: init $R_F=1/m$ (4 s.f.)')
for mk, m, v in zip(MODELS, ms, init):
    ax[1].annotate(f'std={A[mk]["init_rf_std"]:.0e}', (m, v), fontsize=9,
                   textcoords='offset points', xytext=(4, -8))
ax[1].legend(fontsize=10)

# (c) geometry lags loss (use the loss file, single model)
rows = loss['rows']
st = np.array([r['step'] for r in rows], float)
ppl = np.array([r['ppl'] for r in rows]); std = np.array([r['rf_std'] for r in rows])
ppl_prog = (ppl[0] - ppl) / (ppl[0] - ppl[-1])
std_prog = (std - std[0]) / (std[-1] - std[0])
xm = np.maximum(st, 0.5)
ax[2].plot(xm, 100 * ppl_prog, '-s', ms=3, color='#333', label='loss drop (% done)')
ax[2].plot(xm, 100 * std_prog, '-o', ms=3, color='#C44E52', label='$R_F$ divergence (% done)')
ax[2].axvline(512, color='#333', ls=':', lw=1); ax[2].axvline(4000, color='#C44E52', ls=':', lw=1)
ax[2].set_xscale('log'); ax[2].set_xlabel('training step'); ax[2].set_ylabel('% of total change')
ax[2].set_title('(c) Geometry lags loss ($\\rho=0.98$)')
ax[2].annotate('loss 99%\nby step 512', (512, 50), fontsize=9, color='#333',
               textcoords='offset points', xytext=(6, -2))
ax[2].annotate('$R_F$ 50%\nby step 4000', (4000, 50), fontsize=9, color='#C44E52',
               textcoords='offset points', xytext=(6, -20))
ax[2].legend(fontsize=10, loc='center left')

# (d) scaling: amplification and position-structure vs size
amp = [A[mk]['divergence_amplification'] for mk in MODELS]
rho = [A[mk]['rho_pos_convstep'] for mk in MODELS]
axb = ax[3]; axb2 = axb.twinx()
axb.bar(range(4), amp, color='#8172B3', alpha=0.6, width=0.5)
axb.set_yscale('log'); axb.set_ylabel('divergence amplification', color='#8172B3')
axb2.plot(range(4), rho, '-o', color='#C44E52', ms=6)
axb2.set_ylabel(r'$\rho$(layer depth, conv. step)', color='#C44E52')
axb2.set_ylim(-0.4, 0.8); axb2.axhline(0, color='gray', lw=0.5, ls=':')
axb.set_xticks(range(4)); axb.set_xticklabels(['70M', '160M', '410M', '1B'], fontsize=11)
axb.set_title('(d) Both strengthen with scale')
axb.grid(False); axb2.grid(False)

plt.tight_layout()
out = os.path.join(FIG, 'fig_training_dynamics.pdf')
plt.savefig(out, bbox_inches='tight'); plt.savefig(out.replace('.pdf', '.png'), dpi=130, bbox_inches='tight')
print('saved', out)
