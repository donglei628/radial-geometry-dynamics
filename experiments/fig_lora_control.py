"""Figure for §5: LoRA layer-selection control (honest result)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), 'results')
FIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
d = json.load(open(os.path.join(R, 'exp_lora_control.json')))

conds = ['low_rf', 'random', 'first8', 'high_rf', 'last8']
lab = {'low_rf': 'low-$R_F$', 'random': 'random', 'first8': 'first-8',
       'high_rf': 'high-$R_F$', 'last8': 'last-8'}
vals = {c: np.array(d['ppls_' + c]) for c in conds}
order = sorted(conds, key=lambda c: vals[c].mean())

plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(figsize=(6.2, 3.4))
xs = np.arange(len(order))
means = [vals[c].mean() for c in order]
stds = [vals[c].std() for c in order]
colors = ['#C44E52' if c == 'low_rf' else ('#8172B3' if c == 'random' else '#bbbbbb')
          for c in order]
ax.bar(xs, means, yerr=stds, capsize=4, color=colors, alpha=0.75, zorder=2,
       error_kw={'lw': 1.2})
# overlay individual random draws
for c in order:
    if c == 'random':
        x = order.index(c)
        ax.scatter(np.full(len(vals[c]), x) + np.random.RandomState(0).uniform(-0.15, 0.15, len(vals[c])),
                   vals[c], color='#3b2d6b', s=18, zorder=3, label='random draws')
ax.text(0.98, 0.96, f"base (no fine-tuning): {d['base_val_ppl']:.2f}",
        transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#555')
ax.set_xticks(xs); ax.set_xticklabels([lab[c] for c in order])
ax.set_ylabel('validation PPL (lower better)')
ax.set_ylim(10.8, max(m + s for m, s in zip(means, stds)) + 0.15)
ax.set_title('LoRA layer selection (8 layers, rank 8, equal params)')
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.legend(fontsize=8, loc='upper left')
plt.tight_layout()
out = os.path.join(FIG, 'fig_lora_control.pdf')
plt.savefig(out, bbox_inches='tight'); plt.savefig(out.replace('.pdf', '.png'), dpi=130, bbox_inches='tight')
print('saved', out)
