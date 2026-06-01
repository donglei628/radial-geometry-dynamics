"""
§2.2 Training-step: does trace-coherence tau BUILD during R_F ignition (rank PR stable)?
Pythia-160m checkpoints. Per layer M=W_down@W_up: tau=tr(M)/||M||_nuc, PR/m, R_F.
Track mean|tau|, mean PR/m, R_F std vs step. Hypothesis: ignition = coherence-building.
"""
import os, json
import numpy as np
import torch

HERE = os.path.dirname(__file__)
REPO = 'EleutherAI/pythia-160m'
STEPS = [0, 1, 64, 512, 1000, 2000, 4000, 8000, 16000, 64000, 143000]
OUT = os.path.join(HERE, 'exp_nonnormality_training.json')


@torch.no_grad()
def metrics_at(step):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=f'step{step}',
                                                 dtype=torch.float32)
    taus, prs, rfs = [], [], []
    for layer in model.gpt_neox.layers:
        W1 = layer.mlp.dense_h_to_4h.weight.detach().float()
        W2 = layer.mlp.dense_4h_to_h.weight.detach().float()
        M = (W2 @ W1).cpu().numpy().astype(np.float64)
        m = M.shape[0]
        fro2 = (M ** 2).sum()
        sv = np.linalg.svd(M, compute_uv=False)
        nuc = sv.sum(); PR = sv.sum() ** 2 / (sv ** 2).sum()
        trM = np.trace(M); trM2 = np.trace(M @ M)
        a2 = trM ** 2 / fro2; b = trM2 / fro2
        rf = (1 + a2 + b) / (m + 2)
        tau = trM / nuc
        taus.append(abs(tau)); prs.append(PR / m); rfs.append(rf)
    del model
    import gc; gc.collect()
    return {'mean_abs_tau': float(np.mean(taus)), 'mean_PR_over_m': float(np.mean(prs)),
            'rf_mean': float(np.mean(rfs)), 'rf_std': float(np.std(rfs))}


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for s in STEPS:
        if str(s) in res:
            continue
        try:
            res[str(s)] = metrics_at(s)
            r = res[str(s)]
            print(f"step {s:>7}: |tau|={r['mean_abs_tau']:.3f} PR/m={r['mean_PR_over_m']:.3f} "
                  f"R_F mean={r['rf_mean']:.3f} std={r['rf_std']:.3f}", flush=True)
            json.dump(res, open(OUT, 'w'), indent=2)
        except Exception as e:
            print(f"step {s} ERROR: {str(e)[:100]}", flush=True)
    print("\nDoes coherence build during ignition?")
    ss = sorted(int(k) for k in res)
    tau0 = res[str(ss[0])]['mean_abs_tau']; tauF = res[str(ss[-1])]['mean_abs_tau']
    pr0 = res[str(ss[0])]['mean_PR_over_m']; prF = res[str(ss[-1])]['mean_PR_over_m']
    print(f"  |tau|:  init {tau0:.3f} -> final {tauF:.3f}  ({tauF/tau0:.1f}x)")
    print(f"  PR/m:   init {pr0:.3f} -> final {prF:.3f}  ({prF/pr0:.2f}x)")
    print("  => if |tau| rises sharply while PR/m ~stable: ignition = coherence-building, not rank-change.")


if __name__ == '__main__':
    main()
