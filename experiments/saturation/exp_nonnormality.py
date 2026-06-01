"""
Empirical charting of non-normality (measure the wall, don't climb it).

Per layer M = W_down @ W_up, compute eigenvalues {lambda} and singular values {sigma}:
  Henrici departure (Frobenius), normalized:  nu_hat = sqrt(||M||_F^2 - sum|lambda|^2) / ||M||_F  in [0,1)
    nu_hat=0 <=> normal; nu_hat->1 <=> strongly non-normal.
  R_F (measured), a^2 = tr(M)^2/||M||_F^2, PR = (sum sigma)^2/(sum sigma^2),
  R_F_ideal(PR) = (2+PR)/(m+2)  [value if M were symmetric-PSD with these singular values]
  gap = R_F_ideal - R_F_measured.
Hypothesis (insight): nu_hat explains the gap (rho(nu_hat, gap) > 0).
"""
import os, sys, json
import numpy as np
import torch
from scipy import stats

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), '..', 'experiments'))
EXP = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'experiments')
sys.path.insert(0, EXP)
from exp_batch_free_metrics import MODEL_CONFIGS, get_layers, extract_mlp_weights


@torch.no_grad()
def run_model(key, device='cpu'):
    from transformers import AutoModelForCausalLM
    cfg = MODEL_CONFIGS[key]
    print(f"\n# {key}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(cfg['name'], trust_remote_code=True,
                                                 dtype=torch.float32)
    layers = get_layers(model, cfg['type'])
    rows = []
    for li, layer in enumerate(layers):
        W1, W2, _ = extract_mlp_weights(layer, cfg['type'])   # W1=up, W2=down
        M = (W2 @ W1).cpu().numpy().astype(np.float64)
        m = M.shape[0]
        fro2 = float((M ** 2).sum())
        sv = np.linalg.svd(M, compute_uv=False)
        sum_sv = float(sv.sum()); sum_sv2 = float((sv ** 2).sum())
        ev = np.linalg.eigvals(M)
        sum_abs_lam2 = float((np.abs(ev) ** 2).sum())
        nu_hat = float(np.sqrt(max(fro2 - sum_abs_lam2, 0.0)) / np.sqrt(fro2))
        trM = float(np.trace(M)); trM2 = float(np.trace(M @ M))
        a2 = trM ** 2 / fro2; b = trM2 / fro2
        rf = (1 + a2 + b) / (m + 2)
        PR = sum_sv ** 2 / sum_sv2
        rf_ideal = (2 + PR) / (m + 2)
        rows.append({'layer': li, 'pos': li / (len(layers) - 1) if len(layers) > 1 else 0.0,
                     'm': m, 'nu_hat': nu_hat, 'rf': rf, 'a2': a2, 'b': b,
                     'PR': PR, 'PR_over_m': PR / m, 'rf_ideal': rf_ideal,
                     'gap': rf_ideal - rf, 'tr_M': trM})
        print(f"  L{li:2d} nu_hat={nu_hat:.3f} R_F={rf:.3f} PR/m={PR/m:.3f} gap={rf_ideal-rf:.3f}", flush=True)
    del model
    import gc; gc.collect()
    return rows


def main():
    keys = [a for a in sys.argv[1:] if a in MODEL_CONFIGS] or list(MODEL_CONFIGS)
    fp = os.path.join(HERE, 'exp_nonnormality.json')
    res = json.load(open(fp)) if os.path.exists(fp) else {}
    for k in keys:
        try:
            res[k] = run_model(k)
            json.dump(res, open(fp, 'w'), indent=2)
        except Exception as e:
            print(f"  ERROR {k}: {e}")
            import traceback; traceback.print_exc()

    # ---- gating analysis: rho(nu_hat, gap) ----
    print("\n" + "=" * 60)
    print("GATING TEST (3.1-A): does nu_hat explain the gap?")
    print(f"{'model':<13}{'rho(nu,gap)':>12}{'p':>10}{'mean nu':>9}{'rho(nu,depth)':>14}")
    allnu, allgap = [], []
    for k in keys:
        if k not in res: continue
        nu = np.array([r['nu_hat'] for r in res[k]])
        gap = np.array([r['gap'] for r in res[k]])
        pos = np.array([r['pos'] for r in res[k]])
        allnu += list(nu); allgap += list(gap)
        rho, p = stats.spearmanr(nu, gap)
        rd, _ = stats.spearmanr(pos, nu)
        print(f"{k:<13}{rho:>12.3f}{p:>10.1e}{nu.mean():>9.3f}{rd:>14.3f}")
    rho, p = stats.spearmanr(allnu, allgap)
    print(f"{'POOLED':<13}{rho:>12.3f}{p:>10.1e}{np.mean(allnu):>9.3f}")
    json.dump(res, open(fp, 'w'), indent=2)
    print(f"\nSaved {fp}")


if __name__ == '__main__':
    main()
