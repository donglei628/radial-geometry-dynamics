"""
Task 2: R_F vs HT-SR (Heavy-Tailed Self-Regularization) head-to-head.

HT-SR metrics (Martin & Mahoney), computed faithfully without weightwatcher:
  - alpha       : power-law exponent of the ESD tail (eigenvalues of W^T W = sv^2),
                  MLE + KS-based x_min selection (Clauset et al. / weightwatcher).
  - stable_rank : ||W||_F^2 / sigma_max^2
  - matrix_entropy : normalized spectral entropy of singular values in [0,1]
Computed per layer on W_up, W_down, and the composite M = W_down @ W_up.

Question: is R_F (composite-operator radial scalar) orthogonal to these spectral
metrics, or a re-skin? And does R_F predict angular distance beyond HT-SR?

Usage: python exp_htsr_shootout.py            # all cached models
       python exp_htsr_shootout.py qwen2.5-0.5b
"""
import sys, os, json
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import ensure_output_dir
from exp_batch_free_metrics import (MODEL_CONFIGS, get_layers, extract_mlp_weights)

RESULT_FILE = 'exp_htsr_shootout.json'


def powerlaw_alpha(evals, max_xmin_candidates=150):
    """MLE power-law exponent with KS-minimizing x_min (Clauset/weightwatcher).
    evals: 1D positive array (eigenvalues = singular values squared)."""
    ev = np.sort(evals[evals > 1e-12].astype(np.float64))
    n = ev.size
    if n < 20:
        return float('nan')
    # candidate x_min: unique eigenvalues, leave >=10 in the tail; subsample for speed
    hi = n - 10
    cand_idx = np.unique(np.linspace(0, hi - 1, min(max_xmin_candidates, hi)).astype(int))
    best_alpha, best_D = float('nan'), np.inf
    for ci in cand_idx:
        xmin = ev[ci]
        tail = ev[ci:]
        nt = tail.size
        if nt < 10:
            continue
        s = np.log(tail / xmin).sum()
        if s <= 0:
            continue
        alpha = 1.0 + nt / s
        # KS distance between empirical and fitted CDF on the tail
        cdf_emp = np.arange(1, nt + 1) / nt
        cdf_fit = 1.0 - (tail / xmin) ** (-(alpha - 1.0))
        D = np.max(np.abs(cdf_emp - cdf_fit))
        if D < best_D:
            best_D, best_alpha = D, alpha
    return float(best_alpha)


def spec_metrics(W, device):
    """Return (alpha, stable_rank, matrix_entropy) for weight matrix W."""
    Wd = W.to(device).float()
    sv = torch.linalg.svdvals(Wd).cpu().numpy().astype(np.float64)
    sv = sv[sv > 1e-12]
    if sv.size < 2:
        return float('nan'), float('nan'), float('nan')
    evals = sv ** 2
    alpha = powerlaw_alpha(evals)
    stable_rank = float(evals.sum() / evals.max())
    p = evals / evals.sum()
    H = -(p * np.log(p)).sum()
    matrix_entropy = float(H / np.log(p.size))   # normalized to [0,1]
    return alpha, stable_rank, matrix_entropy


def rf_of(M):
    m = M.shape[0]
    tr_M = torch.trace(M).item()
    tr_M2 = torch.trace(M @ M).item()
    fro = (M ** 2).sum().item()
    return (tr_M**2 + fro + tr_M2) / ((m + 2) * fro) if fro > 0 else 0.0


def run_model(key, device):
    from transformers import AutoModelForCausalLM
    cfg = MODEL_CONFIGS[key]
    print(f"\n{'#'*64}\n# {key}\n{'#'*64}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg['name'], trust_remote_code=True, dtype=torch.float32)
    layers = get_layers(model, cfg['type'])
    rows = []
    for li, layer in enumerate(layers):
        W1, W2, _ = extract_mlp_weights(layer, cfg['type'])   # W1=up, W2=down
        M = (W2 @ W1)
        rf = rf_of(M)
        a_up, sr_up, me_up = spec_metrics(W1, device)
        a_dn, sr_dn, me_dn = spec_metrics(W2, device)
        a_M,  sr_M,  me_M  = spec_metrics(M,  device)
        rows.append(dict(layer=li, rf=rf,
                         alpha_up=a_up, alpha_dn=a_dn, alpha_M=a_M,
                         alpha_mean=float(np.nanmean([a_up, a_dn])),
                         sr_up=sr_up, sr_dn=sr_dn, sr_M=sr_M,
                         me_up=me_up, me_dn=me_dn, me_M=me_M))
        print(f"  L{li:2d} rf={rf:.3f} a_up={a_up:.2f} a_dn={a_dn:.2f} "
              f"a_M={a_M:.2f} sr_M={sr_M:.1f} me_M={me_M:.3f}", flush=True)
    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    return rows


def corr_report(rows):
    rf = np.array([r['rf'] for r in rows])
    out = {}
    for key in ['alpha_up', 'alpha_dn', 'alpha_M', 'alpha_mean',
                'sr_up', 'sr_dn', 'sr_M', 'me_up', 'me_dn', 'me_M']:
        v = np.array([r[key] for r in rows], dtype=float)
        mask = np.isfinite(v) & np.isfinite(rf)
        if mask.sum() >= 5:
            rho, p = stats.spearmanr(rf[mask], v[mask])
        else:
            rho, p = float('nan'), float('nan')
        out[key] = {'rho': float(rho), 'p': float(p)}
    return out


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    keys = [a for a in sys.argv[1:] if a in MODEL_CONFIGS] or list(MODEL_CONFIGS)
    fp = os.path.join(ensure_output_dir('results'), RESULT_FILE)
    results = {}
    if os.path.exists(fp):
        results = json.load(open(fp))
    for key in keys:
        try:
            rows = run_model(key, device)
            results[key] = {'layers': rows, 'rho_vs_rf': corr_report(rows)}
            json.dump(results, open(fp, 'w'), indent=2)
            print(f"  Spearman R_F vs:")
            for k, d in results[key]['rho_vs_rf'].items():
                print(f"    {k:<11} rho={d['rho']:+.3f} (p={d['p']:.1e})")
        except Exception as e:
            print(f"  ERROR {key}: {e}")
            import traceback; traceback.print_exc()
    print(f"\nSaved {fp}")


if __name__ == '__main__':
    main()
