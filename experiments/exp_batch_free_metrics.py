"""
Batch "free" experiments: pure weight-based metrics for all models.
No forward pass, no data loading - just matrix operations on weights.

Covers experiments:
  1.15  R_F before/after quantization
  2.01  SVD approximation error vs R_F
  2.04  Effective rank vs R_F (= 9.04)
  2.06  Rank-error curves at various k
  5.03  R_F vs layer depth position
  5.05  Gate projection R_F (SwiGLU only)
  5.06  Attention R_F (non-GQA only)
  9.01  R_F decomposition: tr(M)^2, tr(M^2), ||M||_F^2
  9.02  R_F vs condition number
  9.03  R_F vs spectral gap
  9.04  R_F vs effective rank
  9.05  R_F vs weight norms
  9.07  Random init R_F theoretical verification
  9.09  R_F bounds verification
  10.07 R_F vs weight kurtosis

Usage:
    python exp_batch_free_metrics.py          # all models
    python exp_batch_free_metrics.py gpt2     # single model
"""

import sys, os, json, time, math
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import ensure_output_dir


MODEL_CONFIGS = {
    'gpt2': {'name': 'gpt2', 'type': 'gpt2'},
    'pythia-410m': {'name': 'EleutherAI/pythia-410m', 'type': 'pythia'},
    'pythia-1b': {'name': 'EleutherAI/pythia-1b', 'type': 'pythia'},
    'tinyllama': {'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'type': 'swiglu'},
    'qwen2.5-0.5b': {'name': 'Qwen/Qwen2.5-0.5B', 'type': 'swiglu'},
    'qwen2.5-1.5b': {'name': 'Qwen/Qwen2.5-1.5B', 'type': 'swiglu'},
    'phi-2': {'name': 'microsoft/phi-2', 'type': 'phi2'},
}


# ======================================================================
# Quantization helpers
# ======================================================================

def ternary_quantize(W):
    row_mean = W.abs().mean(dim=1, keepdim=True)
    return torch.sign(W) * row_mean

def int4_rtn_quantize(W):
    row_max = W.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
    scale = row_max / 7.0
    W_q = torch.clamp(torch.round(W / scale), -8, 7)
    return W_q * scale


# ======================================================================
# Weight extraction
# ======================================================================

def get_layers(model, model_type):
    if model_type == 'gpt2':
        return model.transformer.h
    elif model_type == 'pythia':
        return model.gpt_neox.layers
    elif model_type in ('swiglu', 'phi2'):
        return model.model.layers

def extract_mlp_weights(layer, model_type):
    if model_type == 'gpt2':
        W1 = layer.mlp.c_fc.weight.detach().float().T
        W2 = layer.mlp.c_proj.weight.detach().float().T
        return W1, W2, None
    elif model_type == 'pythia':
        W1 = layer.mlp.dense_h_to_4h.weight.detach().float()
        W2 = layer.mlp.dense_4h_to_h.weight.detach().float()
        return W1, W2, None
    elif model_type == 'swiglu':
        W1 = layer.mlp.up_proj.weight.detach().float()
        W2 = layer.mlp.down_proj.weight.detach().float()
        W_gate = layer.mlp.gate_proj.weight.detach().float()
        return W1, W2, W_gate
    elif model_type == 'phi2':
        W1 = layer.mlp.fc1.weight.detach().float()
        W2 = layer.mlp.fc2.weight.detach().float()
        return W1, W2, None

def extract_attn_vo(layer, model_type):
    """Extract W_V and W_O for attention R_F. Returns None if GQA or unsupported."""
    try:
        if model_type == 'gpt2':
            W_qkv = layer.attn.c_attn.weight.detach().float().T  # [3*h, h]
            h = W_qkv.shape[1]
            W_V = W_qkv[2*h:3*h, :]  # [h, h]
            W_O = layer.attn.c_proj.weight.detach().float().T  # [h, h]
            return W_V, W_O
        elif model_type == 'pythia':
            W_qkv = layer.attention.query_key_value.weight.detach().float()
            h = W_qkv.shape[1]
            W_V = W_qkv[2*h:3*h, :]
            W_O = layer.attention.dense.weight.detach().float()
            return W_V, W_O
        elif model_type == 'swiglu':
            W_V = layer.self_attn.v_proj.weight.detach().float()
            W_O = layer.self_attn.o_proj.weight.detach().float()
            if W_V.shape[0] != W_O.shape[1]:
                return None, None
            return W_V, W_O
        elif model_type == 'phi2':
            if hasattr(layer.self_attn, 'v_proj'):
                W_V = layer.self_attn.v_proj.weight.detach().float()
                W_O = layer.self_attn.dense.weight.detach().float()
                if W_V.shape[0] != W_O.shape[1]:
                    return None, None
                return W_V, W_O
            return None, None
    except Exception:
        return None, None


# ======================================================================
# Core metric computation
# ======================================================================

def compute_rf(M, m):
    tr_M = torch.trace(M).item()
    tr_M2 = torch.trace(M @ M).item()
    fro_sq = (M ** 2).sum().item()
    denom = (m + 2) * fro_sq
    if denom < 1e-15:
        return 0.0, tr_M, tr_M2, fro_sq
    rf = (tr_M**2 + fro_sq + tr_M2) / denom
    return rf, tr_M, tr_M2, fro_sq


def compute_layer_metrics(M, m, W1, W2, W_gate, W_V, W_O, layer_idx, num_layers):
    """Compute all metrics for one layer."""
    r = {}
    r['layer_idx'] = layer_idx
    r['layer_pos'] = layer_idx / (num_layers - 1) if num_layers > 1 else 0.0

    # --- 9.01: R_F decomposition ---
    tr_M = torch.trace(M).item()
    tr_M2 = torch.trace(M @ M).item()
    fro_sq = (M ** 2).sum().item()
    denom = (m + 2) * fro_sq if fro_sq > 1e-15 else 1e-15

    rf = (tr_M**2 + fro_sq + tr_M2) / denom
    r['rf'] = rf
    r['tr_M'] = tr_M
    r['tr_M2'] = tr_M2
    r['fro_sq'] = fro_sq
    r['rf_comp_trM2'] = tr_M**2 / denom
    r['rf_comp_trM2M'] = tr_M2 / denom
    r['rf_comp_fro'] = fro_sq / denom   # always = 1/(m+2)
    r['tr_M_over_m'] = tr_M / m

    # --- SVD of M (feeds 9.02, 9.03, 9.04, 2.01, 2.06) ---
    S = torch.linalg.svdvals(M).numpy()

    # 9.02: condition number
    r['cond'] = float(S[0] / max(S[-1], 1e-12))
    r['log_cond'] = math.log10(r['cond']) if r['cond'] < 1e15 else 15.0

    # 9.03: spectral gap
    if len(S) >= 2:
        r['spec_gap'] = float(S[0] - S[1])
        r['spec_gap_ratio'] = float((S[0] - S[1]) / max(S[0], 1e-12))
    else:
        r['spec_gap'] = 0.0
        r['spec_gap_ratio'] = 0.0

    # 9.04 / 2.04: effective rank
    S_sum = S.sum()
    if S_sum > 1e-12:
        p = S / S_sum
        p = p[p > 1e-15]
        entropy = -(p * np.log(p)).sum()
        r['eff_rank'] = float(math.exp(entropy))
    else:
        r['eff_rank'] = 0.0

    # Stable rank = ||M||_F^2 / ||M||_2^2
    r['stable_rank'] = float(fro_sq / max(S[0]**2, 1e-12))

    # 9.05: norms
    r['M_fro'] = math.sqrt(fro_sq)
    r['M_spec'] = float(S[0])
    r['M_nuc'] = float(S.sum())
    r['W1_fro'] = float(W1.norm().item())
    r['W2_fro'] = float(W2.norm().item())

    # 10.07: kurtosis
    M_flat = M.flatten()
    mu = M_flat.mean()
    std = M_flat.std()
    if std > 1e-12:
        r['M_kurtosis'] = float(((M_flat - mu) / std).pow(4).mean().item() - 3)
    else:
        r['M_kurtosis'] = 0.0

    # 2.01 / 2.06: SVD approximation errors
    total_energy = (S ** 2).sum()
    svd_err = {}
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if k >= len(S):
            svd_err[str(k)] = 0.0
        else:
            retained = (S[:k] ** 2).sum()
            svd_err[str(k)] = float(1.0 - retained / total_energy)
    r['svd_errors'] = svd_err

    # Top-10 singular values (normalized)
    r['top10_sv'] = (S[:10] / max(S[0], 1e-12)).tolist()

    # --- 5.05: Gate R_F (SwiGLU) ---
    if W_gate is not None:
        M_gate = W2 @ W_gate
        rf_g, _, _, _ = compute_rf(M_gate, m)
        r['rf_gate'] = rf_g

    # --- 5.06: Attention R_F ---
    if W_V is not None and W_O is not None:
        M_attn = W_O @ W_V
        rf_a, _, _, _ = compute_rf(M_attn, m)
        r['rf_attn'] = rf_a

    # --- 1.15: R_F after quantization ---
    M_t = ternary_quantize(W2) @ ternary_quantize(W1)
    rf_t, _, _, _ = compute_rf(M_t, m)
    r['rf_ternary'] = rf_t
    r['rf_delta_ternary'] = rf_t - rf

    M_q4 = int4_rtn_quantize(W2) @ int4_rtn_quantize(W1)
    rf_q4, _, _, _ = compute_rf(M_q4, m)
    r['rf_int4'] = rf_q4
    r['rf_delta_int4'] = rf_q4 - rf

    return r


# ======================================================================
# Per-model runner
# ======================================================================

def run_model(model_key, cfg):
    from transformers import AutoModelForCausalLM

    print(f"\n{'='*70}")
    print(f"  {model_key} ({cfg['name']})")
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        cfg['name'], trust_remote_code=True,
        torch_dtype=torch.float16, device_map='cpu',
    )
    model.eval()
    load_t = time.time() - t0

    hid = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    if hasattr(model.config, 'intermediate_size'):
        m_ff = model.config.intermediate_size
    elif hasattr(model.config, 'n_inner') and model.config.n_inner:
        m_ff = model.config.n_inner
    else:
        m_ff = 4 * hid

    print(f"  Loaded in {load_t:.1f}s  m={hid}  m_ff={m_ff}  layers={n_layers}", flush=True)

    layers_mod = get_layers(model, cfg['type'])
    layer_results = []

    for i, layer in enumerate(layers_mod):
        W1, W2, W_gate = extract_mlp_weights(layer, cfg['type'])
        W_V, W_O = extract_attn_vo(layer, cfg['type'])
        M = W2 @ W1

        metrics = compute_layer_metrics(M, hid, W1, W2, W_gate, W_V, W_O, i, n_layers)
        layer_results.append(metrics)

        extra = ""
        if 'rf_gate' in metrics:
            extra += f" rf_gate={metrics['rf_gate']:.4f}"
        if 'rf_attn' in metrics:
            extra += f" rf_attn={metrics['rf_attn']:.4f}"
        print(f"  L{i:>2}  R_F={metrics['rf']:.4f}  eff_rank={metrics['eff_rank']:>6.1f}  "
              f"log_cond={metrics['log_cond']:>5.1f}  spec_gap_r={metrics['spec_gap_ratio']:.3f}  "
              f"kurtosis={metrics['M_kurtosis']:>7.1f}{extra}", flush=True)

    # --- 9.07: Random init verification ---
    print(f"\n  [9.07] Random init R_F verification...", flush=True)
    theoretical_rf = (hid + 3) / (hid * (hid + 2))
    baseline = 1.0 / (hid + 2)

    rand_rfs = []
    for _ in range(50):
        W1r = torch.randn(m_ff, hid) / math.sqrt(hid)
        W2r = torch.randn(hid, m_ff) / math.sqrt(m_ff)
        Mr = W2r @ W1r
        rfr, _, _, _ = compute_rf(Mr, hid)
        rand_rfs.append(rfr)
    rand_mean = np.mean(rand_rfs)
    rand_std = np.std(rand_rfs)
    print(f"  Theoretical ≈ {theoretical_rf:.8f}")
    print(f"  Empirical   = {rand_mean:.8f} ± {rand_std:.8f}")
    print(f"  Baseline 1/(m+2) = {baseline:.8f}")

    random_init = {
        'theoretical_rf': theoretical_rf,
        'empirical_mean': rand_mean,
        'empirical_std': rand_std,
        'baseline_1_over_m2': baseline,
        'm': hid, 'm_ff': m_ff, 'n_trials': 50,
    }

    # --- 9.09: Bounds ---
    rf_vals = [r['rf'] for r in layer_results]
    rf_min, rf_max = min(rf_vals), max(rf_vals)
    bounds = {
        'rf_min': rf_min, 'rf_max': rf_max,
        'lower_bound': baseline, 'upper_bound': 1.0,
        'min_ok': rf_min >= baseline - 1e-6,
        'max_ok': rf_max <= 1.0 + 1e-6,
    }
    print(f"\n  [9.09] R_F bounds: [{rf_min:.4f}, {rf_max:.4f}]  "
          f"theory: [{baseline:.6f}, 1.0]  OK={bounds['min_ok'] and bounds['max_ok']}")

    # --- Correlations ---
    print(f"\n  Cross-metric Spearman ρ with R_F:", flush=True)

    corr_targets = {
        'eff_rank':       [r['eff_rank'] for r in layer_results],
        'stable_rank':    [r['stable_rank'] for r in layer_results],
        'log_cond':       [r['log_cond'] for r in layer_results],
        'spec_gap_ratio': [r['spec_gap_ratio'] for r in layer_results],
        'M_fro':          [r['M_fro'] for r in layer_results],
        'M_spec':         [r['M_spec'] for r in layer_results],
        'M_kurtosis':     [r['M_kurtosis'] for r in layer_results],
        'W1_fro':         [r['W1_fro'] for r in layer_results],
        'W2_fro':         [r['W2_fro'] for r in layer_results],
        'rf_comp_trM2':   [r['rf_comp_trM2'] for r in layer_results],
        'rf_comp_trM2M':  [r['rf_comp_trM2M'] for r in layer_results],
        'tr_M_over_m':    [r['tr_M_over_m'] for r in layer_results],
        'rf_ternary':     [r['rf_ternary'] for r in layer_results],
        'rf_int4':        [r['rf_int4'] for r in layer_results],
        'rf_delta_ternary': [r['rf_delta_ternary'] for r in layer_results],
        'rf_delta_int4':  [r['rf_delta_int4'] for r in layer_results],
        'svd_err_1':      [r['svd_errors']['1'] for r in layer_results],
        'svd_err_4':      [r['svd_errors']['4'] for r in layer_results],
        'svd_err_16':     [r['svd_errors']['16'] for r in layer_results],
        'layer_pos':      [r['layer_pos'] for r in layer_results],
    }

    if 'rf_gate' in layer_results[0]:
        corr_targets['rf_gate'] = [r['rf_gate'] for r in layer_results]
    if 'rf_attn' in layer_results[0]:
        corr_targets['rf_attn'] = [r['rf_attn'] for r in layer_results]

    correlations = {}
    for name, vals in corr_targets.items():
        if len(set(vals)) <= 1:
            correlations[name] = {'rho': 0.0, 'p': 1.0}
        else:
            rho, p = stats.spearmanr(rf_vals, vals)
            correlations[name] = {'rho': round(float(rho), 4), 'p': round(float(p), 8)}

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]['rho']), reverse=True)
    for name, v in sorted_corr:
        sig = "***" if v['p'] < 0.001 else "**" if v['p'] < 0.01 else "*" if v['p'] < 0.05 else ""
        print(f"    {name:<20} ρ={v['rho']:>+.4f}  p={v['p']:.2e}  {sig}")

    # Cleanup
    del model
    import gc; gc.collect()

    return {
        'model_key': model_key,
        'model_name': cfg['name'],
        'model_type': cfg['type'],
        'hidden_size': hid,
        'intermediate_size': m_ff,
        'num_layers': n_layers,
        'layers': layer_results,
        'random_init': random_init,
        'bounds': bounds,
        'correlations': correlations,
    }


# ======================================================================
# Cross-model summary
# ======================================================================

def print_cross_model_summary(all_results):
    print(f"\n{'='*70}")
    print(f"  CROSS-MODEL SUMMARY")
    print(f"{'='*70}")

    # Collect all correlation names that appear in every model
    all_names = set()
    for res in all_results.values():
        all_names.update(res['correlations'].keys())

    common_names = all_names.copy()
    for res in all_results.values():
        common_names &= set(res['correlations'].keys())

    # Print table: metric × model
    header = f"  {'metric':<20}"
    for key in all_results:
        header += f" {key:>14}"
    print(header)
    print(f"  {'-'*20}" + f" {'-'*14}" * len(all_results))

    # Sort by mean |ρ|
    def mean_abs_rho(name):
        rhos = [abs(res['correlations'][name]['rho']) for res in all_results.values()
                if name in res['correlations']]
        return np.mean(rhos) if rhos else 0

    for name in sorted(common_names, key=mean_abs_rho, reverse=True):
        row = f"  {name:<20}"
        for key, res in all_results.items():
            c = res['correlations'].get(name, {'rho': 0, 'p': 1})
            sig = "*" if c['p'] < 0.05 else " "
            row += f" {c['rho']:>+8.3f}{sig:>5}"
            # row += f" {c['rho']:>+.3f}{'*' if c['p']<0.05 else ' ':>1}"
        print(row)

    # R_F range per model
    print(f"\n  R_F range per model:")
    for key, res in all_results.items():
        rfs = [l['rf'] for l in res['layers']]
        print(f"    {key:<18} [{min(rfs):.4f}, {max(rfs):.4f}]  "
              f"mean={np.mean(rfs):.4f}  std={np.std(rfs):.4f}")

    # Random init verification
    print(f"\n  Random init R_F verification (9.07):")
    for key, res in all_results.items():
        ri = res['random_init']
        print(f"    {key:<18} theory={ri['theoretical_rf']:.8f}  "
              f"empirical={ri['empirical_mean']:.8f}±{ri['empirical_std']:.8f}")


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 70)
    print("BATCH FREE METRICS — 17 weight-only experiments")
    print("=" * 70)

    if len(sys.argv) > 1:
        keys = sys.argv[1:]
        if keys[0] == 'all':
            keys = list(MODEL_CONFIGS.keys())
    else:
        keys = list(MODEL_CONFIGS.keys())

    all_results = {}
    for key in keys:
        if key not in MODEL_CONFIGS:
            print(f"Unknown model: {key}, skipping")
            continue
        try:
            all_results[key] = run_model(key, MODEL_CONFIGS[key])
        except Exception as e:
            print(f"\n  ERROR on {key}: {e}")
            import traceback; traceback.print_exc()

    if len(all_results) > 1:
        print_cross_model_summary(all_results)

    # Save
    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'exp_batch_free_metrics.json')

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            v = convert(obj)
            if v is not obj:
                return v
            return super().default(obj)

    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    print(f"\nAll results saved to {filepath}")


if __name__ == '__main__':
    main()
