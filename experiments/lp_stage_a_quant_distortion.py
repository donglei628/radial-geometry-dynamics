"""
Quantization Sensitivity Stage A: Per-layer quantization distortion vs R_F.

For each model and each layer, compute:
  1. R_F (from trace formula, zero-cost)
  2. Quantization distortion under ternary / INT8 / INT4

Then correlate R_F with distortion (Spearman rho).

Hypothesis: high R_F => low distortion (more robust to quantization)

Usage:
    python lp_stage_a_quant_distortion.py [model_key|all]
"""

import sys
import os
import json
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import radial_fraction, make_x_hat_torch, ensure_output_dir


MODEL_CONFIGS = {
    'gpt2': {
        'name': 'gpt2',
        'type': 'gpt2',
    },
    'pythia-410m': {
        'name': 'EleutherAI/pythia-410m',
        'type': 'pythia',
    },
    'pythia-1b': {
        'name': 'EleutherAI/pythia-1b',
        'type': 'pythia',
    },
    'tinyllama': {
        'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'type': 'swiglu',
    },
    'qwen2.5-0.5b': {
        'name': 'Qwen/Qwen2.5-0.5B',
        'type': 'swiglu',
    },
    'qwen2.5-1.5b': {
        'name': 'Qwen/Qwen2.5-1.5B',
        'type': 'swiglu',
    },
    'phi-2': {
        'name': 'microsoft/phi-2',
        'type': 'phi2',
    },
}


# ======================================================================
# Quantization methods
# ======================================================================

def ternary_quantize(W):
    """Per-row ternary: W_ij -> sign(W_ij) * mean(|W_i.|)"""
    alpha = W.abs().mean(dim=1, keepdim=True)
    return torch.sign(W) * alpha


def int8_rtn_quantize(W):
    """Per-row INT8 round-to-nearest quantization."""
    row_max = W.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
    scale = row_max / 127.0
    W_q = torch.clamp(torch.round(W / scale), -128, 127)
    return W_q * scale


def int4_rtn_quantize(W):
    """Per-row INT4 round-to-nearest quantization."""
    row_max = W.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
    scale = row_max / 7.0
    W_q = torch.clamp(torch.round(W / scale), -8, 7)
    return W_q * scale


QUANT_METHODS = {
    'ternary': ternary_quantize,
    'int8': int8_rtn_quantize,
    'int4': int4_rtn_quantize,
}


# ======================================================================
# Trace formula
# ======================================================================

def compute_trace_formula(M, m):
    tr_M = float(torch.trace(M).item())
    M_fro_sq = float((M ** 2).sum().item())
    tr_M2 = float(torch.trace(M @ M).item())
    denom = (m + 2) * M_fro_sq
    if denom < 1e-15:
        return 0.0
    return (tr_M ** 2 + M_fro_sq + tr_M2) / denom


# ======================================================================
# Weight extraction (same as lp_stage1)
# ======================================================================

def get_layers_and_weights(model, model_type):
    if model_type == 'gpt2':
        layers = model.transformer.h
        def get_mlp_weights(layer):
            W1 = layer.mlp.c_fc.weight.detach().float().T
            W2 = layer.mlp.c_proj.weight.detach().float().T
            return W1, W2
        def get_full_fn(layer, W1, W2):
            def f(x):
                return (W2 @ F.gelu(W1 @ x.T)).T
            return f

    elif model_type == 'pythia':
        layers = model.gpt_neox.layers
        def get_mlp_weights(layer):
            W1 = layer.mlp.dense_h_to_4h.weight.detach().float()
            W2 = layer.mlp.dense_4h_to_h.weight.detach().float()
            return W1, W2
        def get_full_fn(layer, W1, W2):
            def f(x):
                return (W2 @ F.gelu(W1 @ x.T)).T
            return f

    elif model_type == 'swiglu':
        layers = model.model.layers
        def get_mlp_weights(layer):
            W1 = layer.mlp.up_proj.weight.detach().float()
            W2 = layer.mlp.down_proj.weight.detach().float()
            return W1, W2
        def get_full_fn(layer, W1, W2):
            W_gate = layer.mlp.gate_proj.weight.detach().float()
            def f(x):
                return (W2 @ (F.silu(W_gate @ x.T) * (W1 @ x.T))).T
            return f

    elif model_type == 'phi2':
        layers = model.model.layers
        def get_mlp_weights(layer):
            W1 = layer.mlp.fc1.weight.detach().float()
            W2 = layer.mlp.fc2.weight.detach().float()
            return W1, W2
        def get_full_fn(layer, W1, W2):
            def f(x):
                return (W2 @ F.gelu(W1 @ x.T, approximate='tanh')).T
            return f
    else:
        raise ValueError(f"Unknown: {model_type}")

    return layers, get_mlp_weights, get_full_fn


# ======================================================================
# Core: measure quantization distortion per layer
# ======================================================================

def measure_layer_distortion(W1, W2, full_fn, m, quant_method_name, n_samples=2048, n_seeds=5):
    """
    Measure normalized MSE from quantizing W2 (down_proj).

    distortion = E[||F(x) - F_q(x)||^2] / E[||F(x)||^2]

    where F_q uses quantized W2.
    """
    quant_fn = QUANT_METHODS[quant_method_name]

    distortions = []
    for seed in range(n_seeds):
        x_hat = make_x_hat_torch(n_samples, m, 'gaussian', seed=seed)

        with torch.no_grad():
            F_orig = full_fn(x_hat)

            W2_q = quant_fn(W2)
            # Recompute with quantized W2 — need to reconstruct full_fn logic
            # We quantize W2 only (down_proj), keeping W1 (up_proj) intact
            # This requires access to the activation function, which full_fn encapsulates
            # So we compute F_q by replacing W2 in the pipeline

            # For simplicity: compute intermediate activation, then apply W2_q
            z = W1 @ x_hat.T  # (hidden_mid, n_samples)
            # We need to apply the same activation as full_fn
            # This is a limitation — we'll compute distortion at the linear level: M vs M_q
            M_orig = W2 @ W1
            M_q = W2_q @ W1
            F_q = (M_q @ x_hat.T).T
            F_lin = (M_orig @ x_hat.T).T

            # Linear distortion: using M_q vs M
            num = ((F_lin - F_q) ** 2).sum(1).mean().item()
            den = (F_lin ** 2).sum(1).mean().item()
            if den > 1e-15:
                distortions.append(num / den)
            else:
                distortions.append(0.0)

    return float(np.mean(distortions))


def measure_layer_distortion_nonlinear(full_fn, full_fn_quantized, m, n_samples=2048, n_seeds=5):
    """Measure distortion for nonlinear (full) path."""
    distortions = []
    for seed in range(n_seeds):
        x_hat = make_x_hat_torch(n_samples, m, 'gaussian', seed=seed)
        with torch.no_grad():
            F_orig = full_fn(x_hat)
            F_q = full_fn_quantized(x_hat)
            num = ((F_orig - F_q) ** 2).sum(1).mean().item()
            den = (F_orig ** 2).sum(1).mean().item()
            if den > 1e-15:
                distortions.append(num / den)
            else:
                distortions.append(0.0)
    return float(np.mean(distortions))


def make_quantized_full_fn(model_type, layer, W1, W2_q):
    """Build a full nonlinear fn with quantized W2."""
    if model_type == 'gpt2':
        def f(x):
            return (W2_q @ F.gelu(W1 @ x.T)).T
        return f
    elif model_type == 'pythia':
        def f(x):
            return (W2_q @ F.gelu(W1 @ x.T)).T
        return f
    elif model_type == 'swiglu':
        W_gate = layer.mlp.gate_proj.weight.detach().float()
        def f(x):
            return (W2_q @ (F.silu(W_gate @ x.T) * (W1 @ x.T))).T
        return f
    elif model_type == 'phi2':
        def f(x):
            return (W2_q @ F.gelu(W1 @ x.T, approximate='tanh')).T
        return f
    else:
        raise ValueError(f"Unknown: {model_type}")


# ======================================================================
# Main experiment
# ======================================================================

def run_single_model(model_key, cfg):
    from transformers import AutoModelForCausalLM

    print(f"\n{'='*70}")
    print(f"  {model_key}: {cfg['name']}")
    print(f"{'='*70}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        cfg['name'], trust_remote_code=True,
        dtype=torch.float32, device_map='cpu',
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    config = model.config
    m = config.hidden_size
    num_layers = config.num_hidden_layers
    print(f"  m={m}, layers={num_layers}")

    layers_list, get_mlp_weights, get_full_fn = get_layers_and_weights(model, cfg['type'])

    print(f"\n  {'Layer':>5} {'R_F':>10} {'dist_tern':>10} {'dist_int8':>10} {'dist_int4':>10} "
          f"{'dist_nl_tern':>12} {'dist_nl_int4':>12}")
    print(f"  {'-'*82}")

    layer_results = []

    for layer_idx in range(num_layers):
        layer = layers_list[layer_idx]
        W1, W2 = get_mlp_weights(layer)
        M = W2 @ W1

        # R_F (trace formula)
        rf = compute_trace_formula(M, m)

        # Full nonlinear fn
        full_fn = get_full_fn(layer, W1, W2)

        # Distortion for each quantization method (linear level)
        dist_ternary = measure_layer_distortion(W1, W2, full_fn, m, 'ternary')
        dist_int8 = measure_layer_distortion(W1, W2, full_fn, m, 'int8')
        dist_int4 = measure_layer_distortion(W1, W2, full_fn, m, 'int4')

        # Nonlinear distortion (quantize W2 in full pipeline)
        W2_q_tern = ternary_quantize(W2)
        W2_q_int4 = int4_rtn_quantize(W2)
        fn_q_tern = make_quantized_full_fn(cfg['type'], layer, W1, W2_q_tern)
        fn_q_int4 = make_quantized_full_fn(cfg['type'], layer, W1, W2_q_int4)
        dist_nl_tern = measure_layer_distortion_nonlinear(full_fn, fn_q_tern, m)
        dist_nl_int4 = measure_layer_distortion_nonlinear(full_fn, fn_q_int4, m)

        print(f"  {layer_idx:>5} {rf:>10.6f} {dist_ternary:>10.6f} {dist_int8:>10.6f} "
              f"{dist_int4:>10.6f} {dist_nl_tern:>12.6f} {dist_nl_int4:>12.6f}")

        layer_results.append({
            'layer': layer_idx,
            'R_F_formula': rf,
            'distortion_ternary_linear': dist_ternary,
            'distortion_int8_linear': dist_int8,
            'distortion_int4_linear': dist_int4,
            'distortion_ternary_nonlinear': dist_nl_tern,
            'distortion_int4_nonlinear': dist_nl_int4,
        })

    # Correlations
    rf_vals = [r['R_F_formula'] for r in layer_results]
    print(f"\n  Correlations (R_F vs distortion):")
    print(f"  {'Quant method':<25} {'Spearman rho':>12} {'p-value':>12} {'Pearson r':>12}")
    print(f"  {'-'*63}")

    corr_results = {}
    for key in ['distortion_ternary_linear', 'distortion_int8_linear', 'distortion_int4_linear',
                 'distortion_ternary_nonlinear', 'distortion_int4_nonlinear']:
        dist_vals = [r[key] for r in layer_results]
        sp_rho, sp_p = stats.spearmanr(rf_vals, dist_vals)
        pr_r, pr_p = stats.pearsonr(rf_vals, dist_vals)
        label = key.replace('distortion_', '')
        print(f"  {label:<25} {sp_rho:>12.4f} {sp_p:>12.2e} {pr_r:>12.4f}")
        corr_results[key] = {
            'spearman_rho': float(sp_rho),
            'spearman_p': float(sp_p),
            'pearson_r': float(pr_r),
            'pearson_p': float(pr_p),
        }

    result = {
        'model_key': model_key,
        'model_name': cfg['name'],
        'model_type': cfg['type'],
        'hidden_size': m,
        'num_layers': num_layers,
        'layers': layer_results,
        'correlations': corr_results,
    }

    del model
    import gc; gc.collect()
    return result


def main():
    print("=" * 70)
    print("QUANTIZATION SENSITIVITY — STAGE A: R_F vs DISTORTION")
    print("=" * 70)

    if len(sys.argv) > 1:
        keys = sys.argv[1:]
        if keys == ['all']:
            keys = list(MODEL_CONFIGS.keys())
    else:
        keys = list(MODEL_CONFIGS.keys())

    all_results = {}
    for key in keys:
        if key not in MODEL_CONFIGS:
            print(f"Unknown: {key}, skipping")
            continue
        try:
            all_results[key] = run_single_model(key, MODEL_CONFIGS[key])
        except Exception as e:
            print(f"\n  ERROR on {key}: {e}")
            import traceback; traceback.print_exc()

    # Save
    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'quant_stage_a_rf_distortion.json')
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {filepath}")

    # Cross-model summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("CROSS-MODEL SUMMARY: Spearman rho (R_F vs distortion)")
        print(f"{'='*70}")
        print(f"  {'Model':<18} {'ternary_lin':>12} {'int8_lin':>12} {'int4_lin':>12} "
              f"{'tern_nonlin':>12} {'int4_nonlin':>12}")
        print(f"  {'-'*80}")
        rhos_by_method = {}
        for key, res in all_results.items():
            c = res['correlations']
            vals = []
            for method in ['distortion_ternary_linear', 'distortion_int8_linear',
                           'distortion_int4_linear', 'distortion_ternary_nonlinear',
                           'distortion_int4_nonlinear']:
                rho = c[method]['spearman_rho']
                vals.append(rho)
                rhos_by_method.setdefault(method, []).append(rho)
            print(f"  {key:<18} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f} "
                  f"{vals[3]:>12.4f} {vals[4]:>12.4f}")

        print(f"\n  {'MEAN':<18}", end='')
        for method in ['distortion_ternary_linear', 'distortion_int8_linear',
                       'distortion_int4_linear', 'distortion_ternary_nonlinear',
                       'distortion_int4_nonlinear']:
            print(f" {np.mean(rhos_by_method[method]):>12.4f}", end='')
        print()


if __name__ == '__main__':
    main()
