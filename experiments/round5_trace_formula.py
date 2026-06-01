"""
Round 5 Follow-up: Verify the trace formula for R_F(linear)
=============================================================

Theory: For M = W2*W1 acting linearly on x_hat:
    R_F(linear) = [tr(M)^2 + ||M||_F^2 + tr(M^2)] / [m * ||M||_F^2]
                = 1/m + tr(M)^2 / (m * ||M||_F^2) + tr(M^2) / (m * ||M||_F^2)

Test: Compute this formula and compare to measured R_F(linear).
"""

import sys
import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as torchF

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import radial_fraction, make_x_hat_torch, ensure_output_dir


def run_trace_formula_test():
    print("=" * 70)
    print("TRACE FORMULA VERIFICATION")
    print("R_F(linear) = [tr(M)^2 + ||M||_F^2 + tr(M^2)] / [m * ||M||_F^2]")
    print("=" * 70)

    from transformers import AutoModelForCausalLM

    MODEL_CONFIGS = {
        'tinyllama': {
            'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'type': 'swiglu',
        },
        'pythia-1b': {
            'name': 'EleutherAI/pythia-1b',
            'type': 'standard_mlp',
        },
    }

    B = 4096
    n_seeds = 10

    all_results = {}

    for model_key, cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_key}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            cfg['name'], trust_remote_code=True,
            torch_dtype=torch.float32, device_map='cpu',
        )
        model.eval()

        config = model.config
        m = config.hidden_size
        num_layers = config.num_hidden_layers

        if cfg['type'] == 'swiglu':
            decoder_layers = model.model.layers
        else:
            decoder_layers = model.gpt_neox.layers

        print(f"    m={m}, 1/m={1/m:.6f}")
        print(f"\n    {'Layer':>5} {'R_F emp':>10} {'R_F formula':>12} {'ratio':>8} "
              f"{'tr(M)^2/m||M||':>15} {'||M||_F^2/m':>12} {'tr(M^2)/m||M||':>15}")
        print(f"    {'-'*85}")

        layer_results = []

        for layer_idx in range(num_layers):
            layer = decoder_layers[layer_idx]
            mlp = layer.mlp

            # Get W1 and W2 for the linear product M = W2 * W1
            if cfg['type'] == 'swiglu':
                W_up = mlp.up_proj.weight.detach().float()    # (intermediate, m)
                W_down = mlp.down_proj.weight.detach().float() # (m, intermediate)
                W1 = W_up
                W2 = W_down
            else:
                W1 = mlp.dense_h_to_4h.weight.detach().float()  # (4m, m)
                W2 = mlp.dense_4h_to_h.weight.detach().float()  # (m, 4m)

            # Compute M = W2 * W1 (m x m)
            M = W2 @ W1  # (m, m)

            # Trace formula components
            tr_M = float(torch.trace(M).item())
            M_fro_sq = float((M ** 2).sum().item())  # ||M||_F^2
            tr_M2 = float(torch.trace(M @ M).item())  # tr(M^2)

            # Formula prediction
            R_F_formula = (tr_M ** 2 + M_fro_sq + tr_M2) / (m * M_fro_sq)

            # Component breakdown
            term1 = 1.0 / m  # baseline
            term2 = tr_M ** 2 / (m * M_fro_sq)  # trace term
            term3 = tr_M2 / (m * M_fro_sq)  # quadratic form term

            # Empirical R_F(linear)
            RF_emp_values = []
            for seed in range(n_seeds):
                x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed)
                with torch.no_grad():
                    F_out = (M @ x_hat.T).T
                RF_emp_values.append(radial_fraction(x_hat, F_out))
            RF_emp = float(np.mean(RF_emp_values))

            ratio = RF_emp / R_F_formula if R_F_formula > 1e-10 else float('inf')

            print(f"    {layer_idx:>5} {RF_emp:>10.6f} {R_F_formula:>12.6f} {ratio:>8.4f} "
                  f"{term2:>15.6f} {M_fro_sq/m:>12.4f} {term3:>15.6f}")

            layer_results.append({
                'layer': layer_idx,
                'R_F_empirical': RF_emp,
                'R_F_formula': R_F_formula,
                'ratio': ratio,
                'tr_M': tr_M,
                'M_fro_sq': M_fro_sq,
                'tr_M2': tr_M2,
                'term_baseline': term1,
                'term_trace': term2,
                'term_quadratic': term3,
            })

        # Summary
        ratios = [r['ratio'] for r in layer_results]
        print(f"\n    Formula accuracy: mean ratio = {np.mean(ratios):.4f}, "
              f"std = {np.std(ratios):.4f}")
        print(f"    (ratio = 1.0 means formula is exact)")

        # Dominant term analysis
        trace_terms = [r['term_trace'] for r in layer_results]
        quad_terms = [r['term_quadratic'] for r in layer_results]
        print(f"\n    Dominant term:")
        print(f"      tr(M)^2 / (m||M||_F^2): mean={np.mean(trace_terms):.6f}")
        print(f"      tr(M^2) / (m||M||_F^2): mean={np.mean(quad_terms):.6f}")
        print(f"      Baseline 1/m:            {1/m:.6f}")

        # Now test: does R_F(full, nonlinear) correlate with R_F_formula?
        # Load nonlinear composition results
        nl_path = os.path.join(os.path.dirname(__file__), 'results', 'nonlinear_composition.json')
        if os.path.exists(nl_path):
            with open(nl_path) as f:
                nl_data = json.load(f)
            if model_key in nl_data:
                nl_layers = nl_data[model_key]['layers']
                from scipy.stats import pearsonr, spearmanr

                rf_full = [l['R_F_full'] for l in nl_layers]
                rf_formula = [r['R_F_formula'] for r in layer_results]

                if len(rf_full) == len(rf_formula):
                    pr, pp = pearsonr(rf_full, rf_formula)
                    sr, sp = spearmanr(rf_full, rf_formula)
                    print(f"\n    Correlation R_F(nonlinear) vs R_F(trace formula):")
                    print(f"      Pearson r = {pr:.4f} (p = {pp:.2e})")
                    print(f"      Spearman rho = {sr:.4f} (p = {sp:.2e})")

                    # Suppression factor alpha = R_F(full) / R_F(linear)
                    alphas = [f/l if l > 1e-10 else 0 for f, l in zip(rf_full, rf_formula)]
                    print(f"\n    Activation suppression factor alpha = R_F(full)/R_F(formula):")
                    print(f"      mean = {np.mean(alphas):.4f}, std = {np.std(alphas):.4f}")
                    print(f"      range = [{min(alphas):.4f}, {max(alphas):.4f}]")

        all_results[model_key] = {
            'model': cfg['name'],
            'type': cfg['type'],
            'hidden_size': m,
            'num_layers': num_layers,
            'layers': layer_results,
        }

        del model
        import gc; gc.collect()

    # Save
    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'trace_formula_verification.json')
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {filepath}")

    return all_results


if __name__ == '__main__':
    run_trace_formula_test()
