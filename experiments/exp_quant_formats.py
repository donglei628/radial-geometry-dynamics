"""
Quantization-format distortion vs R_F (weight-only, CPU, ~minutes).

  1.09 - NF4 (QLoRA normal-float-4) distortion vs R_F
  1.10 - Group quantization (g128) INT4 distortion vs R_F
  1.11 - Per-channel (per-input-col) vs per-row INT4 granularity vs R_F
  1.12 - Attention (M_attn = W_O @ W_V) quantization vs MLP quantization

Per layer, quantize MLP weights with each scheme, recompute M = W_down @ W_up,
distortion = ||M - M_q||_F / ||M||_F. Correlate with R_F. Consistent with the
Stage-A protocol (1.01/1.02).

Usage:
    python exp_quant_formats.py
"""

import sys
import os
import json
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import ensure_output_dir

MODELS = {
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B',
    'tinyllama':    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B',
}

NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928, -0.5250730, -0.3949175, -0.2844444, -0.1847463,
    -0.0911838, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.3379152,
    0.4407098, 0.5626170, 0.7229568, 1.0])


def q_int4_per_row(W):
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 7.0
    return torch.clamp(torch.round(W / s), -8, 7) * s


def q_int4_per_col(W):
    s = W.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / 7.0
    return torch.clamp(torch.round(W / s), -8, 7) * s


def q_int4_group(W, g=128):
    r, c = W.shape
    pad = (g - c % g) % g
    if pad:
        W = torch.cat([W, torch.zeros(r, pad, dtype=W.dtype)], dim=1)
    Wg = W.view(r, -1, g)
    s = Wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / 7.0
    Wq = torch.clamp(torch.round(Wg / s), -8, 7) * s
    Wq = Wq.view(r, -1)
    return Wq[:, :c]


def q_nf4(W):
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    Wn = (W / s).unsqueeze(-1)            # [r, c, 1]
    lv = NF4_LEVELS.to(W.dtype)
    idx = (Wn - lv).abs().argmin(dim=-1)  # [r, c]
    return lv[idx] * s


def distortion(M, Mq):
    return ((M - Mq).norm() / M.norm().clamp(min=1e-12)).item()


def run_model(key, name, device='cpu'):
    from transformers import AutoModelForCausalLM
    print(f"\n{'#'*68}\n# {key}\n{'#'*68}")
    model = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=True)
    model.eval()
    layers = model.model.layers
    schemes = {'int4_row': q_int4_per_row, 'int4_col': q_int4_per_col,
               'int4_g128': q_int4_group, 'nf4': q_nf4}
    rf, dist = [], {s: [] for s in schemes}
    attn_dist = []
    for layer in layers:
        W1 = layer.mlp.up_proj.weight.detach().float()
        W2 = layer.mlp.down_proj.weight.detach().float()
        M = W2 @ W1
        m = M.shape[0]
        tr_M = torch.trace(M).item(); tr_M2 = torch.trace(M @ M).item()
        fro = (M ** 2).sum().item()
        rf.append((tr_M**2 + fro + tr_M2) / ((m + 2) * fro) if fro > 0 else 0.0)
        for sname, qf in schemes.items():
            Mq = qf(W2) @ qf(W1)
            dist[sname].append(distortion(M, Mq))
        # 1.12: attention M_attn = W_O @ W_V
        attn = layer.self_attn
        Wv = attn.v_proj.weight.detach().float()
        Wo = attn.o_proj.weight.detach().float()
        # head dims may differ (GQA): M_attn = Wo @ Wv is [m, kv_dim]@... ensure shapes
        try:
            Ma = Wo @ Wv
            Maq = q_int4_per_row(Wo) @ q_int4_per_row(Wv)
            attn_dist.append(distortion(Ma, Maq))
        except Exception:
            attn_dist.append(float('nan'))

    rf_a = np.array(rf)
    print(f"  {len(layers)} layers")
    print(f"\n  Spearman ρ(R_F, distortion) per scheme:")
    out = {'rf': rf, 'distortion': dist, 'attn_distortion': attn_dist}
    for sname in schemes:
        rho, p = stats.spearmanr(rf_a, np.array(dist[sname]))
        print(f"    {sname:<12} ρ={rho:+.3f} (p={p:.2e})")
        out[f'rho_{sname}'] = {'rho': float(rho), 'p': float(p)}
    # 1.12 attn
    ad = np.array(attn_dist)
    if not np.isnan(ad).all():
        rho, p = stats.spearmanr(rf_a, ad)
        mlp_mean = np.mean(dist['int4_row'])
        attn_mean = np.nanmean(ad)
        print(f"\n  1.12 attn: ρ(R_F, attn_dist)={rho:+.3f} (p={p:.2e})")
        print(f"    mean MLP int4 distortion={mlp_mean:.4f}  "
              f"attn int4 distortion={attn_mean:.4f}")
        out['rho_attn'] = {'rho': float(rho), 'p': float(p),
                           'mlp_mean': float(mlp_mean), 'attn_mean': float(attn_mean)}
    del model
    import gc; gc.collect()
    return out


def main():
    keys = [a for a in sys.argv[1:] if a in MODELS] or list(MODELS)
    fp = os.path.join(ensure_output_dir('results'), 'exp_quant_formats.json')
    results = {}
    if os.path.exists(fp):
        with open(fp) as f:
            results = json.load(f)
    for key in keys:
        try:
            results[key] = run_model(key, MODELS[key])
            with open(fp, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"  ERROR {key}: {e}")
            import traceback; traceback.print_exc()

    # Cross-model summary
    print(f"\n{'='*68}\n  CROSS-MODEL ρ(R_F, distortion) summary\n{'='*68}")
    schemes = ['int4_row', 'int4_col', 'int4_g128', 'nf4']
    print(f"  {'model':<16} " + ' '.join(f'{s:>10}' for s in schemes))
    for key in keys:
        if key in results and 'rho_int4_row' in results[key]:
            row = ' '.join(f"{results[key][f'rho_{s}']['rho']:>+10.3f}" for s in schemes)
            print(f"  {key:<16} {row}")
    print(f"\nSaved to {fp}")


if __name__ == '__main__':
    main()
