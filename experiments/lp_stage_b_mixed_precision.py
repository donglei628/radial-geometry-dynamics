"""
Quantization Sensitivity Stage B: Mixed-Precision Quantization + PPL.

For SwiGLU models, compare quantization strategies:
  1. FP16 baseline (no quantization)
  2. Uniform INT4 (all layers INT4)
  3. R_F-guided (low R_F → keep FP16, high R_F → INT4)
  4. Inverse R_F (high R_F → keep FP16, control group)
  5. Random (random layers keep FP16, averaged over 5 seeds)

Measures perplexity on WikiText-2 test set.

Usage:
    python lp_stage_b_mixed_precision.py [model_key] [fp16_budget_pct]
    python lp_stage_b_mixed_precision.py qwen2.5-0.5b 25
    python lp_stage_b_mixed_precision.py tinyllama 25
    python lp_stage_b_mixed_precision.py all
"""

import sys
import os
import json
import time
import math
import copy
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import ensure_output_dir


MODEL_CONFIGS = {
    'qwen2.5-0.5b': {
        'name': 'Qwen/Qwen2.5-0.5B',
        'type': 'swiglu',
    },
    'tinyllama': {
        'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'type': 'swiglu',
    },
    'qwen2.5-1.5b': {
        'name': 'Qwen/Qwen2.5-1.5B',
        'type': 'swiglu',
    },
}

FP16_BUDGET_PCTS = [10, 20, 25, 30, 40, 50]


# ======================================================================
# Quantization
# ======================================================================

def int4_rtn_quantize(W):
    """Per-row INT4 round-to-nearest."""
    row_max = W.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
    scale = row_max / 7.0
    W_q = torch.clamp(torch.round(W / scale), -8, 7)
    return W_q * scale


# ======================================================================
# R_F computation
# ======================================================================

def compute_rf_per_layer(model, model_type):
    """Compute R_F for each layer (trace formula, zero cost)."""
    if model_type == 'swiglu':
        layers = model.model.layers
    else:
        raise ValueError(f"Unsupported: {model_type}")

    rf_values = []
    for layer in layers:
        W1 = layer.mlp.up_proj.weight.detach().float()
        W2 = layer.mlp.down_proj.weight.detach().float()
        M = W2 @ W1
        m = M.shape[0]
        tr_M = torch.trace(M).item()
        tr_M2 = torch.trace(M @ M).item()
        fro_sq = (M ** 2).sum().item()
        denom = (m + 2) * fro_sq
        rf = (tr_M ** 2 + fro_sq + tr_M2) / denom if denom > 1e-15 else 0.0
        rf_values.append(rf)
    return rf_values


# ======================================================================
# Layer quantization strategies
# ======================================================================

def get_fp16_layers(strategy, rf_values, num_layers, n_fp16, seed=None):
    """
    Return set of layer indices to keep in FP16.
    Remaining layers will be quantized to INT4.
    """
    if strategy == 'rf_guided':
        # Keep layers with LOWEST R_F in FP16 (most sensitive)
        sorted_idx = sorted(range(num_layers), key=lambda i: rf_values[i])
        return set(sorted_idx[:n_fp16])

    elif strategy == 'inverse_rf':
        # Keep layers with HIGHEST R_F in FP16 (control, should be worse)
        sorted_idx = sorted(range(num_layers), key=lambda i: rf_values[i], reverse=True)
        return set(sorted_idx[:n_fp16])

    elif strategy == 'random':
        rng = np.random.default_rng(seed)
        return set(rng.choice(num_layers, size=n_fp16, replace=False).tolist())

    elif strategy == 'uniform_int4':
        return set()  # No layers in FP16

    elif strategy == 'fp16_baseline':
        return set(range(num_layers))  # All layers in FP16

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def quantize_model_mlp(model, model_type, fp16_layers):
    """
    Quantize MLP weights (up_proj, down_proj, gate_proj) of layers NOT in fp16_layers.
    Modifies model in-place.
    """
    if model_type == 'swiglu':
        layers = model.model.layers
    else:
        raise ValueError(f"Unsupported: {model_type}")

    n_quantized = 0
    for i, layer in enumerate(layers):
        if i not in fp16_layers:
            with torch.no_grad():
                layer.mlp.up_proj.weight.copy_(
                    int4_rtn_quantize(layer.mlp.up_proj.weight.float()).to(layer.mlp.up_proj.weight.dtype))
                layer.mlp.down_proj.weight.copy_(
                    int4_rtn_quantize(layer.mlp.down_proj.weight.float()).to(layer.mlp.down_proj.weight.dtype))
                layer.mlp.gate_proj.weight.copy_(
                    int4_rtn_quantize(layer.mlp.gate_proj.weight.float()).to(layer.mlp.gate_proj.weight.dtype))
            n_quantized += 1
    return n_quantized


# ======================================================================
# Perplexity evaluation
# ======================================================================

def evaluate_ppl(model, tokenizer, device='cpu', max_length=2048, stride=512):
    """
    Evaluate perplexity on WikiText-2 test set using sliding window.
    """
    from datasets import load_dataset

    print(f"    Loading WikiText-2...", flush=True)
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(dataset['text'])
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids

    seq_len = input_ids.size(1)
    total_steps = (seq_len - 1) // stride + 1
    print(f"    seq_len={seq_len}, stride={stride}, ~{total_steps} steps", flush=True)

    nlls = []
    n_tokens = 0
    step = 0

    for begin_loc in range(0, seq_len - 1, stride):
        end_loc = min(begin_loc + max_length, seq_len)

        input_chunk = input_ids[:, begin_loc:end_loc].to(device)

        with torch.no_grad():
            outputs = model(input_chunk)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_chunk[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum'
        )
        nlls.append(loss.item())
        n_tokens += shift_labels.numel()
        step += 1

        if step % 100 == 0:
            print(f"    step {step}/{total_steps}, tokens={n_tokens}", flush=True)

        if end_loc >= seq_len:
            break

    ppl = math.exp(sum(nlls) / n_tokens)
    return ppl, n_tokens


# ======================================================================
# Main experiment
# ======================================================================

def run_single_model(model_key, cfg, fp16_budgets=None, device='cuda'):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if fp16_budgets is None:
        fp16_budgets = FP16_BUDGET_PCTS

    print(f"\n{'='*70}")
    print(f"  STAGE B: {model_key} ({cfg['name']})")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(cfg['name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1: Load model and compute R_F
    print(f"\n  [1/3] Loading model and computing R_F...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        cfg['name'], trust_remote_code=True,
        dtype=torch.float16, device_map=device,
    )
    model.eval()
    load_time = time.time() - t0

    config = model.config
    m = config.hidden_size
    num_layers = config.num_hidden_layers
    print(f"  Loaded in {load_time:.1f}s (m={m}, layers={num_layers})", flush=True)

    rf_values = compute_rf_per_layer(model, cfg['type'])
    print(f"  R_F range: {min(rf_values):.4f} - {max(rf_values):.4f}", flush=True)

    # Save original state_dict for resetting
    print(f"  Saving original weights...", flush=True)
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Step 2: FP16 baseline PPL
    print(f"\n  [2/3] Evaluating FP16 baseline PPL...")
    t0 = time.time()
    ppl_fp16, n_tokens = evaluate_ppl(model, tokenizer, device=device)
    t_ppl = time.time() - t0
    print(f"  FP16 PPL = {ppl_fp16:.2f} ({n_tokens} tokens, {t_ppl:.1f}s)")

    # Step 3: Test each strategy at each budget level
    print(f"\n  [3/3] Testing quantization strategies...")

    all_results = []

    for budget_pct in fp16_budgets:
        n_fp16 = max(1, int(num_layers * budget_pct / 100))
        n_int4 = num_layers - n_fp16

        print(f"\n  --- Budget: {budget_pct}% FP16 ({n_fp16} FP16 + {n_int4} INT4) ---")

        strategies = ['rf_guided', 'inverse_rf', 'uniform_int4']
        random_seeds = [0, 1, 2, 3, 4]

        for strategy in strategies:
            # Reset weights
            model.load_state_dict(original_state)

            fp16_set = get_fp16_layers(strategy, rf_values, num_layers, n_fp16)
            n_q = quantize_model_mlp(model, cfg['type'], fp16_set)

            t0 = time.time()
            ppl, _ = evaluate_ppl(model, tokenizer, device=device)
            t_eval = time.time() - t0

            ppl_ratio = ppl / ppl_fp16

            fp16_list = sorted(fp16_set)
            print(f"  {strategy:<15} PPL={ppl:>8.2f} (x{ppl_ratio:.3f}) "
                  f"[{t_eval:.1f}s] FP16={fp16_list}")

            all_results.append({
                'budget_pct': budget_pct,
                'n_fp16': n_fp16,
                'n_int4': n_int4,
                'strategy': strategy,
                'ppl': ppl,
                'ppl_ratio': ppl_ratio,
                'fp16_layers': fp16_list,
                'seed': None,
            })

        # Random: average over seeds
        random_ppls = []
        for seed in random_seeds:
            model.load_state_dict(original_state)
            fp16_set = get_fp16_layers('random', rf_values, num_layers, n_fp16, seed=seed)
            quantize_model_mlp(model, cfg['type'], fp16_set)
            ppl, _ = evaluate_ppl(model, tokenizer, device=device)
            random_ppls.append(ppl)

        mean_random_ppl = np.mean(random_ppls)
        std_random_ppl = np.std(random_ppls)
        print(f"  {'random (5 seeds)':<15} PPL={mean_random_ppl:>8.2f}±{std_random_ppl:.2f} "
              f"(x{mean_random_ppl/ppl_fp16:.3f})")

        all_results.append({
            'budget_pct': budget_pct,
            'n_fp16': n_fp16,
            'n_int4': n_int4,
            'strategy': 'random_mean',
            'ppl': mean_random_ppl,
            'ppl_std': std_random_ppl,
            'ppl_ratio': mean_random_ppl / ppl_fp16,
            'random_ppls': random_ppls,
            'seed': 'averaged',
        })

    # Summary table
    print(f"\n  {'='*70}")
    print(f"  SUMMARY: {model_key} (FP16 baseline PPL = {ppl_fp16:.2f})")
    print(f"  {'='*70}")
    print(f"  {'Budget':>7} {'rf_guided':>12} {'inverse_rf':>12} {'random':>12} {'uniform':>12} "
          f"{'rf_vs_rand':>12}")
    print(f"  {'-'*70}")

    for budget_pct in fp16_budgets:
        budget_results = {r['strategy']: r for r in all_results if r['budget_pct'] == budget_pct}
        rf_ppl = budget_results.get('rf_guided', {}).get('ppl', 0)
        inv_ppl = budget_results.get('inverse_rf', {}).get('ppl', 0)
        rand_ppl = budget_results.get('random_mean', {}).get('ppl', 0)
        uni_ppl = budget_results.get('uniform_int4', {}).get('ppl', 0)
        rf_vs_rand = (rf_ppl - rand_ppl) / rand_ppl * 100 if rand_ppl > 0 else 0

        print(f"  {budget_pct:>6}% {rf_ppl:>12.2f} {inv_ppl:>12.2f} {rand_ppl:>12.2f} "
              f"{uni_ppl:>12.2f} {rf_vs_rand:>+11.2f}%")

    result = {
        'model_key': model_key,
        'model_name': cfg['name'],
        'hidden_size': m,
        'num_layers': num_layers,
        'ppl_fp16': ppl_fp16,
        'rf_values': rf_values,
        'results': all_results,
    }

    del model, original_state
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    print("=" * 70)
    print("STAGE B: MIXED-PRECISION QUANTIZATION + PPL")
    print("=" * 70)

    if len(sys.argv) > 1:
        keys = sys.argv[1:]
        if keys[0] == 'all':
            keys = list(MODEL_CONFIGS.keys())
        budgets = None
        if len(sys.argv) > 2 and sys.argv[2].isdigit():
            budgets = [int(sys.argv[2])]
    else:
        keys = list(MODEL_CONFIGS.keys())
        budgets = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = {}
    for key in keys:
        if key not in MODEL_CONFIGS:
            print(f"Unknown: {key}, skipping")
            continue
        try:
            all_results[key] = run_single_model(key, MODEL_CONFIGS[key],
                                                 fp16_budgets=budgets, device=device)
        except Exception as e:
            print(f"\n  ERROR on {key}: {e}")
            import traceback; traceback.print_exc()

    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'quant_stage_b_mixed_precision.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {filepath}")


if __name__ == '__main__':
    main()
