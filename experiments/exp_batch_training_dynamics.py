"""
Theme 4: Training Dynamics of R_F via Pythia checkpoints.

Pythia releases full model snapshots at step 0, 1, 2, 4, ..., 143000.
R_F is a geometric evolution quantity — this experiment tracks how it
develops from random init, which layers stabilize first, and whether the
initial value matches theory. This is R_F's natural home: a descriptive
quantity with no simpler substitute.

Experiments covered:
  4.01 - Pythia R_F evolution across checkpoints (per-layer trajectories)
  4.04 - Per-layer convergence speed (which layers stabilize first)
  4.06 - R_F at random init: theoretical value vs measured

CPU-only (weight-only R_F computation), so it does not contend with any
GPU experiment. Saves incrementally per (model, checkpoint) — safe to
interrupt and resume; already-computed checkpoints are skipped.

Usage:
    python exp_batch_training_dynamics.py                 # default: 160m then 410m
    python exp_batch_training_dynamics.py pythia-70m
    python exp_batch_training_dynamics.py pythia-160m pythia-410m
    python exp_batch_training_dynamics.py analysis        # analysis only
"""

import sys
import os
import json
import time
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import ensure_output_dir

RESULT_FILE = 'exp_batch_training_dynamics.json'

MODEL_CONFIGS = {
    'pythia-70m':  {'name': 'EleutherAI/pythia-70m',  'm': 512,  'layers': 6},
    'pythia-160m': {'name': 'EleutherAI/pythia-160m', 'm': 768,  'layers': 12},
    'pythia-410m': {'name': 'EleutherAI/pythia-410m', 'm': 1024, 'layers': 24},
    'pythia-1b':   {'name': 'EleutherAI/pythia-1b',   'm': 2048, 'layers': 16},
}

# Log-spaced checkpoints spanning the full training run.
CHECKPOINTS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]


# ======================================================================
# R_F computation for Pythia (GPT-NeoX MLP)
# ======================================================================

def compute_rf_pythia(model):
    """R_F per MLP layer. M = W_down @ W_up = dense_4h_to_h @ dense_h_to_4h."""
    layers = model.gpt_neox.layers
    out = []
    for layer in layers:
        W1 = layer.mlp.dense_h_to_4h.weight.detach().float()  # [4m, m] up
        W2 = layer.mlp.dense_4h_to_h.weight.detach().float()  # [m, 4m] down
        M = W2 @ W1                                            # [m, m]
        m = M.shape[0]
        tr_M = torch.trace(M).item()
        tr_M2 = torch.trace(M @ M).item()
        fro_sq = (M ** 2).sum().item()
        denom = (m + 2) * fro_sq
        rf = (tr_M ** 2 + fro_sq + tr_M2) / denom if denom > 1e-15 else 0.0
        out.append({
            'rf': rf,
            'tr_M': tr_M,
            'tr_M2': tr_M2,
            'fro_sq': fro_sq,
            'signed_tr_ratio': tr_M / math.sqrt(fro_sq) if fro_sq > 1e-15 else 0.0,
        })
    return out


# ======================================================================
# Result I/O
# ======================================================================

def load_results():
    outdir = ensure_output_dir('results')
    fp = os.path.join(outdir, RESULT_FILE)
    if os.path.exists(fp):
        with open(fp) as f:
            return json.load(f)
    return {}


def save_results(data):
    outdir = ensure_output_dir('results')
    fp = os.path.join(outdir, RESULT_FILE)
    with open(fp, 'w') as f:
        json.dump(data, f, indent=2)


# ======================================================================
# Per-model checkpoint sweep
# ======================================================================

def run_model(model_key, cfg, free_cache=True):
    from transformers import GPTNeoXForCausalLM

    print(f"\n{'#'*70}")
    print(f"# TRAINING DYNAMICS: {model_key} ({cfg['name']})")
    print(f"{'#'*70}")

    results = load_results()
    model_results = results.get(model_key, {})
    checkpoints = model_results.get('checkpoints', {})

    done = set(int(k) for k in checkpoints.keys())
    remaining = [s for s in CHECKPOINTS if s not in done]
    print(f"  {len(done)}/{len(CHECKPOINTS)} checkpoints done, {len(remaining)} remaining")

    for step in remaining:
        rev = f"step{step}"
        print(f"\n  [step {step}] loading revision {rev}...", flush=True)
        t0 = time.time()
        try:
            model = GPTNeoXForCausalLM.from_pretrained(
                cfg['name'], revision=rev,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            print(f"    FAILED to load {rev}: {e}", flush=True)
            continue
        model.eval()
        t_load = time.time() - t0

        rf_data = compute_rf_pythia(model)
        rfs = [d['rf'] for d in rf_data]
        t_total = time.time() - t0

        print(f"    loaded+computed in {t_total:.1f}s (load {t_load:.1f}s)  "
              f"R_F range [{min(rfs):.5f}, {max(rfs):.5f}]  "
              f"mean {np.mean(rfs):.5f}  std {np.std(rfs):.5f}", flush=True)

        checkpoints[str(step)] = {
            'step': step,
            'layers': rf_data,
            'rf_per_layer': rfs,
            'rf_mean': float(np.mean(rfs)),
            'rf_std': float(np.std(rfs)),
            'rf_min': float(min(rfs)),
            'rf_max': float(max(rfs)),
        }

        model_results['checkpoints'] = checkpoints
        model_results['model_name'] = cfg['name']
        model_results['m'] = cfg['m']
        model_results['num_layers'] = cfg['layers']
        results[model_key] = model_results
        save_results(results)

        # Free disk: remove this revision's HF cache snapshot
        del model
        import gc; gc.collect()
        if free_cache:
            try:
                _free_revision_cache(cfg['name'], rev)
            except Exception as e:
                print(f"    (cache cleanup skipped: {e})", flush=True)

    print(f"\n  {model_key}: all {len(CHECKPOINTS)} checkpoints complete")
    return model_results


def _free_revision_cache(repo_id, revision):
    """Delete a single revision snapshot from the HF cache to save disk."""
    from huggingface_hub import scan_cache_dir
    cache = scan_cache_dir()
    to_delete = []
    for repo in cache.repos:
        if repo.repo_id == repo_id:
            for rev in repo.revisions:
                if revision in rev.refs:
                    to_delete.append(rev.commit_hash)
    if to_delete:
        cache.delete_revisions(*to_delete).execute()


# ======================================================================
# Analysis: 4.01 trajectories, 4.04 convergence, 4.06 init theory
# ======================================================================

def run_analysis(results=None):
    from scipy import stats
    if results is None:
        results = load_results()

    print(f"\n{'#'*70}")
    print(f"# ANALYSIS: R_F training dynamics (4.01 / 4.04 / 4.06)")
    print(f"{'#'*70}")

    analysis = {}
    for model_key, data in results.items():
        if model_key == 'analysis':
            continue
        checkpoints = data.get('checkpoints', {})
        if len(checkpoints) < 3:
            continue

        steps = sorted(int(k) for k in checkpoints.keys())
        m = data['m']
        num_layers = data['num_layers']

        print(f"\n  {'='*64}")
        print(f"  {model_key} (m={m}, L={num_layers}, {len(steps)} checkpoints)")
        print(f"  {'='*64}")

        # 4.06: init value vs theory
        # For i.i.d. random M, E[R_F] = (m+3)/(m(m+2)) ... but Pythia M=W2@W1
        # is a product so measured step0 is the empirical baseline.
        init = checkpoints[str(steps[0])]
        final = checkpoints[str(steps[-1])]
        theory_iid = (m + 3) / (m * (m + 2))
        uniform_floor = 1.0 / (m + 2)
        print(f"\n  4.06 — Init (step {steps[0]}) vs theory:")
        print(f"    measured mean R_F @init: {init['rf_mean']:.6f}  "
              f"(std {init['rf_std']:.6f})")
        print(f"    theory (m+3)/(m(m+2)):   {theory_iid:.6f}")
        print(f"    uniform floor 1/(m+2):   {uniform_floor:.6f}")
        print(f"    final mean R_F:          {final['rf_mean']:.6f}  "
              f"(std {final['rf_std']:.6f})")
        print(f"    divergence amplification (std final/init): "
              f"{final['rf_std']/max(init['rf_std'],1e-9):.1f}x")

        # 4.01: per-layer trajectories — print compact table
        print(f"\n  4.01 — Per-layer R_F trajectory (rows=layers, cols=steps):")
        show_steps = [steps[0]] + steps[max(1, len(steps)//3)::max(1, len(steps)//4)]
        show_steps = sorted(set(show_steps + [steps[-1]]))
        hdr = "    L  " + " ".join(f"{s:>8d}" for s in show_steps)
        print(hdr)
        rf_matrix = np.array([checkpoints[str(s)]['rf_per_layer'] for s in steps])  # [T, L]
        for li in range(num_layers):
            traj = [checkpoints[str(s)]['rf_per_layer'][li] for s in show_steps]
            row = "    " + f"{li:>2d} " + " ".join(f"{v:>8.4f}" for v in traj)
            print(row)

        # 4.04: convergence — step at which each layer reaches 90% of its
        # total change from init to final.
        print(f"\n  4.04 — Convergence (step to reach 90% of init→final change):")
        conv_steps = []
        for li in range(num_layers):
            v0 = rf_matrix[0, li]
            vf = rf_matrix[-1, li]
            total = vf - v0
            if abs(total) < 1e-6:
                conv_steps.append(steps[-1])
                continue
            target = v0 + 0.9 * total
            reached = steps[-1]
            for ti, s in enumerate(steps):
                v = rf_matrix[ti, li]
                if (total > 0 and v >= target) or (total < 0 and v <= target):
                    reached = s
                    break
            conv_steps.append(reached)
        # Correlate convergence step with layer position
        positions = np.arange(num_layers) / (num_layers - 1)
        rho_pos, p_pos = stats.spearmanr(positions, conv_steps)
        order = np.argsort(conv_steps)
        print(f"    earliest-stabilizing layers: {order[:3].tolist()} "
              f"(steps {[conv_steps[i] for i in order[:3]]})")
        print(f"    latest-stabilizing layers:   {order[-3:].tolist()} "
              f"(steps {[conv_steps[i] for i in order[-3:]]})")
        print(f"    ρ(layer_pos, conv_step) = {rho_pos:+.3f} (p={p_pos:.2e})")

        # divergence (cross-layer std) over training
        print(f"\n  Divergence (cross-layer R_F std) over training:")
        for s in show_steps:
            c = checkpoints[str(s)]
            print(f"    step {s:>7d}: std={c['rf_std']:.5f}  "
                  f"range=[{c['rf_min']:.4f}, {c['rf_max']:.4f}]")

        analysis[model_key] = {
            'init_rf_mean': init['rf_mean'],
            'init_rf_std': init['rf_std'],
            'final_rf_mean': final['rf_mean'],
            'final_rf_std': final['rf_std'],
            'theory_iid': theory_iid,
            'uniform_floor': uniform_floor,
            'divergence_amplification': final['rf_std'] / max(init['rf_std'], 1e-9),
            'convergence_steps': conv_steps,
            'rho_pos_convstep': rho_pos,
            'p_pos_convstep': p_pos,
        }

    results['analysis'] = analysis
    save_results(results)
    return analysis


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 70)
    print("THEME 4: TRAINING DYNAMICS (Pythia checkpoints, CPU)")
    print("=" * 70)

    if len(sys.argv) > 1:
        keys = sys.argv[1:]
    else:
        keys = ['pythia-160m', 'pythia-410m']

    if 'analysis' in keys:
        run_analysis()
        return

    for key in keys:
        if key not in MODEL_CONFIGS:
            print(f"Unknown: {key}, skipping")
            continue
        try:
            run_model(key, MODEL_CONFIGS[key])
        except Exception as e:
            print(f"\n  ERROR on {key}: {e}")
            import traceback; traceback.print_exc()

    run_analysis()
    print(f"\nDone. Results in results/{RESULT_FILE}")


if __name__ == '__main__':
    main()
