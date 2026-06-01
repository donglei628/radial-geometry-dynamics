"""
4.05 - R_F change rate vs loss decrease (Pythia-160m checkpoints).

Re-loads key Pythia-160m checkpoints, computes WikiText PPL (proxy for training
loss), and aligns the R_F "ignition" (cross-layer std surge) with the loss
curve. Tests whether R_F differentiation happens DURING rapid loss drop.

CPU. R_F std read from exp_batch_training_dynamics.json (already computed).

Usage:
    python exp_4_05_loss.py
"""

import sys, os, json, math
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import ensure_output_dir

MODEL = 'EleutherAI/pythia-160m'
STEPS = [0, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]


def ppl(model, ids, device, n_chunks=8, seqlen=512):
    import torch.nn.functional as F
    nlls = 0.0; ntok = 0
    for i in range(n_chunks):
        s = i * seqlen
        if s + seqlen + 1 > ids.numel():
            break
        chunk = ids[s:s+seqlen+1].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(chunk[:, :-1]).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               chunk[0, 1:].reshape(-1), reduction='sum')
        nlls += loss.item(); ntok += seqlen
    return math.exp(nlls / ntok)


def main():
    from transformers import GPTNeoXForCausalLM, AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # R_F std per step from training dynamics
    td = json.load(open(os.path.join(ensure_output_dir('results'),
                                      'exp_batch_training_dynamics.json')))
    ckpts = td['pythia-160m']['checkpoints']
    rf_std = {int(s): ckpts[str(s)]['rf_std'] for s in STEPS if str(s) in ckpts}

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok('\n\n'.join(
        __import__('datasets').load_dataset('wikitext', 'wikitext-2-raw-v1',
                                             split='test')['text']),
        return_tensors='pt').input_ids[0]

    rows = []
    for step in STEPS:
        print(f"  [step {step}] loading + PPL...", flush=True)
        try:
            model = GPTNeoXForCausalLM.from_pretrained(
                MODEL, revision=f"step{step}", torch_dtype=torch.float32).to(device)
            model.eval()
        except Exception as e:
            print(f"    FAILED: {e}")
            continue
        p = ppl(model, ids, device)
        rows.append({'step': step, 'ppl': p, 'rf_std': rf_std.get(step)})
        print(f"    PPL={p:.2f}  R_F std={rf_std.get(step)}", flush=True)
        del model
        import gc; gc.collect(); torch.cuda.empty_cache()

    print(f"\n  {'step':>8} {'PPL':>10} {'R_F std':>10}")
    for r in rows:
        print(f"  {r['step']:>8} {r['ppl']:>10.2f} "
              f"{(r['rf_std'] if r['rf_std'] is not None else float('nan')):>10.5f}")

    # Correlate R_F std with -log(PPL) progress
    valid = [r for r in rows if r['rf_std'] is not None]
    steps = np.array([r['step'] for r in valid])
    ppls = np.array([r['ppl'] for r in valid])
    stds = np.array([r['rf_std'] for r in valid])
    # loss proxy = log PPL; "progress" = how much loss dropped from start
    logppl = np.log(ppls)
    rho, p = stats.spearmanr(stds, -logppl)
    print(f"\n  ρ(R_F std, -logPPL i.e. loss-progress) = {rho:+.3f} (p={p:.2e})")
    # ignition window: where does most R_F std appear vs most loss drop?
    if len(valid) > 2:
        d_std = np.diff(stds)
        d_loss = -np.diff(logppl)
        ig_std = steps[1:][np.argmax(d_std)]
        ig_loss = steps[1:][np.argmax(d_loss)]
        print(f"  Max R_F-std jump at step {ig_std}; max loss-drop at step {ig_loss}")

    out = {'rows': rows, 'rho_std_lossprogress': {'rho': float(rho), 'p': float(p)}}
    fp = os.path.join(ensure_output_dir('results'), 'exp_4_05_loss.json')
    with open(fp, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {fp}")


if __name__ == '__main__':
    main()
