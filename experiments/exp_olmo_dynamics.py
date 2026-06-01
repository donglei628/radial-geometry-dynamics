"""
Second architecture family for the R_F Evolution Law: OLMo-1B-hf (SwiGLU, non-GELU).

Replicates the Pythia training-dynamics protocol on a different architecture family
to lift the "geometry lags loss / second phase" story from n=1 to n=2 families.

For each checkpoint (revision): download to an isolated cache, compute per-layer
R_F (+ signed tr ratio) and validation PPL, then DELETE the cache (peak disk ~5GB).
Random-init floor = the step-0 prediction E[R_F]=1/m on the OLMo architecture.

OLMo-1B-hf earliest public checkpoint is step 1000; the floor is obtained from a
config-only random init (no download).
"""
import os, json, math, shutil, gc
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
TMP = os.path.join(HERE, '_olmo_cache')        # isolated, deleted per ckpt
RESULT = os.path.join(HERE, 'results', 'exp_olmo_dynamics.json')
REPO = 'allenai/OLMo-1B-hf'

CKPTS = [  # (step, branch)
    (1000, 'step1000-tokens4B'), (2000, 'step2000-tokens8B'),
    (3000, 'step3000-tokens12B'), (4000, 'step4000-tokens16B'),
    (5000, 'step5000-tokens20B'), (10000, 'step10000-tokens41B'),
    (20000, 'step20000-tokens83B'), (40000, 'step40000-tokens167B'),
    (80000, 'step80000-tokens335B'), (117850, 'step117850-tokens494B'),
    (330000, 'step330000-tokens1383B'), (640000, 'step640000-tokens2683B'),
    (738000, 'step738000-tokens3094B'),
]


def rf_signed(model):
    rfs, trs = [], []
    for L in model.model.layers:
        W1 = L.mlp.up_proj.weight.detach().float()
        W2 = L.mlp.down_proj.weight.detach().float()
        M = W2 @ W1; m = M.shape[0]
        tr = torch.trace(M).item(); tr2 = torch.trace(M @ M).item(); fro = (M ** 2).sum().item()
        rfs.append((tr**2 + fro + tr2) / ((m + 2) * fro) if fro > 0 else 0.0)
        trs.append(tr / (fro ** 0.5) if fro > 0 else 0.0)
    return rfs, trs


def get_val(tok, n=32, seqlen=512):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    ids = tok('\n\n'.join(ds['text']), return_tensors='pt').input_ids[0]
    return [ids[i*seqlen:i*seqlen+seqlen+1] for i in range(n)
            if (i+1)*seqlen+1 <= ids.numel()]


@torch.no_grad()
def eval_ppl(model, val, device):
    model.eval(); nll = 0.0; ntok = 0
    for ch in val:
        ch = ch.unsqueeze(0).to(device)
        lg = model(ch[:, :-1]).logits
        nll += F.cross_entropy(lg.reshape(-1, lg.size(-1)), ch[0, 1:].reshape(-1),
                               reduction='sum').item()
        ntok += ch.numel() - 1
    return math.exp(nll / ntok)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(REPO)
    val = get_val(tok)
    res = json.load(open(RESULT)) if os.path.exists(RESULT) else {'repo': REPO, 'ckpts': {}}

    # --- random-init floor (= step 0), no download ---
    if 'floor' not in res:
        cfg = AutoConfig.from_pretrained(REPO)
        torch.manual_seed(0)
        rm = AutoModelForCausalLM.from_config(cfg).to(torch.float32)
        rfs, trs = rf_signed(rm)
        m = rm.model.layers[0].mlp.up_proj.weight.shape[1]
        res['floor'] = {'rf_mean': float(np.mean(rfs)), 'rf_std': float(np.std(rfs)),
                        'theory_1_over_m': 1.0 / m, 'm': m, 'rf': rfs}
        print(f"[floor] random-init R_F mean={np.mean(rfs):.6f} std={np.std(rfs):.2e} "
              f"vs 1/m={1/m:.6f} (m={m})", flush=True)
        del rm; gc.collect()
        json.dump(res, open(RESULT, 'w'), indent=2)

    # --- checkpoints: download (retry+resume) -> compute -> delete ---
    import time
    for step, branch in CKPTS:
        if str(step) in res['ckpts']:
            continue
        os.makedirs(TMP, exist_ok=True)
        model = None
        for attempt in range(6):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    REPO, revision=branch, cache_dir=TMP,
                    torch_dtype=torch.float32).to(device).eval()
                break
            except Exception as e:
                print(f"[{step}] attempt {attempt+1}/6 failed: "
                      f"{type(e).__name__}: {str(e)[:80]}", flush=True)
                gc.collect(); torch.cuda.empty_cache()
                time.sleep(8)   # keep partial cache for resume, brief backoff
        if model is None:
            print(f"[{step}] GAVE UP after 6 attempts", flush=True)
            shutil.rmtree(TMP, ignore_errors=True)
            continue
        try:
            rfs, trs = rf_signed(model)
            ppl = eval_ppl(model, val, device)
            res['ckpts'][str(step)] = {'step': step, 'branch': branch,
                                       'rf': rfs, 'signed_tr': trs,
                                       'rf_mean': float(np.mean(rfs)),
                                       'rf_std': float(np.std(rfs)), 'val_ppl': ppl}
            print(f"[{step:>7}] R_F mean={np.mean(rfs):.4f} std={np.std(rfs):.4f}  PPL={ppl:.2f}", flush=True)
            json.dump(res, open(RESULT, 'w'), indent=2)
        finally:
            del model
            gc.collect(); torch.cuda.empty_cache()
            shutil.rmtree(TMP, ignore_errors=True)   # free disk

    print("\nDONE. Saved", RESULT)
    # quick summary
    ck = res['ckpts']
    steps = sorted(int(s) for s in ck)
    print(f"\n{'step':>8}{'R_F std':>10}{'R_F mean':>10}{'PPL':>9}")
    print(f"{'floor(0)':>8}{res['floor']['rf_std']:>10.4f}{res['floor']['rf_mean']:>10.4f}{'-':>9}")
    for s in steps:
        c = ck[str(s)]
        print(f"{s:>8}{c['rf_std']:>10.4f}{c['rf_mean']:>10.4f}{c['val_ppl']:>9.2f}")


if __name__ == '__main__':
    main()
