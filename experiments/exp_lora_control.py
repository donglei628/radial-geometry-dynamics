"""
§5 LoRA layer-selection CONTROL experiment (rigorous version of 6.03).

Question: does putting LoRA on the LOW-R_F layers (predicted "most plastic")
beat the baselines that matter -- RANDOM layers and LAST-N layers -- at equal
trainable-parameter budget? The old 6.03 lacked random/last baselines and had no
seeds; and here low-R_F happens to overlap late layers, so the last-N control is
essential to rule out "R_F is just a depth proxy".

Design (Qwen2.5-0.5B, 24 layers, WikiText-2):
  - budget: adapt N_SEL=8 MLP layers, rank=8 -> identical trainable params for all.
  - conditions: low_rf, high_rf, last8, first8  (each over SEEDS training seeds)
                random  (RAND_SETS distinct random 8-layer sets, 1 seed each)
  - metric: validation PPL, mean +/- std. Base (no-FT) PPL reported for reference.
"""
import sys, os, json, math, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import ensure_output_dir

NAME = 'Qwen/Qwen2.5-0.5B'
RESULT = 'exp_lora_control.json'
N_SEL = 8
RANK = 8
STEPS = 400
BS = 4
SEQ = 256
LR = 2e-4
SEEDS = 3          # training seeds for deterministic selections
RAND_SETS = 6      # distinct random layer sets


def rf_per_layer(model):
    rfs = []
    for L in model.model.layers:
        W1 = L.mlp.up_proj.weight.detach().float()
        W2 = L.mlp.down_proj.weight.detach().float()
        M = W2 @ W1; m = M.shape[0]
        tr = torch.trace(M).item(); tr2 = torch.trace(M @ M).item(); fro = (M**2).sum().item()
        rfs.append((tr**2 + fro + tr2) / ((m + 2) * fro) if fro > 0 else 0.0)
    return rfs


def get_data(tok, split, n):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    ids = tok('\n\n'.join(ds['text']), return_tensors='pt').input_ids[0]
    chunks = []
    for i in range(n):
        s = i * SEQ
        if s + SEQ + 1 > ids.numel():
            break
        chunks.append(ids[s:s + SEQ + 1])
    return chunks


def eval_ppl(model, val, device):
    model.eval(); nll = 0.0; ntok = 0
    with torch.no_grad():
        for ch in val:
            ch = ch.unsqueeze(0).to(device)
            lg = model(ch[:, :-1]).logits
            nll += F.cross_entropy(lg.reshape(-1, lg.size(-1)), ch[0, 1:].reshape(-1),
                                   reduction='sum').item()
            ntok += SEQ
    return math.exp(nll / ntok)


def run_once(layers_idx, seed, train, val, device):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        NAME, trust_remote_code=True, dtype=torch.float16, device_map=device)
    cfg = LoraConfig(r=RANK, lora_alpha=2 * RANK,
                     target_modules=['up_proj', 'down_proj', 'gate_proj'],
                     layers_to_transform=list(layers_idx), lora_dropout=0.0,
                     bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, cfg); model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    rng = np.random.default_rng(seed)
    for _ in range(STEPS):
        batch = [train[i] for i in rng.integers(0, len(train), BS)]
        x = torch.stack(batch).to(device)
        lg = model(x[:, :-1]).logits
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    ppl = eval_ppl(model, val, device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    return ppl, npar / 1e6


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(NAME, trust_remote_code=True)
    probe = AutoModelForCausalLM.from_pretrained(NAME, trust_remote_code=True,
                                                 dtype=torch.float16, device_map=device)
    rf = rf_per_layer(probe); n = len(rf)
    base_val = None
    train = get_data(tok, 'train', 2000)
    val = get_data(tok, 'validation', 64)
    base_val = eval_ppl(probe, val, device)
    del probe
    import gc; gc.collect(); torch.cuda.empty_cache()

    order = sorted(range(n), key=lambda i: rf[i])
    sel = {
        'low_rf':  sorted(order[:N_SEL]),
        'high_rf': sorted(order[-N_SEL:]),
        'last8':   list(range(n - N_SEL, n)),
        'first8':  list(range(N_SEL)),
    }
    print(f"Base (no-FT) val PPL = {base_val:.3f}   |  n_layers={n}, budget={N_SEL} layers, rank={RANK}")
    for k, v in sel.items():
        print(f"  {k:<8}: {v}")
    rng = np.random.default_rng(123)
    rand_sets = [sorted(rng.choice(n, N_SEL, replace=False).tolist()) for _ in range(RAND_SETS)]
    print(f"  random : {rand_sets}")
    print(f"  train={len(train)} val={len(val)} chunks, STEPS={STEPS}\n")

    fp = os.path.join(ensure_output_dir('results'), RESULT)
    res = json.load(open(fp)) if os.path.exists(fp) else {}
    res.update({'rf': rf, 'base_val_ppl': base_val, 'selections': sel,
                'rand_sets': rand_sets, 'config': {'N_SEL': N_SEL, 'RANK': RANK,
                'STEPS': STEPS, 'BS': BS, 'SEQ': SEQ, 'LR': LR}})

    # deterministic selections: multiple training seeds
    for cname, lidx in sel.items():
        key = f'ppls_{cname}'
        if key not in res:
            res[key] = []
        for sd in range(len(res[key]), SEEDS):
            t0 = time.time()
            ppl, npar = run_once(lidx, sd, train, val, device)
            res[key].append(ppl)
            print(f"  {cname:<8} seed{sd}  val_PPL={ppl:.3f}  ({npar:.2f}M, {time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(fp, 'w'), indent=2)

    # random sets: one seed each
    if 'ppls_random' not in res:
        res['ppls_random'] = []
    for j in range(len(res['ppls_random']), RAND_SETS):
        t0 = time.time()
        ppl, npar = run_once(rand_sets[j], 100 + j, train, val, device)
        res['ppls_random'].append(ppl)
        print(f"  random   set{j}  val_PPL={ppl:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(fp, 'w'), indent=2)

    # summary
    print("\n" + "=" * 60 + f"\nSUMMARY  (base no-FT = {base_val:.3f})\n" + "=" * 60)
    print(f"  {'condition':<10}{'mean':>8}{'std':>7}{'min':>8}{'n':>4}")
    rows = {}
    for cname in ['low_rf', 'last8', 'random', 'first8', 'high_rf']:
        v = np.array(res[f'ppls_{cname}'])
        rows[cname] = v
        print(f"  {cname:<10}{v.mean():>8.3f}{v.std():>7.3f}{v.min():>8.3f}{len(v):>4}")
    # key comparisons
    lo = rows['low_rf']
    print("\n  Key comparisons (lower PPL = better):")
    print(f"    low_rf {lo.mean():.3f} vs random {rows['random'].mean():.3f}  "
          f"(delta {lo.mean()-rows['random'].mean():+.3f})")
    print(f"    low_rf {lo.mean():.3f} vs last8  {rows['last8'].mean():.3f}  "
          f"(delta {lo.mean()-rows['last8'].mean():+.3f})")
    print(f"    low_rf {lo.mean():.3f} vs high_rf {rows['high_rf'].mean():.3f}  "
          f"(delta {lo.mean()-rows['high_rf'].mean():+.3f})")
    json.dump(res, open(fp, 'w'), indent=2)
    print(f"\nSaved {fp}")


if __name__ == '__main__':
    main()
