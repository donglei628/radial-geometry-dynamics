"""
Verify the sign mechanism behind 10.06 (R_F vs angular distance).

Per MLP layer, over the REAL post-LN input distribution x_hat, compare:
  cos_real   = cos(x_hat, MLP(x_hat))          # real nonlinear sublayer
  cos_linear = cos(x_hat, M @ x_hat)           # linearized M = W_down @ W_up
  tr_M, R_F                                     # weight-space radial structure

Hypothesis: high-R_F layers have tr_M>0 and cos_linear>0 (weight-space radial,
near +c*I), but cos_real<0 (activation inverts the functional radial sign).
"""
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

MODELS = {
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B',
    'tinyllama':    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
}


def get_data(tok, n_seq=32, seqlen=512):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(ds['text'])
    ids = tok(text, return_tensors='pt').input_ids[0]
    chunks = []
    for i in range(n_seq):
        s = i * seqlen
        if s + seqlen + 1 > ids.numel():
            break
        chunks.append(ids[s:s + seqlen + 1])
    return chunks


@torch.no_grad()
def run(key, name, device='cuda'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n{'#'*68}\n# {key}\n{'#'*68}", flush=True)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, dtype=torch.float32, device_map=device)
    model.eval()
    layers = model.model.layers
    n = len(layers)
    chunks = get_data(tok)

    # Precompute M = W_down @ W_up and tr_M, R_F per layer
    trM = np.zeros(n); rf = np.zeros(n)
    Ms = []
    for li, layer in enumerate(layers):
        W1 = layer.mlp.up_proj.weight.detach().float()   # [inter, hid]
        W2 = layer.mlp.down_proj.weight.detach().float()  # [hid, inter]
        M = W2 @ W1                                        # [hid, hid]
        Ms.append(M)
        m = M.shape[0]
        tr_M = torch.trace(M).item()
        tr_M2 = torch.trace(M @ M).item()
        fro = (M ** 2).sum().item()
        trM[li] = tr_M
        rf[li] = (tr_M**2 + fro + tr_M2) / ((m + 2) * fro) if fro > 0 else 0.0

    cos_real = np.zeros(n); cos_lin = np.zeros(n); cnt = 0
    store = {}

    def mk_hook(li):
        def hook(module, inp, out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])   # [N, hid]
            y = out.detach().float().reshape(-1, out.shape[-1])         # [N, hid]
            yl = x @ Ms[li].T                                            # [N, hid] linear M x
            cr = F.cosine_similarity(x, y, dim=-1).mean().item()
            cl = F.cosine_similarity(x, yl, dim=-1).mean().item()
            store[li] = (cr, cl)
        return hook

    handles = [layer.mlp.register_forward_hook(mk_hook(li))
               for li, layer in enumerate(layers)]
    for ch in chunks:
        ids = ch[:-1].unsqueeze(0).to(device)
        store.clear()
        model(ids)
        for li in range(n):
            cr, cl = store[li]
            cos_real[li] += cr; cos_lin[li] += cl
        cnt += 1
    for h in handles:
        h.remove()
    cos_real /= cnt; cos_lin /= cnt

    order = np.argsort(-rf)
    print(f"\n{'layer':>5} {'R_F':>7} {'tr_M':>9} {'cos_lin':>8} {'cos_real':>9}")
    for i in order:
        print(f"{i:>5} {rf[i]:>7.3f} {trM[i]:>9.1f} {cos_lin[i]:>8.3f} {cos_real[i]:>9.3f}")

    top = order[:6]; bot = order[-6:]
    print(f"\nTOP-6 R_F : tr_M>0={(trM[top]>0).all()}  "
          f"cos_lin mean={cos_lin[top].mean():+.3f}  cos_real mean={cos_real[top].mean():+.3f}")
    print(f"BOT-6 R_F : cos_lin mean={cos_lin[bot].mean():+.3f}  cos_real mean={cos_real[bot].mean():+.3f}")
    rho_r, p_r = stats.spearmanr(rf, 1 - cos_real)
    rho_l, p_l = stats.spearmanr(rf, 1 - cos_lin)
    print(f"\nSpearman R_F vs angular_real(1-cos_real): {rho_r:+.3f} (p={p_r:.1e})")
    print(f"Spearman R_F vs angular_lin (1-cos_lin) : {rho_l:+.3f} (p={p_l:.1e})")
    n_inv = int(((cos_lin > 0) & (cos_real < 0)).sum())
    print(f"Layers with cos_lin>0 BUT cos_real<0 (sign inversion): {n_inv}/{n}")

    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    return {'rf': rf.tolist(), 'tr_M': trM.tolist(),
            'cos_real': cos_real.tolist(), 'cos_lin': cos_lin.tolist()}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    keys = [a for a in sys.argv[1:] if a in MODELS] or list(MODELS)
    out = {}
    for k in keys:
        out[k] = run(k, MODELS[k], device)
    fp = os.path.join(os.path.dirname(__file__), 'results', 'verify_angular_sign.json')
    with open(fp, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {fp}")


if __name__ == '__main__':
    main()
