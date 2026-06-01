"""
Sign-flip attribution (minimal set) — spec: exp_sign_flip_attribution_spec.

Goal: LOCALIZE (not discover) the functional radial sign-flip from the linear
skeleton M=W_down*W_up (positive) to the full nonlinear MLP (negative), to the
<activation / gate / interaction> step, and RANK its per-layer strength by R_F.

Arms per layer, per real input x_hat (= Norm(h) fed to the MLP):
  SwiGLU (Qwen/Llama):
    S0  = W_down( W_up x )                          # linear proxy (R_F's object)
    S1  = W_down( (W_gate x) (.) (W_up x) )         # +gate, no activation (bilinear)
    S1p = W_down( SiLU(W_up x) )                    # +activation on up-path, no gate
    S2  = W_down( SiLU(W_gate x) (.) (W_up x) )     # full real MLP
  GELU (Pythia/GPT2): S0 = Wd(Wu x), S2 = Wd(act(Wu x))   (no gate)

Metric: signed functional radial cos(x_hat, y_arm), per token, mean over tokens.
Self-check: ||S2 - real_mlp_out|| / ||real_mlp_out|| < 1e-3  (rules out hook bug).
Sign-flip: sign(cos_layer^arm) != sign(cos_layer^S0), both |cos| >= tau_sign.
All math in fp32.
"""
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import ensure_output_dir

RESULT_FILE = 'exp_sign_flip_attribution.json'
TAU_SIGN = 0.02

MODELS = {
    'qwen2.5-0.5b': ('Qwen/Qwen2.5-0.5B', 'swiglu'),
    'tinyllama':    ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'swiglu'),
    'pythia-410m':  ('EleutherAI/pythia-410m', 'gelu'),
}


def get_data(tok, n_seq=64, seqlen=512):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(ds['text'])
    ids = tok(text, return_tensors='pt').input_ids[0]
    chunks = []
    for i in range(n_seq):
        s = i * seqlen
        if s + seqlen > ids.numel():
            break
        chunks.append(ids[s:s + seqlen])
    return chunks


def get_layers_mlp(model, arch):
    if arch == 'swiglu':
        return [(l, l.mlp) for l in model.model.layers]
    elif arch == 'gelu':   # pythia / gpt-neox
        return [(l, l.mlp) for l in model.gpt_neox.layers]


def rf_per_layer(mlps, arch):
    rfs, trs = [], []
    for mlp in mlps:
        if arch == 'swiglu':
            W1 = mlp.up_proj.weight.detach().float()
            W2 = mlp.down_proj.weight.detach().float()
        else:
            W1 = mlp.dense_h_to_4h.weight.detach().float()
            W2 = mlp.dense_4h_to_h.weight.detach().float()
        M = W2 @ W1
        m = M.shape[0]
        trM = torch.trace(M).item(); trM2 = torch.trace(M @ M).item()
        fro = (M ** 2).sum().item()
        rfs.append((trM**2 + fro + trM2) / ((m + 2) * fro) if fro > 0 else 0.0)
        trs.append(trM / (fro ** 0.5) if fro > 0 else 0.0)
    return rfs, trs


def cos_mean(x, y):
    return F.cosine_similarity(x.float(), y.float(), dim=-1).reshape(-1)


@torch.no_grad()
def run_model(key, name, arch, device='cuda'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n{'#'*66}\n# {key}  ({arch})\n{'#'*66}", flush=True)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, dtype=torch.float32, device_map=device).eval()
    pairs = get_layers_mlp(model, arch)
    mlps = [mlp for (_, mlp) in pairs]
    n = len(mlps)
    rfs, trs = rf_per_layer(mlps, arch)
    chunks = get_data(tok)
    print(f"  {len(chunks)} seqs x {chunks[0].numel()} tok, {n} layers")

    arms = ['S0', 'S1', 'S1p', 'S2'] if arch == 'swiglu' else ['S0', 'S2']
    acc = {a: np.zeros(n) for a in arms}      # sum of per-token cos
    cnt = np.zeros(n)
    selfcheck = np.zeros(n)

    store = {}

    def mk_hook(li, mlp):
        def hook(module, inp, out):
            x = inp[0].float()
            if arch == 'swiglu':
                hu = mlp.up_proj(x); hg = mlp.gate_proj(x); act = mlp.act_fn
                ys = {'S0': mlp.down_proj(hu),
                      'S1': mlp.down_proj(hg * hu),
                      'S1p': mlp.down_proj(act(hu)),
                      'S2': mlp.down_proj(act(hg) * hu)}
            else:
                hu = mlp.dense_h_to_4h(x); act = mlp.act
                ys = {'S0': mlp.dense_4h_to_h(hu),
                      'S2': mlp.dense_4h_to_h(act(hu))}
            xc = x.reshape(-1, x.shape[-1])
            cs = {a: cos_mean(xc, ys[a].reshape(-1, ys[a].shape[-1])) for a in arms}
            # self-check S2 vs real out
            err = (ys['S2'] - out.float()).norm() / (out.float().norm() + 1e-9)
            store[li] = (cs, float(err), xc.shape[0])
        return hook

    handles = [mlp.register_forward_hook(mk_hook(li, mlp))
               for li, (_, mlp) in enumerate(pairs)]
    for ch in chunks:
        ids = ch.unsqueeze(0).to(device)
        store.clear()
        model(ids)
        for li in range(n):
            cs, err, ntok = store[li]
            for a in arms:
                acc[a][li] += cs[a].sum().item()
            cnt[li] += ntok
            selfcheck[li] = max(selfcheck[li], err)
    for h in handles:
        h.remove()

    cos_layer = {a: (acc[a] / cnt) for a in arms}
    max_err = float(selfcheck.max())
    print(f"  HOOK SELF-CHECK max ||S2-real||/||real|| = {max_err:.2e}  "
          f"{'OK' if max_err < 1e-3 else 'FAIL!!'}")

    rf = np.array(rfs)
    out = {'arch': arch, 'rf': rfs, 'signed_tr_ratio': trs,
           'cos_layer': {a: cos_layer[a].tolist() for a in arms},
           'self_check_max_relerr': max_err, 'n_layers': n}

    print(f"\n  arm   cos(mean)  ρ(R_F,1-cos)   #sign-flip-vs-S0 (|cos|>={TAU_SIGN})")
    s0 = cos_layer['S0']
    for a in arms:
        c = cos_layer[a]
        rho, p = stats.spearmanr(rf, 1 - c)
        flips = int(np.sum((np.sign(c) != np.sign(s0)) &
                           (np.abs(c) >= TAU_SIGN) & (np.abs(s0) >= TAU_SIGN)))
        out[f'rho_{a}'] = {'rho': float(rho), 'p': float(p)}
        out[f'nflip_{a}'] = flips
        print(f"  {a:<4}  {c.mean():+8.3f}   {rho:+6.3f} (p={p:.1e})   {flips}/{n}")

    # high-R_F vs low-R_F mean cos per arm (does the flip concentrate in high R_F?)
    order = np.argsort(-rf)
    hi, lo = order[:max(3, n//4)], order[-max(3, n//4):]
    print(f"\n  high-R_F vs low-R_F mean cos:")
    for a in arms:
        c = cos_layer[a]
        print(f"    {a:<4} high={c[hi].mean():+.3f}  low={c[lo].mean():+.3f}")
        out[f'cos_hi_{a}'] = float(c[hi].mean()); out[f'cos_lo_{a}'] = float(c[lo].mean())

    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    keys = [a for a in sys.argv[1:] if a in MODELS] or list(MODELS)
    fp = os.path.join(ensure_output_dir('results'), RESULT_FILE)
    results = json.load(open(fp)) if os.path.exists(fp) else {}
    for key in keys:
        name, arch = MODELS[key]
        try:
            results[key] = run_model(key, name, arch, device)
            json.dump(results, open(fp, 'w'), indent=2)
        except Exception as e:
            print(f"  ERROR {key}: {e}")
            import traceback; traceback.print_exc()
    print(f"\nSaved {fp}")


if __name__ == '__main__':
    main()
