"""
Ignition mechanism verification: is R_F differentiation driven by the growth
of tr(M) = <W2, W1^T> alignment?

R_F is tr(M)^2-dominated. tr(M) = <W2, W1^T>_F is the alignment between W2 and
W1 transpose. At init they are ~orthogonal (tr(M)~0 => R_F~floor). "Ignition"
should coincide with tr(M) alignment growing.

Uses existing exp_batch_training_dynamics.json (has tr_M, fro_sq, signed_tr_ratio
per layer per checkpoint).

Usage:
    python exp_ignition_mechanism.py
"""
import os, json, math
import numpy as np
from scipy import stats

RES = os.path.join(os.path.dirname(__file__), 'results')


def main():
    d = json.load(open(os.path.join(RES, 'exp_batch_training_dynamics.json')))
    print("=" * 72)
    print("IGNITION MECHANISM: R_F differentiation vs tr(M)=<W2,W1^T> alignment")
    print("=" * 72)

    for mk in ['pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-70m']:
        if mk not in d:
            continue
        ck = d[mk]['checkpoints']
        steps = sorted(int(s) for s in ck.keys())
        m = d[mk]['m']
        print(f"\n{'='*60}\n{mk} (m={m})\n{'='*60}")

        # For each checkpoint: mean |signed_tr_ratio| across layers (alignment),
        # and R_F std (differentiation). signed_tr_ratio = tr(M)/||M||_F.
        print(f"  {'step':>7} {'mean|tr/||M|||':>14} {'R_F std':>10} {'R_F mean':>10}")
        align, rfstd, rfmean = [], [], []
        for s in steps:
            layers = ck[str(s)]['layers']
            str_ratio = [abs(l['signed_tr_ratio']) for l in layers]
            rfs = ck[str(s)]['rf_per_layer']
            a = float(np.mean(str_ratio))
            align.append(a); rfstd.append(ck[str(s)]['rf_std']); rfmean.append(np.mean(rfs))
            print(f"  {s:>7} {a:>14.4f} {ck[str(s)]['rf_std']:>10.5f} {np.mean(rfs):>10.5f}")

        # correlation: does alignment growth track R_F mean growth?
        rho, p = stats.spearmanr(align, rfmean)
        print(f"\n  ρ(mean|tr/||M|||, mean R_F) across training = {rho:+.3f} (p={p:.2e})")

        # ignition timing: where does alignment / R_F jump most (log-step)
        align = np.array(align); rfmean = np.array(rfmean)
        d_align = np.diff(align); d_rf = np.diff(rfmean)
        ig_a = steps[1:][int(np.argmax(d_align))]
        ig_r = steps[1:][int(np.argmax(d_rf))]
        print(f"  max alignment jump at step {ig_a}; max R_F jump at step {ig_r}")

        # per-layer: does final R_F rank match final alignment rank?
        final = ck[str(steps[-1])]
        fa = [abs(l['signed_tr_ratio']) for l in final['layers']]
        fr = final['rf_per_layer']
        rho_l, p_l = stats.spearmanr(fa, fr)
        print(f"  per-layer final: ρ(|tr/||M|||, R_F) = {rho_l:+.3f} (p={p_l:.2e})")


if __name__ == '__main__':
    main()
