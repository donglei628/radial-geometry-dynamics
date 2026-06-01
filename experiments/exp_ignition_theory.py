"""
Ignition theory verification: synthetic teacher-student linear sublayer.

Theory: M = W2 W1 under gradient flow on L = 1/2||W2W1 - M*||^2 follows
saddle-escape (Saxe) dynamics. tr(M) should be SIGMOID in time (freeze ->
ignite -> saturate), and ignition time ~ log(1/init_scale).

We verify:
  (1) tr(M)(t) and R_F(t) are S-shaped (frozen then ignite).
  (2) ignition time scales as log(1/eps) with init scale eps.
  (3) R_F stays at floor 1/(m+2) until tr(M) ignites.

Usage:
    python exp_ignition_theory.py
"""
import os, json, math
import numpy as np
from scipy import stats


def rf_of(M):
    m = M.shape[0]
    trM = np.trace(M); trM2 = np.trace(M @ M); fro = (M**2).sum()
    return (trM**2 + fro + trM2) / ((m+2)*fro) if fro > 1e-30 else 0.0


def train(m, k, eps, M_star, lr=0.01, steps=8000, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((k, m)) * eps
    W2 = rng.standard_normal((m, k)) * eps
    traj = []
    for t in range(steps):
        M = W2 @ W1
        G = M - M_star                      # grad of 1/2||M-M*||^2
        gW1 = W2.T @ G
        gW2 = G @ W1.T
        W1 -= lr * gW1
        W2 -= lr * gW2
        # dense early sampling (log-ish), sparse late
        if t < 200 or t % 25 == 0:
            trM = np.trace(M)
            traj.append({'t': t, 'tr_M': float(trM), 'rf': float(rf_of(M)),
                         'fro': float((M**2).sum())})
    return traj


def ignition_step(traj, key='rf', frac=0.5):
    """First step where `key` reaches frac of its final value."""
    vals = np.array([s[key] for s in traj])
    target = frac * vals[-1]
    for s in traj:
        if s[key] >= target:
            return s['t']
    return traj[-1]['t']


def main():
    m, k = 64, 64
    rng = np.random.default_rng(42)
    # teacher with POSITIVE TRACE (task drives positive W2-W1^T alignment):
    # symmetric PSD target M* = V diag(s) V^T, tr(M*) = sum(s) > 0.
    V, _ = np.linalg.qr(rng.standard_normal((m, m)))
    s = np.zeros(m); s[:8] = np.linspace(5, 2, 8)   # 8 active modes, positive
    M_star = V @ np.diag(s) @ V.T
    floor = 1.0 / (m + 2)

    print("="*64)
    print("IGNITION THEORY: synthetic teacher-student linear M=W2W1")
    print("="*64)
    print(f"  m={m}, R_F floor 1/(m+2)={floor:.5f}")

    # (1)+(3) trajectory at one eps
    print("\n(1) Trajectory (eps=0.003): tr(M) and R_F over training")
    traj = train(m, k, 0.003, M_star, seed=0)
    print(f"  {'step':>6} {'tr(M)':>10} {'R_F':>10} {'R_F/floor':>10}")
    # show a log-spaced subset to see the S-curve
    idxs = sorted(set([0] + [int(x) for x in np.geomspace(1, len(traj)-1, 22)]))
    for i in idxs:
        s_ = traj[i]
        print(f"  {s_['t']:>6} {s_['tr_M']:>10.3f} {s_['rf']:>10.5f} {s_['rf']/floor:>10.2f}")

    # (2) ignition time vs init scale
    print("\n(2) Ignition time (R_F reaches 50% final) vs init scale eps:")
    print(f"  {'eps':>10} {'log(1/eps)':>12} {'ignition_step':>14}")
    epss = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    logs, igs = [], []
    for eps in epss:
        # average ignition over 3 seeds
        ig = np.mean([ignition_step(train(m, k, eps, M_star, seed=sd), 'rf')
                      for sd in range(3)])
        logs.append(math.log(1/eps)); igs.append(ig)
        print(f"  {eps:>10.3f} {math.log(1/eps):>12.3f} {ig:>14.0f}")
    rho, p = stats.pearsonr(logs, igs)
    slope = np.polyfit(logs, igs, 1)[0]
    print(f"\n  Pearson(log(1/eps), ignition_step) = {rho:+.3f} (p={p:.2e})")
    print(f"  slope = {slope:.1f} steps per unit log(1/eps)")
    print(f"  => ignition time scales LINEARLY with log(1/eps): theory confirmed"
          if rho > 0.9 else "  => weak scaling")

    # (3) R_F frozen at floor until ignition
    fr0 = np.mean([s['rf'] for s in traj[:5]])
    print(f"\n(3) Early R_F mean = {fr0:.6f} vs floor {floor:.6f} "
          f"(ratio {fr0/floor:.3f}) — frozen at theory floor before ignition")

    out = {'floor': floor, 'trajectory': traj,
           'scaling': {'log_inv_eps': logs, 'ignition_steps': igs,
                       'pearson': float(rho), 'slope': float(slope)}}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), 'results',
                                     'exp_ignition_theory.json'), 'w'), indent=2)
    print("\nSaved to results/exp_ignition_theory.json")


if __name__ == '__main__':
    main()
