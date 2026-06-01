"""
Lemma 2 verification (improved): saddle-escape ignition of tr(M) and R_F.

Synthetic teacher-student linear sublayer M = W2 W1, gradient flow on
L = 1/2 ||M - M*||_F^2 with a RADIAL teacher M* = s I (the component R_F sees).

Derivation to verify:
  - Under (approximately) balanced init, the radial mode a(t) := tr(M)/m obeys the
    logistic ODE  da/dt = 2 a (s - a)  in gradient-flow time, hence
        a(t) = s / (1 + (s/a0 - 1) e^{-2 s t}),
    a frozen -> ignite -> saturate sigmoid.
  - Half-saturation (ignition) time  t* = (1/2s) ln(s/a0 - 1) ~ (1/s) ln(1/eps),
    LINEAR in ln(1/eps); in discrete steps with lr eta, slope d(step*)/d ln(1/eps) ~ 1/(eta s).
  - R_F stays at the init floor 1/m until tr(M) ignites, then rises (Lemma 3).

Fixes vs the old run: small lr (gradient-flow regime), per-step sampling, radial
teacher, ignition measured on tr(M) saturation (not a 50%-of-noisy-final on R_F).
"""
import os, json, math
import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(__file__), 'results', 'exp_lemma2_ignition.json')


def rf_of(M):
    m = M.shape[0]
    trM = np.trace(M); trM2 = np.trace(M @ M); fro = (M ** 2).sum()
    return (trM ** 2 + fro + trM2) / ((m + 2) * fro) if fro > 1e-30 else 0.0


def train(m, k, eps, s, eta=2e-3, steps=20000, seed=0, sample_every=1):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((k, m)) * eps
    W2 = rng.standard_normal((m, k)) * eps
    sI = s * np.eye(m)
    traj = []
    for t in range(steps):
        M = W2 @ W1
        G = M - sI
        W1n = W1 - eta * (W2.T @ G)
        W2n = W2 - eta * (G @ W1.T)
        W1, W2 = W1n, W2n
        if t % sample_every == 0 or t == steps - 1:
            traj.append((t, float(np.trace(M)), float(rf_of(M)), float((M ** 2).sum())))
    return traj


def ignition_step(traj, m, s, frac=0.5):
    """First step where tr(M) reaches frac of its radial saturation m*s."""
    target = frac * m * s
    for (t, trM, rf, fro) in traj:
        if trM >= target:
            return t
    return traj[-1][0]


def main():
    m, k, s = 64, 64, 4.0
    floor = 1.0 / m
    print("=" * 66)
    print("LEMMA 2 IGNITION (radial teacher M*=sI, gradient-flow regime)")
    print(f"  m={m}, k={k}, s={s}, R_F init floor 1/m={floor:.5f}")
    print("=" * 66)

    # (1) one trajectory: show S-curve and fit the closed-form sigmoid
    eta = 2e-3
    traj = train(m, k, 0.01, s, eta=eta, seed=0)
    a = np.array([tr / m for (_, tr, _, _) in traj])         # radial mode tr(M)/m
    a0 = a[0]
    print(f"\n(1) Trajectory (eps=0.01). radial mode a=tr(M)/m, a0={a0:.2e}, saturates at s={s}")
    print(f"  {'step':>7} {'tr(M)':>10} {'a/s':>7} {'R_F':>9} {'R_F/floor':>10}")
    idxs = sorted(set([0] + [int(x) for x in np.geomspace(1, len(traj) - 1, 16)]))
    for i in idxs:
        t, tr, rf, fro = traj[i]
        print(f"  {t:>7} {tr:>10.3f} {tr/(m*s):>7.3f} {rf:>9.5f} {rf/floor:>10.2f}")
    # closed-form sigmoid fit a(t)=s/(1+(s/a0-1)e^{-2 s tau}), tau = eta*step (flow time)
    steps_arr = np.array([t for (t, _, _, _) in traj], dtype=float)
    tau = eta * steps_arr
    a_pred = s / (1 + (s / max(a0, 1e-12) - 1) * np.exp(-2 * s * tau))
    rel_err = np.nanmean(np.abs(a_pred - a) / (np.abs(a) + 1e-6))
    print(f"  closed-form sigmoid mean rel-err = {rel_err:.3f}  (a(t)=s/(1+(s/a0-1)e^-2s tau), tau=eta*step)")

    # (2) ignition time vs ln(1/eps): expect LINEAR, slope ~ 1/(eta s)
    print("\n(2) Ignition step (tr(M) reaches 50% of m*s) vs ln(1/eps):")
    print(f"  {'eps':>9} {'ln(1/eps)':>10} {'ign_step':>9}")
    epss = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4]
    logs, igs = [], []
    for eps in epss:
        ig = np.mean([ignition_step(train(m, k, eps, s, eta=eta, seed=sd), m, s)
                      for sd in range(3)])
        logs.append(math.log(1 / eps)); igs.append(ig)
        print(f"  {eps:>9.4f} {math.log(1/eps):>10.3f} {ig:>9.0f}")
    logs, igs = np.array(logs), np.array(igs)
    rho, p = stats.pearsonr(logs, igs)
    slope, intercept = np.polyfit(logs, igs, 1)
    r2 = rho ** 2
    print(f"\n  Pearson(ln(1/eps), ign_step) = {rho:+.4f}  R^2={r2:.4f}")
    print(f"  measured slope = {slope:.0f} steps / unit ln(1/eps)")
    print(f"  predicted slope 1/(eta*s) = {1/(eta*s):.0f}  (ratio {slope/(1/(eta*s)):.2f})")

    # (3) frozen-at-floor before ignition
    pre = [rf for (_, _, rf, _) in traj[:50]]
    print(f"\n(3) pre-ignition R_F mean = {np.mean(pre):.6f} vs floor {floor:.6f} "
          f"(ratio {np.mean(pre)/floor:.3f}) -- frozen at floor")

    json.dump({'m': m, 'k': k, 's': s, 'eta': eta, 'floor': floor,
               'trajectory': [list(x) for x in traj],
               'sigmoid_rel_err': float(rel_err),
               'scaling': {'ln_inv_eps': logs.tolist(), 'ign_steps': igs.tolist(),
                           'pearson': float(rho), 'r2': float(r2),
                           'slope': float(slope),
                           'slope_pred': float(1/(eta*s))}}, open(OUT, 'w'), indent=2)
    print(f"\nSaved {OUT}")


if __name__ == '__main__':
    main()
