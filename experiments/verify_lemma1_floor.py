"""
Verify Lemma 1 (Initialization floor) numerically.

Setup: W1 in R^{k x m}, W2 in R^{m x k}, iid entries mean 0 var s1^2, s2^2,
independent. M = W2 W1, sigma^2 = s1^2 s2^2.

Claims to verify:
  (i)   E[tr(M)] = 0
        E[tr(M)^2] = E[tr(M^2)] = m k sigma^2 ;  E[||M||_F^2] = m^2 k sigma^2
  (ii)  radial ratio tr(M)/||M||_F has RMS = 1/sqrt(m)  (O_p(1/sqrt m))
  (iii) E[R_F] = 1/m + O(1/m^2)  (k-independent);  1/(m+2) is the structural floor
  (iv)  concentration: std(R_F)/E[R_F] -> 0  (explains "layer std ~ 0" at init)
  (v)   distribution-free: holds for non-Gaussian entries (uniform), needs only finite 4th moment
"""
import numpy as np

rng = np.random.default_rng(0)


def sample_stats(m, k, trials=2000, s1=1.0, s2=1.0, dist='normal'):
    sig2 = s1 * s1 * s2 * s2
    trM, trM_sq, trM2, fro, rf, radial = [], [], [], [], [], []
    for _ in range(trials):
        if dist == 'normal':
            W1 = rng.normal(0, s1, (k, m)); W2 = rng.normal(0, s2, (m, k))
        else:  # uniform with matching variance: U[-a,a], var=a^2/3 -> a=sqrt(3)*s
            a1, a2 = np.sqrt(3) * s1, np.sqrt(3) * s2
            W1 = rng.uniform(-a1, a1, (k, m)); W2 = rng.uniform(-a2, a2, (m, k))
        M = W2 @ W1
        t = np.trace(M); t2 = np.trace(M @ M); f = (M * M).sum()
        trM.append(t); trM_sq.append(t * t); trM2.append(t2); fro.append(f)
        rf.append((t * t + t2 + f) / ((m + 2) * f))
        radial.append(t / np.sqrt(f))
    return dict(
        E_trM=np.mean(trM), E_trM2_sq=np.mean(trM_sq), E_trM2=np.mean(trM2),
        E_fro=np.mean(fro), E_rf=np.mean(rf), std_rf=np.std(rf),
        rms_radial=np.sqrt(np.mean(np.square(radial))), sig2=sig2)


print("(i)/(iii) moment identities and floor  [normal init]")
print(f"{'m':>5}{'k':>6}{'E[trM]':>9}{'E[trM^2]/mksig':>15}{'E[trM2]/mksig':>14}"
      f"{'E[fro]/m2ksig':>14}{'E[RF]':>10}{'1/m':>10}{'1/(m+2)':>10}")
for (m, k) in [(64, 256), (128, 512), (256, 1024), (512, 2048)]:
    s = sample_stats(m, k, trials=3000)
    mk = m * k * s['sig2']
    print(f"{m:>5}{k:>6}{s['E_trM']:>9.2f}{s['E_trM2_sq']/mk:>15.3f}"
          f"{s['E_trM2']/mk:>14.3f}{s['E_fro']/(m*mk):>14.3f}"
          f"{s['E_rf']:>10.6f}{1/m:>10.6f}{1/(m+2):>10.6f}")

print("\n(ii) radial ratio tr(M)/||M||_F  RMS vs 1/sqrt(m)")
for (m, k) in [(64, 256), (256, 1024), (512, 2048)]:
    s = sample_stats(m, k, trials=3000)
    print(f"  m={m:<5} RMS={s['rms_radial']:.5f}   1/sqrt(m)={1/np.sqrt(m):.5f}")

print("\n(iv) concentration: std(R_F)/E[R_F] shrinks with m  (-> 'layer std ~ 0')")
for (m, k) in [(64, 256), (128, 512), (256, 1024), (512, 2048)]:
    s = sample_stats(m, k, trials=3000)
    print(f"  m={m:<5} CV = std/E = {s['std_rf']/s['E_rf']:.4f}")

print("\n(v) distribution-free: uniform entries (var-matched)")
for (m, k) in [(128, 512), (512, 2048)]:
    s = sample_stats(m, k, trials=3000, dist='uniform')
    print(f"  m={m:<5} E[RF]={s['E_rf']:.6f}  1/m={1/m:.6f}  CV={s['std_rf']/s['E_rf']:.4f}")

print("\n(iii') k-independence: fix m=256, vary k")
for k in [256, 512, 1024, 2048]:
    s = sample_stats(256, k, trials=3000)
    print(f"  k={k:<5} E[RF]={s['E_rf']:.6f}  1/m={1/256:.6f}")
