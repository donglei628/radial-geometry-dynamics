"""
E1-Extended: Common utilities for mapping the radial fraction landscape.

All measurement functions, input generators, and model builders are here.
Individual experiments import from this module.
"""

import numpy as np
import math
import os

try:
    import torch
    import torch.nn.functional as torchF
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch not installed. Some experiments require it.")


# ============================================================
# Core measurement: Radial Fraction
# ============================================================

def radial_fraction(x_hat, F_output):
    """
    Compute R_F = E[(xhat^T F)^2 / m] / E[||F||^2].

    Args:
        x_hat: (B, m) normalized direction, ||x_hat[i]|| = sqrt(m)
        F_output: (B, m) update vector

    Returns:
        scalar R_F estimate
    """
    if HAS_TORCH and isinstance(x_hat, torch.Tensor):
        m = x_hat.shape[-1]
        # (xhat^T F) for each sample
        dot = torch.sum(x_hat * F_output, dim=-1)  # (B,)
        radial_sq = dot ** 2 / m  # (B,)
        full_sq = torch.sum(F_output ** 2, dim=-1)  # (B,)
        return (radial_sq.mean() / full_sq.mean()).item()
    else:
        # NumPy path
        m = x_hat.shape[-1]
        dot = np.sum(x_hat * F_output, axis=-1)
        radial_sq = dot ** 2 / m
        full_sq = np.sum(F_output ** 2, axis=-1)
        return float(np.mean(radial_sq) / np.mean(full_sq))


def radial_fraction_np(x_hat, F_output):
    """NumPy-only version of radial_fraction."""
    m = x_hat.shape[-1]
    dot = np.sum(x_hat * F_output, axis=-1)
    radial_sq = dot ** 2 / m
    full_sq = np.sum(F_output ** 2, axis=-1)
    return float(np.mean(radial_sq) / np.mean(full_sq))


# ============================================================
# Input generators
# ============================================================

def make_x_hat_torch(B, m, distribution='gaussian', seed=None):
    """Generate normalized inputs on the sphere ||x|| = sqrt(m)."""
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    if distribution == 'gaussian':
        x = torch.randn(B, m, generator=g)
    elif distribution == 'sparse':
        # 90% zeros, 10% Gaussian
        x = torch.randn(B, m, generator=g)
        mask = (torch.rand(B, m, generator=g) < 0.1).float()
        x = x * mask
        # Ensure no all-zero rows
        zero_rows = (x.norm(dim=-1) < 1e-8)
        if zero_rows.any():
            x[zero_rows] = torch.randn(zero_rows.sum(), m, generator=g)
    elif distribution == 'heavy_tail':
        # Student-t with df=3
        normal = torch.randn(B, m, generator=g)
        chi2 = torch.distributions.Chi2(df=3).sample((B, 1))
        x = normal / torch.sqrt(chi2 / 3)
    elif distribution == 'spike':
        # One large outlier per row
        x = torch.randn(B, m, generator=g) * 0.1
        idx = torch.randint(0, m, (B,), generator=g)
        x[torch.arange(B), idx] = 3.0
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Normalize to ||x|| = sqrt(m)
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return x * math.sqrt(m) / norm


def make_x_hat_np(B, m, distribution='gaussian', rng=None):
    """NumPy version: generate normalized inputs on the sphere ||x|| = sqrt(m)."""
    if rng is None:
        rng = np.random.default_rng()

    if distribution == 'gaussian':
        x = rng.standard_normal((B, m))
    elif distribution == 'sparse':
        x = rng.standard_normal((B, m))
        mask = (rng.random((B, m)) < 0.1).astype(float)
        x = x * mask
        zero_rows = np.linalg.norm(x, axis=-1) < 1e-8
        if np.any(zero_rows):
            x[zero_rows] = rng.standard_normal((np.sum(zero_rows), m))
    elif distribution == 'heavy_tail':
        x = rng.standard_t(df=3, size=(B, m))
    elif distribution == 'spike':
        x = rng.standard_normal((B, m)) * 0.1
        idx = rng.integers(0, m, size=B)
        x[np.arange(B), idx] = 3.0
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return x * np.sqrt(m) / norm


# ============================================================
# MLP builders
# ============================================================

def make_mlp_torch(m, m_hidden=None, activation='relu', seed=None):
    """
    Build a two-layer MLP: F(x) = W2 * phi(W1 * x).
    Standard init: W1 ~ N(0, 1/m), W2 ~ N(0, 1/m_hidden).
    Returns a callable F(x_hat) -> F_output.
    """
    if m_hidden is None:
        m_hidden = m
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    W1 = torch.randn(m_hidden, m, generator=g) / math.sqrt(m)
    W2 = torch.randn(m, m_hidden, generator=g) / math.sqrt(m_hidden)

    act_fns = {
        'identity': lambda z: z,
        'relu': torch.relu,
        'gelu': lambda z: torchF.gelu(z),
        'silu': lambda z: torchF.silu(z),
        'tanh': torch.tanh,
    }
    phi = act_fns[activation]

    def F(x):
        # x: (B, m) -> output: (B, m)
        return (W2 @ phi(W1 @ x.T)).T
    return F, W1, W2


def make_correlated_mlp_torch(m, rho, activation='relu', seed=None):
    """
    Build a two-layer MLP with correlated W1 and W2.
    W2 = rho * (W1^T normalized to same Fro norm as W2_indep) + sqrt(1-rho^2) * W2_indep
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    W1 = torch.randn(m, m, generator=g) / math.sqrt(m)
    W2_indep = torch.randn(m, m, generator=g) / math.sqrt(m)

    if rho > 0:
        W1T_fro = W1.T.norm()
        W2_indep_fro = W2_indep.norm()
        # Normalize W1^T to have same Frobenius norm as W2_indep
        W1T_scaled = W1.T * (W2_indep_fro / W1T_fro)
        W2 = rho * W1T_scaled + math.sqrt(1 - rho ** 2) * W2_indep
    else:
        W2 = W2_indep

    act_fns = {
        'identity': lambda z: z,
        'relu': torch.relu,
        'gelu': lambda z: torchF.gelu(z),
        'silu': lambda z: torchF.silu(z),
        'tanh': torch.tanh,
    }
    phi = act_fns[activation]

    def F(x):
        return (W2 @ phi(W1 @ x.T)).T
    return F, W1, W2


# ============================================================
# Sanity checks
# ============================================================

def run_sanity_checks():
    """
    Run sanity checks to verify measurement code is correct.
    These should be run BEFORE any real experiment.
    """
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    m = 256
    B = 2048
    all_passed = True

    # Check 1: ||x_hat|| = sqrt(m)
    print("\n[Check 1] ||x_hat|| = sqrt(m)")
    x_hat = make_x_hat_torch(B, m, 'gaussian', seed=42)
    norms = x_hat.norm(dim=-1)
    expected = math.sqrt(m)
    mean_norm = norms.mean().item()
    print(f"  Expected: {expected:.4f}, Got mean: {mean_norm:.4f}, "
          f"std: {norms.std().item():.4e}")
    if abs(mean_norm - expected) > 0.01:
        print("  FAILED")
        all_passed = False
    else:
        print("  PASSED")

    # Check 2: F = x_hat should give R_F = 1
    print("\n[Check 2] F = x_hat => R_F = 1")
    R = radial_fraction(x_hat, x_hat)
    print(f"  Expected: 1.0000, Got: {R:.4f}")
    if abs(R - 1.0) > 0.01:
        print("  FAILED")
        all_passed = False
    else:
        print("  PASSED")

    # Check 3: F = random vector orthogonal to x_hat => R_F ~ 0
    print("\n[Check 3] F = random orthogonal to x_hat => R_F ~ 0")
    F_rand = torch.randn(B, m)
    # Project out the x_hat component
    dot = torch.sum(x_hat * F_rand, dim=-1, keepdim=True) / m
    F_perp = F_rand - dot * x_hat
    R = radial_fraction(x_hat, F_perp)
    print(f"  Expected: ~0, Got: {R:.6f}")
    if R > 0.01:
        print("  FAILED")
        all_passed = False
    else:
        print("  PASSED")

    # Check 4: F from i.i.d. MLP => R_F ~ 1/m
    print(f"\n[Check 4] F from i.i.d. MLP => R_F ~ 1/m = {1/m:.6f}")
    R_values = []
    for seed in range(20):
        F_fn, _, _ = make_mlp_torch(m, activation='relu', seed=seed)
        x = make_x_hat_torch(B, m, 'gaussian', seed=seed + 1000)
        with torch.no_grad():
            F_out = F_fn(x)
        R_values.append(radial_fraction(x, F_out))
    mean_R = np.mean(R_values)
    print(f"  Expected: {1/m:.6f}, Got mean: {mean_R:.6f}, "
          f"ratio m*R_F = {mean_R * m:.4f}")
    if abs(mean_R * m - 1.0) > 0.3:
        print("  FAILED")
        all_passed = False
    else:
        print("  PASSED")

    # Check 5: Fully correlated (rho=1) should give R_F >> 1/m
    print(f"\n[Check 5] Fully correlated (rho=0.99) => R_F >> 1/m")
    R_values = []
    for seed in range(20):
        F_fn, _, _ = make_correlated_mlp_torch(m, rho=0.99, activation='relu', seed=seed)
        x = make_x_hat_torch(B, m, 'gaussian', seed=seed + 1000)
        with torch.no_grad():
            F_out = F_fn(x)
        R_values.append(radial_fraction(x, F_out))
    mean_R = np.mean(R_values)
    print(f"  Got mean R_F = {mean_R:.6f} (m*R_F = {mean_R * m:.4f})")
    print(f"  Expected >> 1/m = {1/m:.6f}: {'PASSED' if mean_R > 5/m else 'NEEDS INVESTIGATION'}")
    if mean_R <= 5 / m:
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL SANITY CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED -- investigate before running experiments")
    print("=" * 60)
    return all_passed


# ============================================================
# Output helpers
# ============================================================

def ensure_output_dir(subdir='results'):
    """Create output directory if needed."""
    path = os.path.join(os.path.dirname(__file__), subdir)
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == '__main__':
    run_sanity_checks()
