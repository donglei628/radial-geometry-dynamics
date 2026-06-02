# radial-geometry-dynamics

Companion code, pre-computed results, and figures for the paper

**A Trace Formula for Pre-Norm Transformers and the Ignition Dynamics of Radial Geometry**
Lei Dong (Independent Researcher), 2026.

This repository lets a reviewer **read the code and check every number, figure, and table in the
paper against the committed result files** — without re-running anything. Every figure ships with
the exact script that draws it and the exact JSON it reads; every JSON ships with the script that
produced it. Re-running the model-based experiments requires a GPU and the public checkpoints
(Pythia, OLMo, etc.); the synthetic and analysis experiments run on CPU in seconds.

---

## The object

For a linear sublayer `F(x̂) = M x̂` with `M = W_down · W_up`, the **radial fraction**

```
R_F = ( tr(M)^2 + tr(M^2) + ||M||_F^2 ) / ( (m+2) · ||M||_F^2 )
```

is the share of output energy along the input direction. The paper proves an **Evolution Law** with
three phases — (P1) a parameter-free floor `E[R_F] = 1/m` at initialization, (P2) a sigmoidal
**ignition** with time `t* ∝ ln(1/ε)`, (P3) post-ignition differentiation of the layers — and shows
the saturation level is `R_F ≈ (rank) × (coherence)²`, governed by trace coherence, not rank.

---

## Quick start

```bash
pip install -r requirements.txt

# CPU, seconds — synthetic verifications:
python experiments/verify_lemma1_floor.py          # P1: E[R_F]=1/m, Monte-Carlo
python experiments/exp_ignition_theory.py          # P2: sigmoid + t* ∝ ln(1/ε)
python experiments/exp_epsilon_sweep_lag.py        # causal loss–geometry lag, β ln(1/ε)
python experiments/saturation/exp_saturation.py    # saturation = participation ratio

# GPU + HF checkpoints (downloads weights on first run):
python experiments/exp_batch_training_dynamics.py  # Pythia 70M–1B trajectories
python experiments/exp_olmo_dynamics.py            # OLMo (second arch family)

# regenerate every figure from the committed JSON (no GPU needed):
python experiments/fig_training_dynamics.py
python experiments/fig_olmo.py
python experiments/fig_lora_control.py
python experiments/fig_epsilon_lag.py
python experiments/saturation/fig_nonnormality.py
python experiments/saturation/fig_nonnormality_training.py
```

Each experiment writes its JSON to `experiments/results/` (the saturation experiments keep theirs
alongside their scripts in `experiments/saturation/`). Figure scripts only read those JSONs, so they
reproduce the paper's figures offline.

---

## Layout

```
radial-geometry-dynamics/
├── experiments/
│   ├── e1_common.py                 # core: radial_fraction(), input generators, I/O
│   ├── exp_batch_free_metrics.py    # MODEL_CONFIGS, get_layers(), extract_mlp_weights()
│   ├── <experiment scripts>.py      # produce experiments/results/*.json
│   ├── fig_*.py                     # draw the paper figures from the JSONs
│   ├── results/                     # all pre-computed result JSON
│   └── saturation/                  # §6 rank×coherence experiments (self-contained)
├── figures/                         # the 7 figures used in the paper (PDF + PNG)
├── requirements.txt
└── LICENSE                          # MIT
```

---

## Paper → code → data → figure

| Paper claim | Section | Script(s) | Result JSON | Figure |
|---|---|---|---|---|
| Trace formula is exact (ratio ≈ 1.00) | §2–3 | `round5_trace_formula.py` | `trace_formula_verification.json` | — |
| **P1** floor `E[R_F]=1/m` (distribution-free) | §3, Lemma 3.2 | `verify_lemma1_floor.py` | (Monte-Carlo, printed) | — |
| **P2** ignition sigmoid, `t* ∝ ln(1/ε)`, R²=1.0 | §3 | `exp_ignition_theory.py`, `v4_exp_ignition_theory.py`, `exp_lemma2_ignition.py`, `exp_ignition_mechanism.py` | `exp_ignition_theory.json`, `v4_exp_ignition_theory.json`, `exp_lemma2_ignition.json` | — |
| **P1–P3** on Pythia (floor to 0.05%, scale-amplified differentiation, geometry **lags** loss) | §5 | `exp_batch_training_dynamics.py`, `exp_4_05_loss.py` | `exp_batch_training_dynamics.json`, `exp_4_05_loss.json` | `fig_training_dynamics` |
| Parameter-free floor + post-ignition on **OLMo** (SwiGLU) | §5 | `exp_olmo_dynamics.py` | `exp_olmo_dynamics.json` | `fig_olmo` |
| Loss–geometry lag is **causal**: `β ln(1/ε)`, r=1.0 | §3 / App | `exp_epsilon_sweep_lag.py` | `exp_epsilon_sweep_lag.json`, `exp_epsilon_lag_c_robustness.json` | `fig_epsilon_lag` |
| Saturation = participation ratio (synthetic, 1.2% err) | §6 / App | `saturation/exp_saturation.py` | `saturation/exp_saturation.json` | `fig_saturation` |
| `R_F ≈ rank × coherence²`; coherence governs R_F (ρ=0.99 vs 0.63), 7 models / 158 layers | §6 | `saturation/exp_nonnormality.py` | `saturation/exp_nonnormality.json` | `fig_nonnormality` |
| Ignition builds **coherence ×490 at fixed rank** (Pythia-160m) | §6 | `saturation/exp_nonnormality_training.py` | `saturation/exp_nonnormality_training.json` | `fig_nonnormality_training` |
| R_F near-orthogonal to HT-SR yet keeps unique functional signal | §6 / related | `exp_htsr_shootout.py`, `analyze_htsr.py` | `exp_htsr_shootout.json` | — |
| Nonlinearity inverts the functional radial sign; localized to the **SwiGLU gate** | §geometry | `exp_sign_flip_attribution.py`, `verify_angular_sign.py`, `check_sign_reversal.py` | `exp_sign_flip_attribution.json`, `verify_angular_sign.json` | — |
| R_F is a **fragile** per-layer quantization predictor (Spearman ρ −0.92…+0.55, sign-inconsistent; R_F-guided mixed-precision loses to random) | §9, App F (Table 3) | `lp_stage_a_quant_distortion.py`, `lp_stage_b_mixed_precision.py`, `exp_quant_formats.py` | `quant_stage_a_rf_distortion.json`, `quant_stage_b_mixed_precision.json`, `exp_quant_formats.json` | — |
| R_F-guided LoRA selection does **not** beat random | §7 | `exp_lora_control.py` | `exp_lora_control.json` | `fig_lora_control` |

The last three rows are reported in the paper as **negative / honest-scope results** (R_F is not a
quantization router, not a LoRA selector, and is sign-blind). The committed JSONs contain the full
per-model numbers behind those statements.

---

## Reproducibility notes

- **Determinism.** Synthetic experiments (gradient flow, Monte-Carlo floor, ε-sweep) are seeded and
  reproduce the committed JSONs exactly. Model-based experiments read public checkpoints; the weights
  are deterministic, so the reported R_F / coherence / trace numbers are reproducible up to float
  precision and HF version.
- **Models.** Pythia (70M–1B), OLMo-1B-hf, and the cross-architecture set used in §6 / HT-SR are all
  public Hugging Face checkpoints; `exp_batch_free_metrics.py:MODEL_CONFIGS` lists the exact IDs.
- **No silent caps.** Where a sweep is truncated (model list, step grid), it is stated in the script
  and in the paper.

## License

MIT — see `LICENSE`.
