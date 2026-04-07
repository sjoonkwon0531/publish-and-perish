# Publish and Perish

**How AI-Accelerated Writing Without Proportional Verification Investment Degrades Scientific Knowledge**

S. Joon Kwon, Sungkyunkwan University (SKKU)

## Overview

This repository contains the model code, simulation results, and figure generation scripts for the paper *Publish and Perish*. The paper formalizes the asymmetry between AI-accelerated manuscript writing and peer review capacity through a minimal two-variable ODE model, demonstrating a paradoxical decline in verified knowledge output.

## Key Result

Under empirically calibrated parameters (γ = 2.0, δ = 0.5), the model predicts:
- **Honeymoon peak**: K/K₀ = 1.10 at t = 3.5 yr (circa 2026)
- **Paradox onset**: K < K₀ at t ≈ 6 yr (2028)
- **20-year outcome**: K/K₀ = 0.68 (32% knowledge loss)
- **Analytical steady state**: K_ss/K₀ = 0.60 (40% knowledge loss)

## Repository Structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── model.py              # Core ODE model
│   ├── generate_figures.py    # All 4 manuscript figures
│   └── sensitivity.py         # Sensitivity analysis & Monte Carlo
├── data/
│   ├── empirical_data.json    # NeurIPS, ICLR, arXiv, bioRxiv data
│   └── model_results.json     # Baseline simulation output
├── figures/
│   ├── figure1.png            # Model dynamics (4 panels)
│   ├── figure2.png            # Empirical validation (4 panels)
│   ├── figure3.png            # γ-δ parameter space heatmap
│   └── figure4.png            # Policy lever analysis (2 panels)
└── si/
    └── supplementary.md       # Supplementary Information
```

## Quick Start

```bash
pip install numpy scipy matplotlib
python src/model.py           # Run baseline simulation
python src/generate_figures.py # Generate all figures
python src/sensitivity.py      # Run sensitivity analysis
```

## Model Equations

The model couples two state variables (Q: review queue, q: verification quality) with one external input (φ_w: writing AI adoption) and one endogenous coupling (φ_r: review AI adoption):

```
dQ/dt = S₀·(1 + γ·φ_w(t)) − R_max·(1 + δ·φ_r(Q))
dq/dt = −λ·φ_r(Q)·(q − q_min) + μ·(1 − η·φ_r(Q))·(1 − q)

where:
  φ_w(t) = 1/(1 + exp(−k_w·(t − t_w)))   [prescribed logistic]
  φ_r(Q) = Q/(Q + Q_c)                     [endogenous, queue-driven]
  K(t) = R(t)·q(t)                         [knowledge output]
```

## Citation

```bibtex
@article{kwon2026publish,
  title={Publish and Perish: How AI-Accelerated Writing Without Proportional Verification Investment Degrades Scientific Knowledge},
  author={Kwon, S. Joon},
  journal={[to be determined]},
  year={2026}
}
```

## License

MIT License
