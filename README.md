# Vision Modeling Algorithm — Riemannian Diffusion Trajectories for Uncertainty in Medical Imaging 🧠📈

**Objective.** This project analyzes **diffusion trajectories** as **paths on a Riemannian manifold** to quantify and improve **uncertainty** and **sampling stability** in medical deep learning models (with a focus on **score-based diffusion / NCSN++-style** inference).

---

## project
Diffusion models generate samples by iteratively moving through a noisy state space. In medical imaging, tiny instabilities can amplify into clinically-meaningful artifacts, and uncertainty signals can become unreliable.  
This repo treats the diffusion reverse process as **geometry-aware dynamics**: trajectories live on a **curved space**, where local linearization may fail unless it’s controlled.

---

## Key Contributions

1. **Curvature & local geometry diagnostics for diffusion sampling**
   - Investigate how **curvature** and **local geometric approximation quality** affect **sampling stability** in NCSN++-like score models.
   - Practical goal: identify regimes where tangent-space approximations become brittle.

2. **L2 linear approximations as stability proxies**
   - Use **linear approximations in Euclidean L2 space** as *proxies* for the manifold-constrained dynamics.
   - Goal: improve stability of diffusion-based inference while preserving useful uncertainty behavior.

3. **A practical geometric interpretation workflow for medical diffusion**
   - Provide a systematic pipeline: *trajectory logging → geometric signals → stability/uncertainty metrics → proxy-based stabilization*.

---

## Core Idea
Let the diffusion state be \(x_t\). Instead of viewing sampling purely in Euclidean terms, we interpret the reverse trajectory
\[
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0
\]
as a discrete path on a Riemannian manifold \((\mathcal{M}, g)\). Locally, one often relies on tangent-space (linear) approximations:
\[
\mathcal{M} \approx T_{x}\mathcal{M}.
\]
But curvature breaks linearization quality. This repo:
- **measures** local approximation quality (proxy curvature diagnostics),
- **quantifies** stability degradation,
- and **injects L2 proxy solutions** to stabilize inference.

---

## What you get
- Trajectory logging utilities for diffusion sampling (per-step diagnostics)
- Geometry-inspired metrics:
  - local linearization error proxies
  - step-to-step drift/variance growth
  - curvature-sensitive stability indicators
- Proxy-based stabilization hooks:
  - linear L2 proxy solvers (fast approximations)
  - “projection / correction” style updates
- Uncertainty evaluation:
  - predictive variance & stochastic sampling analysis
  - calibration-style metrics (e.g., coverage–risk / AURC-type evaluations if enabled)

> Note: The repo is designed to be **research-friendly and reproducible**: configs, fixed seeds, and deterministic options where possible.

---

## Tech Stack
- **Python**
- **PyTorch**
- **Triton** (optional, for performance kernels)
- **C++** (optional, for custom ops / fast solvers)

---

## Repository Layout (suggested)
```text
.
├── README.md
├── configs/
│   ├── train.yaml
│   ├── sample.yaml
│   └── eval.yaml
├── src/
│   ├── models/
│   │   ├── ncsnpp.py
│   │   └── score_wrapper.py
│   ├── diffusion/
│   │   ├── sde.py
│   │   ├── sampler.py
│   │   └── schedules.py
│   ├── geometry/
│   │   ├── metrics.py          # curvature / linearization proxy metrics
│   │   ├── tangent.py          # local linear approx tools
│   │   └── riemannian_ops.py   # constraint/projection hooks
│   ├── proxy/
│   │   ├── l2_linear_proxy.py  # proxy solvers in L2
│   │   └── stabilizers.py      # proxy-based stabilization strategies
│   ├── eval/
│   │   ├── stability.py
│   │   └── uncertainty.py
│   └── utils/
│       ├── seed.py
│       └── logging.py
└── scripts/
    ├── train.py
    ├── sample.py
    └── eval.py

## Experimental Goals
- Stability
- quantify how trajectory instability depends on noise schedule / step size, curvature proxy metrics, local linear approximation quality
- Uncertainty quality
- check whether proxy stabilization preserves diversity in samples, improves calibration behavior (coverage–risk / AURC-like), reduces catastrophic sampling collapse
