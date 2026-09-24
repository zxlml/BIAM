<div align="center">

# BIAM: Bilevel Interactive Additive Model

**A PyTorch implementation of the Bilevel Interactive Additive Model for learning under missing values, noisy labels, and class imbalance**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
[![Tests](https://img.shields.io/badge/Tests-63%20passed-brightgreen)](#-testing)

**English** | [简体中文](./README_zh.md)

</div>

---

## 📑 Table of Contents

- [✨ Highlights](#-highlights)
- [🔧 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [🧪 Data Corruption Scenarios](#-data-corruption-scenarios)
- [🔬 Ablation Variants](#-ablation-variants)
- [⚙️ Configuration](#️-configuration)
- [🧪 Testing](#-testing)
- [☑️ Todo List](#️-todo-list)
- [🪪 License](#-license)

## ✨ Highlights

Real-world tabular data is rarely clean. **BIAM** tackles the three most common corruptions in a single, interpretable framework:

1. **Missing Values** — explicit missing-indicator intercepts plus missing × feature *interactions*, so the model learns how missingness itself carries signal (supports MCAR / MAR / MNAR).
2. **Noisy Labels** — a **bilevel optimization** scheme learns per-sample weights ν(·;θ) on a clean validation set, automatically down-weighting corrupted training samples.
3. **Class Imbalance** — the same reweighting mechanism dynamically compensates for 1:N imbalanced training distributions, while the test set keeps its natural distribution.

Key design points:

- **Interpretable additive structure**: each feature contributes through a learnable shape function built on hinge bases located at data quantiles.
- **First-order bilevel optimization** (Eq. 4–6): a differentiable *virtual step* makes the upper-level hypergradient cheap — no second-order derivatives.
- **Strong extrapolation**: piecewise-linear hinge shape functions extrapolate far better than piecewise-constant alternatives.

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/zxlml/BIAM.git
cd BIAM

# (Optional) create a virtual environment
conda create -n biam python=3.10 -y
conda activate biam

# Install dependencies
pip install -r requirements.txt
```

> **Note (Windows)**: if you hit OpenMP duplicate-library errors, set the environment variable before running:
> ```powershell
> $env:KMP_DUPLICATE_LIB_OK='TRUE'
> ```

**Minimal requirements**: Python 3.8+, PyTorch 2.0+, NumPy, scikit-learn, pandas, matplotlib. CPU-only execution is fully supported.

## ⚡ Quick Start

### Command Line

```bash
# Regression with 30% label noise and 20% MCAR missing values
python biam_main.py --task regression --noise_ratio 0.3 --missing_ratio 0.2

# Classification with label flipping, class imbalance and MNAR missingness
python biam_main.py --task classification \
    --noise_ratio 0.2 --imbalance_ratio 0.15 \
    --missing_ratio 0.2 --missing_mechanism MNAR

# Fine-grained control over the noise model (paper Eq.: y <- y + N(mu_e, sigma_e))
python biam_main.py --task regression --noise_mean 1.0 --noise_std 0.5 --seed 42

# Ablation: disable bilevel reweighting (BIAM-B variant)
python biam_main.py --task regression --use_bilevel False
```

### Python API

```python
from utils.biam_config import BIAMConfig
from data.biam_data_generator import BIAMDataGenerator
from models.biam_model import BIAMModel
from models.biam_weighting_network import BIAMWeightingNetwork
from gradients.biam_optimizer import BIAMOptimizer

# 1. Configure
config = BIAMConfig()
config.task = 'regression'
config.noise_ratio, config.missing_ratio = 0.3, 0.2

# 2. Generate corrupted data (train / val / test splits, test stays clean)
generator = BIAMDataGenerator(config)
train_loader, val_loader, train_data, val_data, test_data = generator.generate_data()

# 3. Build model + weighting network; place hinge knots at training quantiles
biam_model = BIAMModel(config, config.device)
weighting_network = BIAMWeightingNetwork(config, config.device)
biam_model.additive_model.set_knots(torch.tensor(train_data[0], dtype=torch.float32))

# 4. Bilevel optimization
optimizer = BIAMOptimizer(config, biam_model, weighting_network)
for epoch in range(config.epochs):
    optimizer.train_epoch(train_loader, val_loader, epoch)
    if epoch % 20 == 0:
        print(optimizer.evaluate(test_data))   # {'mse': ..., 'rmse': ..., 'mae': ...}
```

A full end-to-end demo (training, evaluation, interpretation, visualization) is available:

```bash
python demo_biam.py
```

## 🏗️ Architecture

### Core Modules Overview

```
BIAM/
├── biam_main.py                  # Main entry point (CLI)
├── demo_biam.py                  # End-to-end demo
├── run_tests.py                  # Test-suite runner
├── data/                         # Data processing
│   ├── biam_data_generator.py    # Synthetic data + corruption pipeline (noise/imbalance/missing)
│   ├── biam_binarizer.py         # Missing-value indicators & feature binarization
│   ├── biam_dataset_loader.py    # Real-dataset loading utilities
│   └── biam_data_utils.py        # Data helpers
├── models/                       # Model components
│   ├── biam_additive_model.py    # Additive model: hinge bases, missing intercepts & interactions (Eq. 1)
│   ├── biam_weighting_network.py # Sample-weighting network ν(·;θ)
│   └── biam_model.py             # Top-level model wrapper
├── gradients/                    # Optimization
│   ├── biam_optimizer.py         # First-order bilevel optimizer (Eq. 4–6)
│   ├── biam_bilevel_optimizer.py # Alternative bilevel implementation
│   └── biam_gradient_methods.py  # Gradient utilities
├── utils/                        # Configuration, logging, performance tools
│   ├── biam_config.py            # Central configuration + validation
│   └── biam_logger.py            # Structured logging
├── visualization/                # Shape functions, feature importance, training curves
└── tests/                        # Unit / functional / simulation / performance tests
```

### The Additive Model (Eq. 1)

Each feature contributes through a shape function; missingness enters both as an intercept and as an interaction with other features' shape functions:

$$
f(x, m; w) = \beta_0 + \sum_{j} f_j(x_j) + \sum_{j} \beta^{\text{miss}}_j \, m_j + \sum_{j,k,\tau} \alpha_{j,k,\tau} \, \mathbb{I}(m_j = 1) \, h(x_k; \eta_\tau)
$$

where $h(x;\eta) = \max(0, x - \eta)$ is a hinge basis whose knot $\eta_\tau$ is placed at a training-data quantile, and $m_j \in \{0,1\}$ indicates missingness of feature $j$.

### First-Order Bilevel Optimization (Eq. 4–6)

| Step | Update | Purpose |
|------|--------|---------|
| **Step 1** | $w^{(t)} = w^{(t-1)} - \gamma_w \nabla_w R\big(\theta^{(t)}, w^{(t-1)}\big)$ | Real lower-level update of model parameters with ν-weighted loss |
| **Step 2** | $\hat{w}(\theta) = w^{(t-1)} - \gamma_w \nabla_w R\big(\theta^{(t)}, w^{(t)}\big)$ | Differentiable *virtual step* (kept in the autograd graph) |
| **Step 3** | $\theta^{(t+1)} = \theta^{(t)} - \gamma_\theta \nabla_\theta L_{\text{val}}\big(\hat{w}(\theta)\big)$ | Upper-level update via hypergradient through the virtual step |

The weighting network ν takes the **batch-standardized per-sample loss** as input and outputs sample weights; weights are mean-normalized so that reweighting preserves the average gradient scale.

## 🧪 Data Corruption Scenarios

Following the simulation protocol of the paper (§4.1), `BIAMDataGenerator` reproduces all three corruption types — **applied to the training set only unless stated otherwise**, with the test set kept clean:

| Scenario | Mechanism | Applied to |
|----------|-----------|------------|
| **Label noise (regression)** | $y \leftarrow y + \epsilon,\ \epsilon \sim \mathcal{N}(\mu_e, \sigma_e)$ for a fraction $r_1$ of samples | Train |
| **Label noise (classification)** | Random label flipping for a fraction $r_1$ of samples | Train |
| **Class imbalance** | Majority : minority resampled to `imbalance_ratio` | Train |
| **Missing values** | MCAR / MAR / MNAR masking at ratio `missing_ratio` | Train / Val / Test |
| **Clean test labels** | Natural class distribution, no flipped labels | Test |

## 🔬 Ablation Variants

| Variant | Switch | What is removed |
|---------|--------|-----------------|
| **BIAM** (full) | — | — |
| **BIAM-B** | `use_bilevel = False` | Bilevel reweighting → uniform sample weights |
| **BIAM-I** | `use_missing_interactions = False` | Missing × feature interaction terms |
| **BIAM-H** | `basis_type = 'piecewise_constant'` | Piecewise-linear hinge bases → piecewise-constant bases |

## ⚙️ Configuration

All behaviour is controlled via `utils/biam_config.py` (overridable through CLI flags). Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `task` | `'regression'` | `'regression'` or `'classification'` |
| `n_samples` / `n_features` | 1200 / 10 | Synthetic dataset size and dimensionality |
| `noise_ratio` | 0.3 | Fraction of training samples corrupted |
| `noise_mean` / `noise_std` | 1.0 / 0.5 | $\mathcal{N}(\mu_e, \sigma_e)$ of the additive regression noise |
| `imbalance_ratio` | 0.2 | Minority : majority ratio for training-set resampling |
| `missing_ratio` | 0.2 | Fraction of masked entries |
| `missing_mechanism` | `'MCAR'` | `'MCAR'`, `'MAR'`, or `'MNAR'` |
| `n_knots` | 8 | Number of hinge knots per feature (placed at quantiles) |
| `basis_type` | `'piecewise_linear'` | `'piecewise_linear'` or `'piecewise_constant'` (BIAM-H) |
| `use_bilevel` | `True` | `False` disables bilevel reweighting (BIAM-B) |
| `use_missing_interactions` | `True` | `False` removes missing × feature interactions (BIAM-I) |
| `lambda_l2` / `lambda_l0` | 1e-3 / 1e-4 | Smoothness / sparsity penalties on shape functions |
| `upper_lr` / `lower_lr` | 0.1 / 0.05 | Upper- (θ) and lower-level (w) learning rates |
| `epochs` / `batch_size` | 200 / 64 | Training schedule |
| `seed` | 42 | Global random seed |

## 🧪 Testing

The project ships with a 63-case test-suite covering unit, functional, simulation and performance levels:

```bash
# Full suite
python -m pytest tests/ -q

# By level
python -m pytest tests/test_biam_models.py -v        # model internals (hinge bases, missing handling, penalties)
python -m pytest tests/test_biam_data.py -v          # corruption pipeline (noise / imbalance / missing)
python -m pytest tests/test_biam_optimizer.py -v     # bilevel optimization correctness
python -m pytest tests/test_biam_simulation.py -v    # end-to-end behaviour vs. paper claims
python -m pytest tests/test_biam_performance.py -v   # scalability

# Or use the bundled runner
python run_tests.py
```

## ☑️ Todo List

- [ ] Real-dataset benchmarks (ADNI, UCI Adult, Credit)
- [ ] GPU acceleration & mixed-precision training
- [ ] Shape-function visualization for the interaction terms
- [ ] Automatic hyperparameter search for the bilevel learning rates

## 🪪 License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
