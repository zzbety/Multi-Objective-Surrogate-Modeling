# Surrogate-Assisted Multi-Objective Optimization for NAS

Code and experiments for evaluating surrogate-assisted multi-objective optimization (MOO) in neural architecture search (NAS). The project uses **NAS-Bench-201** with the **CIFAR-10** dataset to study whether surrogate-guided Pareto selection can reliably identify truly Pareto-optimal architectures.

## Abstract

Surrogate-assisted multi-objective optimization (MOO) is widely used to reduce evaluation costs in computationally expensive neural architecture search. By replacing expensive objective evaluations with surrogate predictions, optimization can be accelerated; however, the reliability of surrogate-guided Pareto selection remains unclear.

This study investigates the effectiveness of a **single-loop surrogate-assisted MOO framework** in identifying truly Pareto-optimal architectures. Individually tuned surrogate models are employed to predict multiple objectives, from which a **Surrogate Pareto Set (SPS)** is identified in the predicted objective space and then re-evaluated using true objective values to obtain an **Evaluated Surrogate Pareto Set (ESPS)**.

Experiments on NAS-Bench-201 with CIFAR-10 show that surrogate-assisted selection achieves high precision, with **87.8%** of ESPS solutions being truly Pareto-optimal, while capturing **47.4%** of the true Pareto set. Moreover, re-evaluating surrogate-selected candidates reduces surrogate-induced false positives from **64.7%** in the SPS to **12.2%** in the ESPS.

## Project Structure

```
.
├── data/                          # Input dataset
│   └── nasbench201_cifar10_all.csv
├── outputs/
│   ├── surrogate/                 # Surrogate training & tuning results
│   ├── pareto/                    # Pareto front exports (SPF, ESPF, TPF)
│   └── figures/                   # Generated plots
├── step1_performance-pairs_extraction.py
├── step2_train_tune_bmr.py        # Train, tune, and benchmark surrogates
├── step2_visualization.py         # Surrogate prediction visualizations
├── step2_surrogate_pipeline.py    # Surrogate pipeline for MOO
├── step3_moo_paretofronts.py      # Compute SPF, ESPF, TPF
├── step3_analysis.py              # Pareto front analysis & plots
├── paths.py                       # Shared path constants
├── requirements.txt
└── README.md
```

## Pipeline Overview

| Step | Script | Description | Outputs |
|------|--------|-------------|---------|
| 1 | `step1_performance-pairs_extraction.py` | Extract architecture–performance pairs from SyneTune NAS-Bench-201 | `data/nasbench201_cifar10_all.csv` |
| 2a | `step2_train_tune_bmr.py` | Train baseline surrogates, hyperparameter tuning, cross-validation benchmark | `outputs/surrogate/*.csv` |
| 2b | `step2_visualization.py` | True vs. predicted scatter, error analysis, CV boxplots | `outputs/figures/fg01–fg05` |
| 2c | `step2_surrogate_pipeline.py` | Load tuned models; used by Step 3 MOO | — |
| 3a | `step3_moo_paretofronts.py` | Compute Surrogate / Evaluated / True Pareto fronts on test set | `outputs/pareto/*.csv` |
| 3b | `step3_analysis.py` | Venn diagram, Pareto plots, rank preservation analysis | `outputs/figures/fg06–fg09` |

### Key Concepts

- **SPF (Surrogate Pareto Front)**: Non-dominated solutions in the *predicted* objective space.
- **ESPF (Evaluated Surrogate Pareto Front)**: SPF candidates re-evaluated with *true* objectives, then non-dominated filtered.
- **TPF (True Pareto Front)**: Non-dominated solutions using true objectives on the full test set.

### Objectives

Four objectives are optimized (all minimized):

- `metric_valid_error` — validation error
- `metric_runtime` — training runtime
- `metric_latency` — inference latency
- `metric_flops` — floating-point operations

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements:** Python 3.10+ recommended.

## Usage

Run the pipeline in order from the project root:

```bash
# Step 1: Extract NAS-Bench-201 data (skip if data/ already present)
python step1_performance-pairs_extraction.py

# Step 2: Train and tune surrogate models
python step2_train_tune_bmr.py

# Step 2 (optional): Visualize surrogate quality
python step2_visualization.py

# Step 3: Compute Pareto fronts
python step3_moo_paretofronts.py

# Step 3 (optional): Analyze and plot Pareto fronts
python step3_analysis.py
```

> **Note:** `step3_moo_paretofronts.py` imports tuned surrogate configs from `step2_surrogate_pipeline.py`. Run Step 2 before Step 3.

## Data

The dataset contains **15,625** NAS-Bench-201 architectures (6 categorical hyperparameters, 5 choices each) with performance metrics at epoch 200 on CIFAR-10. Step 1 downloads the blackbox via [SyneTune](https://github.com/syne-tune/syne-tune); the pre-extracted CSV is included in `data/` for convenience.

## Results Summary

| Metric | Value |
|--------|-------|
| ESPS precision (truly Pareto-optimal) | 87.8% |
| Recall of true Pareto set | 47.4% |
| False positives in SPS | 64.7% |
| False positives in ESPS | 12.2% |

## License

Academic research code. Add a license file before public release if needed.
