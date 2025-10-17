---

# `README_DG.md` (Invariant / Robust Domain Generalization)

# Domain Generalization (DG) --- Invariant & Robust Methods

## Goal

Train on multiple source domains to learn representations that generalize to an unseen target domain (no target data at training). PACS used: train on three domains, test on held-out (Sketch).

## Methods implemented

- **ERM** --- empirical risk minimization baseline across combined sources.  

- **IRM** (Invariant Risk Minimization) --- penalize variability of optimal classifiers across domains; we tuned penalty weight λ ∈ {1, 10, 100} and found λ = 10 effective.  

- **GroupDRO** --- distributionally robust objective that upweights high-loss domains dynamically (learning rate for domain weights η_q = 0.1 used in experiments).  

- **SAM (Sharpness-Aware Minimization)** --- optimizer that encourages flat minima (perturbation radius ρ = 0.05).

## Setup & hyperparams (summary)

- Backbone: ResNet-50 pretrained on ImageNet; Adam optimizer (lr = 1e-4) for most runs.  

- Epochs: 5 epochs per method for fair comparison (report's experimental choice).  

- Key tuned values: IRM λ = 10; SAM ρ = 0.05; GroupDRO ηq = 0.1. :contentReference[oaicite:9]{index=9}

## Key results (validation average → target)

- ERM: validation avg leading to **67.19%** target accuracy.  

- IRM (λ = 10): **75.41%** target --- best observed DG performance.  

- GroupDRO: **72.82%** target.  

- SAM: **73.43%** target; also yields visibly flatter loss landscape and better robustness in multiple runs. :contentReference[oaicite:10]{index=10}

## Insights

- Enforcing invariance (IRM) tends to reduce source validation accuracy but improves unseen-target accuracy when the invariance assumption holds.  

- SAM's flat minima correlate with better generalization; visualizations in the report illustrate SAM's smaller loss change under perturbations.  

- GroupDRO helps when there is a notably hard domain among sources --- the dynamic reweighting focuses learning on the hardest domains to improve worst-case performance.
