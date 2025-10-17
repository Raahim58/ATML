# Unsupervised Domain Adaptation (UDA) --- Summary & How to Reproduce

## Goal

Adapt from labeled source domains to an unlabeled target domain (multi-source → single-target). Main use case in this project: sources = {Art Painting, Cartoon, Photo} and target = Sketch (PACS).

## Methods implemented

- **Source-only (ERM baseline)** --- train only on source data.

- **DAN (Deep Adaptation Network)** --- MMD-based multi-kernel moment matching. Multi-kernel MMD with 5 Gaussian kernels; progressive adaptation weight schedule.  

- **DANN (Domain-Adversarial Neural Network)** --- adversarial domain discriminator + GRL (progressive λ schedule). Discriminator: [2048 → 1024 → 1024 → 1] with dropout.  

- **CDAN (Conditional DANN)** --- conditions alignment on classifier predictions ([h; p]).  

- **Self-training / Pseudo-labeling** --- train source model, generate pseudo-labels on target using confidence threshold τ = 0.7, fine-tune classifier/backbone (backbone optionally frozen), mix source samples with weight 0.6.

## Experimental setup (summary)

- Backbone: ResNet-50 (ImageNet pretrained).  

- Input: 224×224, ImageNet normalization.  

- Optimizer: Adam, lr = 1e-4 (unless otherwise specified).  

- DAN/DANN/CDAN training: typically **10 epochs** for adaptation (source-only pretrain 5 epochs), progressive λ schedule for alignment weight.  

- Proxy A-distance used to quantify domain divergence (train a linear domain classifier). :contentReference[oaicite:6]{index=6}

## Key quantitative results (PACS: multi-source → Sketch)

- Source-only: **60.43%** (target).  

- DAN: **66.16%** (target).  

- DANN: **74.81%** (target) --- **best** on PACS in these experiments.  

- CDAN: **70.48%** (target).  

- Pseudo-Label: **62.47%** (target).  

*(See report tables for class-wise F1, concept-shift variants and proxy A-distance values.)* :contentReference[oaicite:7]{index=7}

## Observations & tips

- PACS is primarily a **style** shift --- adversarial alignment (DANN) worked especially well because it suppresses stylistic features while preserving shape/semantic features.  

- Proxy A-distance alone was not predictive of final accuracy --- all methods achieved similar A-distance yet performance varied, meaning *which features* are aligned matters.  

- For pseudo-labeling, careful confidence filtering (τ) and mixing strategy are crucial; pseudo-labeling struggled here because initial target accuracy was low (noisy pseudo-labels).
