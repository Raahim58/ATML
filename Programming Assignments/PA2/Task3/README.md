
---

# (CLIP & Prompt Learning for DA/DG)

# CLIP + Prompt Learning (CoOp) — Domain Adaptation & Generalization

## Goal
Leverage large vision–language models (CLIP, ViT-B/32) and prompt-learning (CoOp) to adapt/generalize across PACS domains. Evaluate zero-shot, linear-probe, prompt-tuning, gradient alignment, and open-set robustness.

## Methods & setup
- **Zero-shot CLIP** — text prompt templates: simple prompts (e.g. "a photo of a {class}") and domain-specific prompts ("a {domain} of a {class}").  
- **Linear probing** — train a linear classifier over frozen CLIP image features.  
- **CoOp (Context Optimization)** — learnable context vectors (16 vectors in experiments) trained jointly across source domains. SGD with lr = 0.002 for 10 epochs was used in the report.  
- **Prompt learning + pseudo-labels** — incorporate pseudo-labeled target data with an unsupervised loss during prompt optimization to improve generalization.

Backbone used: CLIP ViT-B/32. :contentReference[oaicite:12]{index=12}

## Key results
- **Linear probe** on CLIP: Source ≈ **99.44%**, Target (Sketch) ≈ **84.22%**.  
- **CoOp prompt learning**: small improvement on target (Sketch) to **85.19%**.  
- **CoOp + pseudo-labeling** (DG setup): notable boost — **~90%** on Sketch in reported experiments.  
- **Open-set evaluation**: For seen classes in target Sketch, accuracy ≈ **80.74%** with average confidence 0.92; unseen classes show much lower confidence (~0.66), indicating weaker open-set robustness. :contentReference[oaicite:13]{index=13}

## Analysis & observations
- Prompt learning helps adapt CLIP to a specific target domain without full fine-tuning of image encoder.  
- Gradient analyses show **alignment among similar source domains** but **conflict (near-orthogonal gradients)** between some sources and target (e.g., Photo vs Sketch), which explains optimization difficulty.  
- Prompt learning improves closed-set DA/DG but struggles in open-set scenarios (confidence drops for unseen classes).
