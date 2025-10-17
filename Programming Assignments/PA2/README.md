# Techniques for Domain Adaptation & Generalization --- Project Summary

**Short description**  

An empirical investigation of major Domain Adaptation (DA) and Domain Generalization (DG) techniques on the PACS benchmark, plus experiments using CLIP prompt learning. We compare statistical alignment (DAN), adversarial alignment (DANN/CDAN), self-training (pseudo-labeling), invariant-learning (IRM), robust optimization (GroupDRO), flatness regularization (SAM), and prompt-learning (CoOp) for both adaptation and generalization settings. Key trade-offs (alignment vs discriminability, adaptation vs generalization) are emphasized.

**Core goals**

- Evaluate representative DA/DG methods under consistent protocols.

- Measure target-domain gains, class-balanced performance, and proxy domain divergence.

- Study computational/algorithmic trade-offs and robustness under concept-shift scenarios.

- Evaluate prompt-learning (CoOp) with CLIP as a foundation-model approach to DA/DG.

**Dataset & backbone**

- Dataset: **PACS** (Art Painting, Cartoon, Photo, Sketch). Source domains combined; Sketch often used as target.  

- Backbone: **ResNet-50** pretrained on ImageNet; 224×224 inputs; Adam optimizer (lr = 1e-4) used in most experiments. :contentReference[oaicite:0]{index=0}

**High-level findings**

- **UDA**: Adversarial alignment (DANN) performed best on PACS-style shifts (DANN target acc ≈ **74.81%** vs source-only 60.43%). DAN and CDAN improved over source-only but less than DANN. Pseudo-labeling gave modest improvements. :contentReference[oaicite:1]{index=1}  

- **DG**: Invariant-learning (IRM, with tuned λ) and SAM (flatness) both improved generalization beyond ERM; IRM (λ=10) achieved the best unseen-domain accuracy (~**75.41%**). SAM also improved generalization and produced flatter minima. :contentReference[oaicite:2]{index=2}  

- **CLIP + Prompting**: Prompt learning (CoOp) on CLIP improved adaptation/generalization over linear probing and zero-shot baselines; adding pseudo-labels during prompt learning boosted generalization in experiments (up to **~90%** on Sketch in DG setting). Gradient alignment analyses reveal domain gradient conflicts during prompt tuning. :contentReference[oaicite:3]{index=3}

**Key references**

- DANN, DAN, CDAN, IRM, SAM, GroupDRO, CoOp (CLIP prompt learning). See full report for exact hyperparameters and tables. :contentReference[oaicite:5]{index=5}
