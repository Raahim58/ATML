Overview
--------

We study **inductive and model biases** across:

-   CNNs (ResNet-50)

-   Vision Transformers (ViT-B/16)

-   Generative models (VAEs & GANs)

-   Multimodal models (CLIP ViT-B/32)

The goal: understand how **architectural assumptions and training objectives** shape what features models rely on, and how this impacts **out-of-distribution (OOD) generalization**.

* * * * *

Datasets & Experiments
----------------------

-   **STL-10** --- supervised baselines, grayscale tests.

-   **Stylized-ImageNet subset** --- shape vs texture cue-conflict.

-   **PACS** --- domain adaptation (Photo, Art, Cartoon → Sketch).

-   **CIFAR-10** --- training/evaluation for VAE and GAN.

-   **OOD internet images** --- generative stress tests.

-   **Custom corruption sets** --- noise, blur, occlusion, patch shuffle.

* * * * *

Key Results (High Level)
------------------------

-   **CNNs** --- texture-biased, rely on local cues.

-   **ViTs** --- capture global structure, robust to occlusion/shuffling.

-   **VAEs** --- stable, reconstruct well but blurry generations.

-   **GANs** --- sharp generations, unstable training, poor OOD.

-   **CLIP** --- multimodal alignment → semantic/shape bias, strong zero-shot and cross-domain robustness.

* * * * *

Practical Takeaways
-------------------

-   **Bias matters.** Architectural choices dictate robustness vs failure cases.

-   **Data & training objective** often more important than architecture (e.g., CLIP).

-   **Trade-offs:**

    -   VAEs = structured latent space but blurry.

    -   GANs = sharp samples but mode collapse.

-   **Human-aligned biases** (shape/semantic) improve OOD generalization.
