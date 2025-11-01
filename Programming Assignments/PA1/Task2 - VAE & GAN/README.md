Overview
--------

We compare **Variational Autoencoders (VAEs)** vs **GANs (DCGAN)** to probe generative inductive biases.\
Focus: fidelity--diversity trade-off, latent space structure, OOD robustness, and training stability.

* * * * *

Datasets & Preprocessing
------------------------

-   **CIFAR-10** --- 60k images (train/test combined).

-   **OOD Internet images** --- anomaly stress test.

-   Preprocessing:

    -   VAE → scale [0,1].

    -   GAN → scale [-1,1].

* * * * *

Key Results
-----------

-   **Reconstruction vs Generation**

    -   VAEs → faithful reconstructions, blurry generations.

    -   GANs → sharp realistic generations, prone to artifacts/mode collapse.

-   **Latent Space**

    -   VAE → structured, smooth, interpretable (good for interpolation).

    -   GAN → discontinuous, sharp transitions, no explicit encoder.

-   **OOD Behavior**

    -   VAE → blurry reconstructions, useful for anomaly detection.

    -   GAN → fails on OOD latent inputs, often noise.

-   **Training Stability**

    -   VAE → stable convergence.

    -   GAN → unstable, oscillatory losses (tricks like label smoothing help).

* * * * *

Practical Takeaways
-------------------

-   **VAEs** are better for tasks needing representation learning, interpolation, anomaly detection.

-   **GANs** excel in **high-fidelity synthesis** but lack diversity/robustness.

-   Both architectures encode **biases that limit generalization** beyond training distribution.

* * * * *

Reproducibility
---------------

-   VAE: Conv encoder/decoder, β annealing variants.

-   GAN: DCGAN-style generator & discriminator.

-   Train with Adam (lr=0.001 for VAE, lr=0.0002 for GAN).

-   Metrics: **FID, IS, MSE** for quantitative comparison.
