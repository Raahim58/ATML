Overview
--------

We compare **ResNet-50 (CNN)** and **ViT-B/16 (Transformer)** to study:

-   Shape vs texture bias.

-   Local vs global inductive biases.

-   Robustness to translations, occlusion, shuffling.

-   Domain adaptation (PACS).

* * * * *

Datasets & Preprocessing
------------------------

-   **STL-10** (baseline supervised fine-tuning).

-   **Stylized-ImageNet subset** (cue-conflict images).

-   **Grayscale STL-10** (local color cue removal).

-   **Occluded / shuffled STL-10** (masked/shuffled patches).

-   **PACS** --- cross-domain evaluation (Photo, Art, Cartoon → Sketch).

Transforms:

`T.Resize((224,224))
T.ToTensor()
T.Normalize(mean, std)`

* * * * *

Key Results
-----------

-   **Shape vs Texture Bias**

    -   ResNet-50 → ~53% texture-biased.

    -   ViT-B/16 → closer to shape preference, but still slight texture bias.

-   **Architectural Bias**

    -   CNNs → robust to translations, weak to occlusion/shuffling.

    -   ViTs → robust to occlusion/shuffling, weaker to translation on small data.

-   **Domain Generalization (PACS)**

    -   CNNs → drop to 54.6% on unseen sketch domain.

    -   ViTs → larger drop to 36.8% (likely due to small dataset).

* * * * *

Practical Takeaways
-------------------

-   CNNs learn **spurious local cues** (color, texture).

-   ViTs leverage **global attention**, making them more robust to missing patches.

-   With small datasets, ViTs may fail to generalize as effectively as CNNs.

-   **Dataset-induced bias** can shift results (ResNet became more shape-aware after multi-domain fine-tuning).

* * * * *

Reproducibility
---------------

-   Models: `torchvision.models.resnet50`, `vit_b_16`.

-   Train/fine-tune on STL-10, test across modified domains.

-   PACS available via Hugging Face: `flwrlabs/pacs`.
