Overview
--------

We evaluate a pre-trained **CLIP (ViT-B/32)** model (no fine-tuning) on:

-   Zero-shot classification (STL-10).

-   Bidirectional image--text retrieval (small curated set).

-   Domain generalization on PACS (photo, art_painting, cartoon, sketch).

The purpose: quantify prompt sensitivity, retrieval asymmetry, and cross-style robustness.

* * * * *

Datasets & preprocessing
------------------------

-   **STL-10** --- zero-shot classification (8,000 test images used).

-   **Retrieval set** --- curated 500 images (50 images × 10 classes from STL-10).

-   **PACS** --- subset filtered to seven classes: `dog, elephant, giraffe, guitar, horse, house, person` (sourced from Hugging Face: `flwrlabs/pacs`).

-   **Transforms (as used in experiments)**:

`T.Resize((224,224))
T.ToTensor()
T.Normalize((0.5,), (0.5,))`

(use the same transform for all datasets to match CLIP input).

* * * * *

Prompt strategies
-----------------

-   **Plain / Generic:** `a {class}`

-   **Descriptive / Photo:** `a photo of a {class}`

-   **Detailed:** long descriptive strings (e.g. `a commercial airplane flying in the sky`, or `an amazing, colourful ... photo of a {class}, {class}.`)

-   PACS experiments also used style prompts: `generic`, `photo`, `artwork`, `painting`, `cartoon`, `sketch`, `drawing` (applied per class, e.g. `a sketch of a giraffe`).

* * * * *

Key results (high level)
------------------------

### Zero-shot classification --- STL-10 (top-1 accuracy)

| Prompt type | Accuracy |
| --- | --- |
| Plain / Generic | **95.83%** |
| Descriptive (best overall) | **96.79%** |
| Detailed | **96.43%** |

**Notable per-class confusions (Descriptive prompts)**

-   `cat` --- weakest (~88.2%), most frequently confused with `deer`.

-   `car ↔ truck` and `airplane ↔ ship` --- persistent coarse-grained confusions.

-   Prompt effects are class-dependent (some classes improve with descriptive/detailed, others degrade under overly detailed prompts).

* * * * *

### Image ↔ Text retrieval (500 images)

-   **Text → Image:** 100% (Basic / Descriptive / Detailed queries)

-   **Image → Text:** Basic/Descriptive = 100%, **Detailed = 90%**

**Why the asymmetry?**\
Detailed queries add specific attributes (action, color, scene) that images may not contain; when matching *from* an image to the most similar text candidate the extra specificity in detailed texts can penalize correct but attribute-sparse images. Text→Image succeeds because CLIP can find images that contain the described attributes among many candidates.

* * * * *

### PACS --- Domain generalization (seven classes)

| Domain | generic | photo | artwork | painting | cartoon | sketch | drawing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **photo** | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| **art_painting** | 95.0 | 96.0 | 95.0 | 95.0 | 94.0 | 94.5 | 94.5 |
| **cartoon** | 95.5 | 96.0 | 97.5 | 96.0 | 96.0 | 96.5 | 96.0 |
| **sketch** | 86.5 | 86.0 | 88.5 | 88.0 | 87.0 | 87.0 | 87.0 |

-   **Best prompt by domain:** photo → `generic` (photo ties), art_painting → `photo` (96.0%), cartoon → `artwork` (97.5%), sketch → `artwork` (88.5%).

-   **Sketches are the hardest**: performance drops to mid-80s; CLIP struggles when sketches remove texture/color cues and simplify shape details.

**Representative failure:** giraffe sketches are often misclassified as horse --- sketches frequently omit giraffe-specific texture (spots) and elongated neck cues, making silhouettes ambiguous.

* * * * *

Practical takeaways
-------------------

-   **Prompt matters.** `a photo of a {class}` is a robust default; however, per-class and per-domain prompt tuning can help.

-   **Retrieval asymmetry.** Use short/descriptive queries for Image→Text tasks when image attributes are inconsistent; detailed queries are best when images reliably contain described attributes.

-   **Domain adaptation.** For stylized domains (paintings, cartoons, sketches) style-aware prompts (`artwork`, `painting`) help but may not fully close the gap --- sketch recognition often needs additional adaptation (fine-tuning, adapters, or sketch-specific prompts).

-   **CLIP bias summary.** The model tends to prioritize semantic/shape cues (supports zero-shot strength) but remains influenced by its pretraining distribution (photorealistic captions), which explains strong photo performance and degraded sketch performance.

* * * * *

Reproducibility / quick instructions
------------------------------------

-   Use the CLIP ViT-B/32 model from the OpenAI CLIP library or Hugging Face.

-   Apply the preprocessing transform above.

-   Zero-shot classification: compute image and text embeddings, choose class whose text embedding has highest cosine similarity to the image embedding (top-1).

-   PACS: use the HF dataset `flwrlabs/pacs`, filter to classes listed above, evaluate per-domain top-1 accuracy for each prompt set.

-   Retrieval: build a small candidate text set (Basic/Descriptive/Detailed) and measure Text→Image and Image→Text top-1 accuracies.

* * * * *

Where to find more
------------------

-   Full experimental details, tables, plots and code references are in the project report.
