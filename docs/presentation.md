## Presentation Overview

This draft is aligned with the current repo state, the presentation rubric, and the outputs being used by `notebooks/cnn_comparison.ipynb`.

Important status note:

- both CNNs now have saved **test** metrics
- the six-model comparison can now be presented on a consistent test-set basis

---

## Overview

**Problem addressed**

Fine-grained vehicle classification across `196` classes in the Stanford Cars dataset. This is difficult because many classes differ by subtle visual cues such as trim, grille shape, headlights, badges, or model year details.

**Dataset used**

Stanford Cars Dataset:

- `16,185` images
- `196` car classes
- combined original train/test split, then resplit into project-specific `70/15/15`
- shared `train/val/test` split used across all methods

**Important preprocessing / feature choices**

- `HOG + SVM`: grayscale conversion, resize to `128x128`, HOG extraction with `pixels_per_cell=(16,16)`
- `Raw-pixel + SVM`: grayscale conversion, resize to `64x64`, flatten raw pixels, then standardize and reduce with PCA before SVM
- `CNN baseline`: resized normalized RGB images with a pretrained `ResNet18`
- `Improved CNN`: resized normalized RGB images with a pretrained `ResNet34`, stronger augmentation, label smoothing, and full fine-tuning

**Methods applied**

1. `HOG + SVM (RBF kernel)`
2. `HOG + SVM (RBF kernel, balanced class weights)`
3. `HOG + SVM (linear kernel)`
4. `Raw pixels + SVM`
5. baseline CNN
6. improved CNN

**Why those methods were appropriate**

The project compares classical feature-engineering pipelines against learned visual representations:

- HOG + SVM is a strong traditional baseline for image classification with fixed-length features
- the balanced HOG + SVM variant tests whether compensating for class imbalance helps
- raw-pixel + SVM tests whether hand-crafted HOG descriptors are actually helping
- CNNs test whether learned hierarchical features outperform classical pipelines on a fine-grained vision problem

**How performance was evaluated**

- Top-1 accuracy
- Top-5 accuracy
- identical `70/15/15` train/validation/test split for all methods
- validation used for model selection
- final comparison now uses test results for all six reported models

**What comparisons or alternative settings were considered**

- HOG + SVM with `RBF` vs `linear` kernels
- HOG + SVM with and without balanced class weights
- HOG features vs raw-pixel features for SVM
- baseline CNN vs improved CNN
- classical ML vs deep learning on the same dataset and split

**What was learned**

- CNNs dramatically outperform the SVM baselines on this dataset
- among the classical methods, `HOG + SVM (RBF)` and `HOG + SVM (RBF, balanced)` are almost identical
- `linear` HOG SVM is weaker than `RBF`
- raw-pixel SVM performs worst, suggesting that raw flattened pixels are a poor classical representation for this task
- the improved CNN is the strongest model overall on the current test comparison

---

## Section-by-section Breakdown

### Section 1: Problem Framing and Motivation

- **Problem:** classify car images at the make/model/year level across `196` classes
- **Why it is hard:** classes are visually similar and often differ only in subtle local details
- **Why it matters:** this is a good test of whether classical hand-crafted features can compete with learned deep features on a fine-grained vision task
- **Project question:** how much performance do we gain by moving from classical SVM pipelines to CNNs, and which feature/kernel choices matter most?

---

### Section 2: Dataset-Specific Understanding and Preprocessing

- **Dataset:** Stanford Cars, `16,185` images, `196` classes
- **Split setup:** original train/test merged and re-split into `70/15/15` with fixed seeds for consistency across models
- **Challenges in this dataset:**
  - many classes are visually close to each other
  - classifying year-level variants requires fine detail
  - real-world images include variation in lighting, angle, background, and scale
  - some classes have more examples than others, motivating the balanced-class SVM experiment

**Preprocessing by method**

- **HOG pipeline**
  - resize to `128x128`
  - convert to grayscale
  - extract HOG descriptors
  - use `pixels_per_cell=(16,16)` because `8x8` was too computationally heavy on available hardware

- **Raw-pixel SVM pipeline**
  - resize to `64x64`
  - convert to grayscale
  - flatten to raw pixel vectors
  - standardize and apply PCA before classification

- **CNN pipelines**
  - convert to RGB
  - resize / crop images for network input
  - normalize with ImageNet mean/std
  - improved CNN also adds stronger augmentation

---

### Section 3: Method Choice and Justification

**HOG + SVM**

- HOG captures local edge orientation and coarse shape information
- this is a reasonable baseline for cars because silhouette and body structure matter
- `RBF` kernel was tested because class boundaries are likely nonlinear
- `linear` kernel was tested to compare a simpler classical margin-based baseline
- `class_weight="balanced"` was tested to check whether giving rarer classes more weight improves performance on the imbalanced class distribution

**Raw-pixel + SVM**

- included as an ablation against HOG
- tests whether the hand-crafted HOG representation is actually adding value over raw flattened pixels
- PCA was required to make this pipeline computationally practical

**Baseline CNN**

- pretrained `ResNet18`
- simple transfer-learning baseline
- appropriate because the dataset is visual and benefits from learned image features

**Improved CNN**

- pretrained `ResNet34`
- stronger augmentation
- dropout, label smoothing, AdamW, cosine LR scheduling
- full fine-tuning by default
- chosen to test whether a stronger transfer-learning recipe substantially improves fine-grained classification

**Why this method mix is good for the project**

- compares hand-crafted features vs learned features
- compares kernel choice and class weighting within the HOG + SVM family
- compares hand-crafted HOG descriptors against raw flattened pixels
- compares simpler vs stronger CNN training recipes
- makes the final conclusions more defensible than using only one baseline and one deep model

---

### Section 4: Evaluation Design and Comparisons

- **Metrics:** Top-1 and Top-5 accuracy
- **Shared split:** same `train/val/test` partition across all models
- **Model selection:** validation performance used to choose best hyperparameters/checkpoints
- **Final reporting:** test-set results for all six models
- **Notebook comparison includes:**
  - `HOG + SVM (RBF)`
  - `HOG + SVM (RBF, balanced)`
  - `HOG + SVM (linear)`
  - `Raw pixels + SVM`
  - baseline CNN
  - improved CNN

**Hyperparameter choices currently used**

- **HOG + SVM (RBF):**
  - best `C = 10.0`
  - best `gamma = scale`

- **HOG + SVM (RBF, balanced):**
  - best `C = 10.0`
  - best `gamma = scale`
  - `class_weight = balanced`

- **HOG + SVM (linear):**
  - best `C = 0.1`

- **Raw pixels + SVM:**
  - `64x64` grayscale raw pixels
  - PCA to `256` components
  - search over `C in [1.0, 10.0]`
  - search over `gamma in [scale, 0.001]`
  - best `C = 10.0`, `gamma = scale`

- **Baseline CNN:**
  - pretrained `ResNet18`
  - trained for `10` epochs
  - test checkpoint came from epoch `6`

- **Improved CNN:**
  - pretrained `ResNet34`
  - stronger augmentation
  - label smoothing
  - AdamW
  - cosine LR scheduler
  - full fine-tuning
  - test checkpoint came from epoch `9`

---

### Section 5: Results and Interpretation

Current results from the saved comparison artifacts:

| Method | Split Shown | Top-1 | Top-5 |
|--------|-------------|-------|-------|
| HOG + SVM (RBF) | Test | 0.0910 | 0.2171 |
| HOG + SVM (RBF, balanced) | Test | 0.0914 | 0.2162 |
| HOG + SVM (linear) | Test | 0.0680 | 0.1808 |
| Raw pixels + SVM | Test | 0.0313 | 0.0840 |
| Baseline CNN | Test | 0.8262 | 0.9650 |
| Improved CNN | Test | 0.8929 | 0.9835 |

**Interpretation**

- The strongest current classical baseline is `HOG + SVM (RBF)`.
- Adding balanced class weights to the `RBF` HOG SVM changes the result only slightly:
  - Top-1 improves from `0.0910` to `0.0914`
  - Top-5 changes slightly downward from `0.2171` to `0.2162`
- That suggests class imbalance was not the main bottleneck in this project; representation quality is the bigger issue.
- Switching from `RBF` to `linear` on the same HOG features reduces performance, which suggests the class boundaries are not well modeled by a simple linear separator.
- Raw-pixel SVM performs much worse than HOG + SVM, which supports the idea that HOG is adding useful structure for a classical model.
- The baseline CNN improves dramatically over all SVM baselines, showing that learned visual features are much better suited to this fine-grained task.
- The improved CNN is the strongest model overall, increasing test Top-1 from `0.8262` to `0.8929` and test Top-5 from `0.9650` to `0.9835`.

**High-level conclusion**

- Fine-grained car classification is much better handled by transfer-learned CNNs than by classical SVM pipelines on hand-crafted or raw features.
- Within the SVM family, `RBF + HOG` is clearly the best option among the tested classical methods.
- Within the CNN family, the stronger training recipe and larger backbone provide a meaningful additional gain over the baseline CNN.

---

### Section 6: Depth of Understanding

**What worked**

- pretrained CNNs worked far better than the classical SVM baselines
- HOG was a much better classical feature representation than raw flattened pixels
- `RBF` kernel performed better than `linear` on HOG features
- balancing class weights did not materially improve the HOG + RBF SVM, which is itself an informative result

**What did not work well**

- raw-pixel SVM was weak and computationally expensive
- HOG with a finer `8x8` setting was too slow for the available hardware
- classical features struggled with the fine-grained nature of the dataset

**Why the methods behaved this way on this dataset**

- HOG captures edges and shape, but it loses color, texture, and many small details
- raw pixels preserve more information than HOG, but in a form that is hard for an SVM to use effectively
- CNNs learn task-specific features that can capture local texture, shape, and higher-level visual cues simultaneously
- the minimal gain from balanced class weights suggests the hardest issue is not mild class imbalance, but the visual complexity of the 196-way fine-grained recognition task

**Tradeoffs**

- SVM pipelines are conceptually simpler and easier to interpret
- CNNs require more compute and tuning, but the performance gain is much larger
- stronger CNN training recipes improve accuracy but also increase implementation complexity

**Limitations**

- no final qualitative misclassification analysis has been written into the presentation draft yet
- the study compares a limited set of feature and model families rather than an exhaustive benchmark
- no ensemble methods or larger modern architectures beyond the current improved CNN were tested

**Next steps**

- add one slide with representative failure cases or confusion examples
- if time permits, discuss why class weighting did not substantially help the HOG + SVM baseline
- mention possible future work such as larger pretrained backbones or ensembles

---

### Section 7: Presentation Clarity and Organization

Suggested slide order:

- Slide 1: title, team members, problem statement
- Slide 2: why fine-grained car classification is hard
- Slide 3: dataset and split setup
- Slide 4: SVM pipelines and variants
- Slide 5: CNN baseline and improved CNN
- Slide 6: evaluation setup and metrics
- Slide 7: full six-model comparison chart from `cnn_comparison.ipynb`
- Slide 8: HOG SVM kernel / class-weight comparison
- Slide 9: interpretation of CNN vs SVM gap and main takeaways
- Slide 10: conclusion, limitations, and next steps

Keep the focus on project decisions, comparisons, and findings rather than textbook algorithm definitions.

---

### Section 8: Team Participation and Individual Contribution

Suggested split:

- one person covers dataset, preprocessing, and SVM design choices
- one person covers CNN design choices, training setup, and model comparison
- both should be able to explain the evaluation setup and the main results

Prepare for Q&A on:

- why `16x16` HOG was used instead of `8x8`
- why raw-pixel SVM needed PCA and `64x64` resizing
- why `RBF` beat `linear` for HOG
- why balanced class weights only changed performance slightly
- why CNNs outperform the SVM pipelines so strongly on this dataset
