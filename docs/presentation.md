## Presentation Overview

Each presentation should focus on your project itself rather than on a general explanation of machine learning algorithms.

---

## Overview

**Please explain the problem you addressed:**
Fine-grained vehicle type classification across 196 car classes. The core challenge is distinguishing visually similar vehicles — not just "car vs. truck" but make, model, and year-level distinctions.

**The dataset you used:**
Stanford Cars Dataset — 16,185 images across 196 classes of cars, with labeled train/test splits.

**The preprocessing or feature choices that were important:**
For SVM: HOG (Histogram of Oriented Gradients) feature extraction — captures edge orientations and shape information from images. For CNN: raw image input with standard resizing/normalization; features are learned automatically.

**The methods you applied:**
1. SVM with HOG features (traditional ML baseline)
2. SVM with raw features 
3. CNN trained with backpropagation (deep learning approach)

**Why those methods were appropriate for your specific dataset and task:**
See Section 3 below.

**Describe how you evaluated your approach:**
Top-1 and Top-5 accuracy on identical train/val/test splits (70/15/15). Manual error analysis of the top 20 misclassifications per method to understand failure modes.

**What comparisons or alternative settings you considered:**
SVM vs. CNN is the primary comparison — a traditional feature-engineering pipeline vs. a learned-representation approach, evaluated on the same splits and metrics.

**What you learned from the results:**
[TODO: fill in after final results are ready]

**If you explored multiple methods, parameter choices, features, or preprocessing strategies, what was the reasoning behind those choices?**
[TODO: fill in after final results are ready — include SVM kernel/C tuning, CNN architecture choices, hyperparameter decisions]

**What the comparisons showed.**
[TODO: fill in after final results are ready]

---

## Section-by-section breakdown

### Section 1: Problem framing and motivation (10 pts)
Clearly explains the problem being addressed, the dataset used, and why the task is meaningful or interesting.

- **Problem:** Vehicle type classification of 196 car classes from images — a hard computer vision task where inter-class differences can be subtle (same make, different year)
  - Q: How accurately can traditional ML methods classify these images?
  - Q: What performance gains do deep learning approaches provide compared to traditional methods (SVM vs. CNN) for this case?
- **Dataset:** Stanford Cars Dataset — 16,185 images, 196 classes, real-world variation in lighting, angle, and background
- **Why:** Tests the limits of both traditional and deep learning approaches. Fine-grained categorization is a known hard problem, making it a meaningful testbed for comparing paradigms. 

---

### Section 2: Dataset-specific understanding and preprocessing (15 pts)
Clearly explains the dataset, important preprocessing steps, feature choices, and any data-related challenges relevant to the project.

- **Dataset:** 16,185 images across 196 classes; imbalanced class sizes; real-world photography (varying pose, lighting, background)
- **Preprocessing:**
  - SVM pipeline: resize images to fixed dimensions → extract HOG features → train SVM on feature vectors
  - CNN pipeline: resize + normalize pixel values → feed into network
- **Feature choices:** HOG chosen because it captures edge orientation and shape — suited to car body silhouettes. CNN learns its own features hierarchically from raw pixels.
- **Data-related challenges:** [TODO: fill in — e.g., class imbalance, image size variation, train/val split strategy]

---

### Section 3: Method choice and justification (15 pts)
Clearly explains the machine learning method(s) used and, more importantly, why those methods were appropriate for this specific dataset and task.

- **SVM + HOG:**
  - HOG captures edge orientations and shape — well-suited to distinguishing car body types (sedan vs. SUV vs. truck) where silhouette structure differs
  - SVM is a strong classical classifier for fixed-length feature vectors like HOG descriptors
  - Hypothesis: performs reasonably on coarse distinctions, but struggles with fine-grained differences (e.g., subtle badge/grill details between makes/models) because HOG discards color and fine texture

- **CNN:**
  - Learns hierarchical features automatically via backpropagation — no need for hand-crafted descriptors
  - Captures both low-level texture/edges and high-level semantic features, which is critical for 196-class fine-grained categorization
  - Expected to outperform SVM precisely because fine-grained distinctions require richer representations than HOG can provide

- **Why this pairing:** SVM+HOG and CNN represent two distinct paradigms — manual feature engineering vs. learned representations. Comparing them directly answers the project's core research question and makes the performance gap interpretable.

---

### Section 4: Evaluation design and comparisons (15 pts)
Clearly explains how performance was evaluated, including metrics, train/test setup or validation strategy, and any comparisons to baselines, alternative methods, or alternative settings.

- **Metrics:** Top-1 accuracy (exact match) and Top-5 accuracy (correct class in top 5 predictions)
- **Splits:** Identical 70/15/15 train/val/test splits used for both methods to ensure fair comparison
- **Error analysis:** Manual inspection of top 20 misclassifications per method — used to understand failure modes qualitatively, not just quantitatively
- **Comparisons:** SVM+HOG vs. CNN on the same task, same splits, same metrics
- **Alternative settings considered:** [TODO: SVM kernel choices, C/gamma tuning; CNN architecture variants, learning rate, regularization]

---

### Section 5: Results and interpretation (20 pts)
Presents results clearly and correctly. Interprets what the results mean, explains what worked and what did not, and draws reasonable conclusions.

- **Results table:**

| Method | Top-1 Accuracy | Top-5 Accuracy |
|--------|---------------|----------------|
| SVM + HOG | [TODO] | [TODO] |
| CNN | [TODO] | [TODO] |

- **What worked:** [TODO]
- **What did not work:** [TODO]
- **Error analysis findings:** [TODO — e.g., did SVM confuse same-model different-years? Did CNN fail on unusual angles?]
- **Conclusions:** [TODO]

---

### Section 6: Depth of understanding (15 pts)
Demonstrates real ownership of the project — design choices, implementation decisions, limitations, and tradeoffs. Explains how methods behaved on *this dataset* specifically.

- **Key insight — HOG limitation:** HOG discards color and fine texture, so it likely struggles with distinguishing cars that share body shape but differ in badge/trim details — exactly the kind of distinction needed for 196-class fine-grained classification
- **Key insight — CNN advantage:** CNN's ability to learn task-specific features means it can pick up on subtle visual cues HOG cannot encode
- **Tradeoffs:** SVM+HOG is faster to train and more interpretable; CNN is more powerful but requires more data, compute, and tuning
- **Limitations:** [TODO — e.g., dataset size relative to 196 classes, no pretrained weights / transfer learning used, etc.]
- **Next steps:** [TODO — e.g., transfer learning with a pretrained backbone, data augmentation, ensemble approaches]

---

### Section 7: Presentation clarity and organization (5 pts)
Clear, well-organized slides that fit within the time limit.

- [ ] Slide 1: Title + team members
- [ ] Slide 2: Problem and motivation
- [ ] Slide 3: Dataset overview
- [ ] Slide 4: Methods — SVM+HOG pipeline
- [ ] Slide 5: Methods — CNN architecture
- [ ] Slide 6: Why these methods (justification)
- [ ] Slide 7: Evaluation setup
- [ ] Slide 8: Results (table/chart)
- [ ] Slide 9: Error analysis / qualitative findings
- [ ] Slide 10: Conclusions + limitations + next steps

---

### Section 8: Team participation and individual contribution (5 pts)
All team members participate and can explain their part of the project.

- [TODO: assign slide ownership per team member]
- Each person should be prepared to answer Q&A about their section's design choices and implementation decisions
