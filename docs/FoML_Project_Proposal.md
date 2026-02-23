# Vehicle Type Classification

## Dataset and Research Question

We will use the **Stanford Cars Dataset**, which contains 16,185 images of 196 classes of cars. The main research questions are:

1. How accurately can traditional ML methods classify these images?
2. What performance gains do deep learning approaches provide compared to traditional methods (SVM vs. CNN) for this case?

---

## Methods and Performance Evaluation

### Support Vector Machine (SVM) with HOG Features
One approach is to use a Support Vector Machine with Histogram of Oriented Gradients (HOG) features. HOG is used to capture edge orientations and shape information, which we hypothesize will work well for car body shapes but may perform poorly for finer classification such as badges on grills.

### Convolutional Neural Network (CNN)
The second approach is to use Convolutional Neural Networks. The CNN will learn hierarchical features automatically through backpropagation.

Both methods will use identical train/validation/test splits (70/15/15 of the training set).

### Evaluation
Both methods will be evaluated using **Top-1** and **Top-5 accuracy** to measure overall classification performance. By manually examining the top 20 misclassifications from each method, we can also analyze errors to understand common failure modes.

---

## Timeline / Team Collaboration

| Weeks | Tasks |
|-------|-------|
| 1–2 | Research basics, set up environment, import Stanford Cars Dataset, extract HOG features, train SVM, get first results |
| 3–4 | Analyze errors, tune SVM parameters, begin research into CNNs, set up PyTorch or TensorFlow |
| 5–7 | Build simple CNN architecture, get it training, debug and tune hyperparameters, train final model, address overfitting |
| 8–9 | Analyze and compare models, analyze errors, research why performance differs. Buffer time if CNN needs more work |
| 10 | Write-up |

---

## References

Krause, J., Stark, M., Deng, J., & Fei-Fei, L. (2013). 3D Object Representations for Fine-Grained Categorization. *IEEE Workshop on 3D Representation and Recognition.*

Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. *CVPR.*
