**Introduction**

_Describe the dataset and problem you are addressing. Provide background on the problem and discuss relevant prior literature. Your introduction should help the reader understand what task you are studying, why it matters, and how your project relates to existing work. For graduate students in CS5262, the report is expected to include a more in-depth literature review and stronger discussion of how the work fits within prior research._

Problem: Vehicle type classification across 196 car classes from images
- Q: How accurately can traditional ML methods classify these images?
- Q: What performance gains do deep learning approaches provide compared to traditional methods (SVM vs. CNN) for this case?

Dataset: Stanford Cars Dataset. 16,185 images, 196 classes, real-world variation in lighting, angle, and background

Why? Tests the limits of both traditional and deep learning approaches. Fine-grained categorization is a known hard problem, making it a meaningful testbed for comparing paradigms.

Challenges: many classes differ only by subtle visual details such as trim, headlights, grille shape, or model year

**Methods** 

_Clearly describe the dataset, preprocessing steps, and features used in your project. Explain the machine learning methods you applied or compared, and justify why those methods were appropriate for your dataset and task. Also describe your experimental design, including how hyperparameters were selected, how performance was evaluated, and what metrics were used. If you compared multiple methods, feature sets, preprocessing choices, or model settings, explain the reasoning behind those comparisons._

We are using the Stanford Cars Dataset which contains 16,185 images of 196 classes of cars, with real-world variation in lighting, angle, and background. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe. This dataset is found on HuggingFace. A 70/15/15 split was used for training/validation/testing.

As for exploratory data analysis, we verified the number of total, training, testing, and validation samples. We also verified the number of classes and check the amount of greyscale images in the raw dataset. We checked the distribution of image height and weight as well as the distribution of classes (interestingly, the number of samples for a class increases as the class number increases). Random samples were also selected (16 random, then five samples of three random classes) for qualitative inspection. Lastly, one image was taken (first element of training data) to be displayed as a resized image, and a visualization of HOG features (gradient patterns of edges and shapes) was shown for inspection and to help us understand HOG better. 

To preprocess the data, the raw dataset was split into 70/15/15 sections for training, testing, and validation. Grayscale images (25 found in train) were handled by stacking the 2D array into 3 channels when needed. Images were resized to 128×128 before feature extraction, and images were converted to grayscale (rgb2gray) as a step before HOG (Histogram of Oriented Gradients) computation. 



**Results.** 

_Present your main findings clearly and accurately. Use tables and figures where appropriate, and make sure all figures include captions. Results should be easy to read and should directly support the claims you make in the report._

Table 1. All Top-1 and Top-5 results for all methods
| Method | Split Shown | Top-1 | Top-5 |
|--------|-------------|-------|-------|
| HOG + SVM (RBF) | Test | 0.0910 | 0.2171 |
| HOG + SVM (RBF, balanced) | Test | 0.0914 | 0.2162 |
| HOG + SVM (linear) | Test | 0.0680 | 0.1808 |
| Raw pixels + SVM | Test | 0.0313 | 0.0840 |
| Baseline CNN | Test | 0.8262 | 0.9650 |
| Improved CNN | Test | 0.8929 | 0.9835 |

Figure 1. HOG SVM variant comparison

Figure 2. Bar graph visualization for comparing all methods

**Discussion and Conclusions.** 

_Interpret your findings carefully. Explain what worked, what did not work, and why. Discuss limitations, challenges, unexpected findings, and possible directions for future work. Where appropriate, connect your discussion to concepts covered in class._

What do the results mean? 
- HOG + SVM (RBF): strongest classical baseline
- HOG + SVM (RBF, balanced): nearly identical to standard RBF
    - Class imbalance was not the main bottleneck in this project
- HOG + SVM (linear): weaker than RBF
    - Class boundaries are not well modeled by a simple linear separator
- Raw pixels + SVM: worst classical result even after PCA
    - HOG is adding useful structure for a classical model.
- CNN performed better than HOG/SVM with regards to both Top-1 and Top-5 accuracy. 

What worked?
- HOG helps a classical model more than raw pixels do, and nonlinear decision boundaries matter more than class reweighting on this dataset.
- CNNs learned visual representations much better than SVM, transfer learning was effective on this dataset

What did not work? 
- Using raw pixels
- linear kernel
- 8x8 HOG
- classical ML in general did not perform that well

Error Analysis
- SVMs often confuse cars with similar silhouettes but different trims or model years because HOG drops color and fine texture.
- Raw-pixel SVM is especially weak because flattened pixels are a poor representation for a classical classifier.
- CNNs still likely fail on unusual viewpoints, occlusion, or visually near-duplicate classes. 

Limitations
- Limited family of models rather than an exhaustive benchmark
- Used 16x16 HOG instead of 8x8 (too expensive on available hardware)
- Raw-pixel SVM required PCA and a lower 64x64 input size just to be practical (vs 128x128 for SVM + HOG).

Tradeoffs explored: 
- Linear vs RBF kernel
- Bias-variance tradeoff: handled by C/gamma hyperparameter testing on val set
- Image resolution


**Contributions.**

_If you worked in a team, briefly describe the contribution of each team member. If you received advice or assistance from other researchers at Vanderbilt, please acknowledge it here._

Team: Understand pipeline, choose hyperparameters and which alternative methods to test
Jesse: EDA, SVM + HOG pipeline
Manu: CNN pipeline

**Resources.**

_If you referred to code examples, online materials, ChatGPT, or other AI tools, please list them and clearly explain how they were used. You are responsible for the correctness of anything you include in the report or implementation, regardless of the tools used._

LLM usage: helping with code generation: importing data, data visualization

**Code.**

_Submit a zip file or link to the code you wrote for the project. External libraries and toolboxes do not need to be included._

Please keep in mind that the report will be evaluated not only on whether all sections are present, but also on the quality of reasoning and technical understanding reflected in the report. In particular, strong reports will clearly justify method choices, show thoughtful evaluation, include meaningful comparisons or ablations where appropriate, and provide careful interpretation of the results. Reports that only describe a pipeline at a surface level, without explaining design decisions or what was learned from the experiments, will not score as highly.

Good reports are usually clear, concise, and well organized. They make it easy for the reader to understand the problem, the methods, the experiments, and the conclusions. Please make sure your writing is readable, your tables and figures are clearly labeled, and your claims are supported by the evidence you present.
