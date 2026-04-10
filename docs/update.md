# Update

- **Desc:** The aim of this report is to ensure that you’re on track. This document should be at most 2 pages long (excluding references), and should contain:
- **Due date:** March 9

### Data exploration: what dataset are you using, and how have you tried to explore / visualize / familiarize yourself with it? What pre-processing steps have been performed, and what features were (or will be) extracted?

- Dataset: Stanford Cars Dataset
    - The Stanford Cars Dataset which contains 16,185 images of 196 classes of cars. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe. This dataset is found on HuggingFace. 
- EDA:
    - Overview: verify number of total, training, testing, and validation samples. Verify the number of classes and check the amount of greyscale images in the raw dataset.
    - Check the width and height distribution for the images. 
        - Can include the 2 figures here.
        - Width  — min: 78, max: 7800, mean: 702
        - Height — min: 41, max: 5400, mean: 484
    - Visualize distribution of classes - how many samples are in each class?
        - Can include figure here.
        - Interestingly, the number of samples for a class increases as the class number increases. 
            - Are higher classes more modern models? 
            - And do more modern models have more images than older models?
        - Samples per class — min: 26, max: 95, mean: 57.8
    - Select random samples for inspection
        - Select 16 images from training set and showed corresponding class
    - Inspect intra-class variation by eye
        - Pick 3 random classes and show 5 examples each
    - Visualize HOG features
        - Take one image (first element of training data), display a resized image, and then show visualization of HOG features (gradient patterns of edges and shapes)
        - hog() returns a tuple, and the second element `hog_img` is a 2D array where each pixel's brightness represents the gradient magnitude/orientation at that location, rendered so we can see the HOG descriptors as little orientation arrows
- Preprocessing:
    - The raw dataset (tanganke/stanford_cars, 16,185 images) is re-split from scratch into train/val/test (70/15/15) using fixed seeds, rather than using the original splits.
    - Grayscale images (25 found in train) are handled by stacking the 2D array into 3 channels when needed (done in the HOG visualization cell).
    - Images are resized to 128×128 before feature extraction.
    - Images are converted to grayscale (rgb2gray) as a step before HOG computation.
- Feature Extraction
    - HOG (Histogram of Oriented Gradients) has been extracted with:
        - orientations=9
        - pixels_per_cell=(16, 16) — originally tried (8, 8) but it was too slow
        - cells_per_block=(2, 2)
        - Results in a feature vector of fixed length per image (visualized in EDA section 6)
    - hog.py is the HOG pipeline script that:
        - Extracts HOG features for all three splits (train/val/test)
        - Handles grayscale images, resizing, and grayscale conversion
        - Saves results to a .npz cache file so extraction only runs once
        - Loads from cache on subsequent runs 

### Preliminary results: describe the ML techniques you have attempted so far, and any results you may have obtained. You may provide 1-2 figures if helpful.

- Tried: HOG 8,8 for pixels per cell - takes way too long to run on my device, so switched to 16,16
- HOG + SVM (RBF kernel) pipeline completed:
    - Grid search over C ∈ {0.1, 1.0, 10.0, 100.0} and gamma ∈ {1e-4, 1e-3, 1e-2, scale}, 16 combinations total, selected by best validation Top-1 accuracy
    - Best params: C=10.0, gamma=scale
    - Test set results: Top-1: 9.10%, Top-5: 21.71%
    - Results are low but expected — HOG loses fine-grained detail (colour, texture, badges) that distinguishes car makes/models, and 196 classes is a hard problem for a linear feature + kernel classifier

    - Generated figures:
        - side-by-side heatmap of Top-1 and Top-5 validation accuracy across the 4×4 (C, gamma) grid, with cell values annotated and the best configuration (C=10, gamma='scale') highlighted with a blue border. Also saves the figure to svm_grid_search.png.
    
    - Figures that would help illustrate results (not created yet):
        - Confusion matrix on test set — shows which classes are most confused with each other
        - Top-1 accuracy per class (bar chart) — highlights which car classes HOG+SVM handles better or worse
        - Val accuracy heatmap (C vs gamma) — visualizes the grid search landscape

### Teamwork: if you are working as part of a team, how have you been collaborating and dividing the work?

- Most of the work done so far has been done in collaboration. This means one member might be writing code/files and pushing them to the repository, but both of us are together in person or on call working through the project. This included brainstorming project ideas, doing research on techniques we might use, and writing the project report. 
- We worked together in determining the project workflow and the initialization of our hog.py and svm.py scripts (like choosing a starting configuration for HOG). Other subsections of the project have seen one member leading more than the other, but in these cases we frequently communicate our findings and the code being used. These sections include the exploratory data analysis on the dataset, led by Jesse, and the running of the SVM on the training + validation sets to determine the best hyperparameters, led by Manu.

### Next steps: based on your progress and any challenges you have encountered, what are the next steps you are considering?

- Initially, we ran HOG with the 8x8 pixels per cell configuration. This took too long, so we switched to the 16x16 configuration. Future steps would include testing different HOG configurations like the pixels per cell or block size. 
- Upon running SVM with RBF kernel on the chosen HOG configuration, we found many repeating values, like 0.0054 and 0.0964 for top-1 accuracy and 0.0227 and 0.2208 for top-5 accuracy. Even though we expect low accuracy for HOG + SVM even with perfect hyperparameters, this is an issue that we'll look a little more into. We might try finer C values (ex. 2, 5, 7 instead of just 1, 10). 
- Lastly, once we finish testing with HOG + SVM, we plan on moving to using CNN, which is expected to produce much better results than HOG. 