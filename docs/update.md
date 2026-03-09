# Update

- **Desc:** The aim of this report is to ensure that you’re on track. This document should be at most 2 pages long (excluding references), and should contain:
- **Due date:** March 9

### Data exploration: what dataset are you using, and how have you tried to explore / visualize / familiarize yourself with it? What pre-processing steps have been performed, and what features were (or will be) extracted?

- Dataset: Stanford Cars Dataset
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

### Teamwork: if you are working as part of a team, how have you been collaborating and

dividing the work?

### Next steps: based on your progress and any challenges you have encountered, what are the next steps you are considering?
