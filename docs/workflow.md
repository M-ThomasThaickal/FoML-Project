# Project Workflow

This document is a runbook for the current project pipeline.

The project now compares six approaches on the same Stanford Cars split:

- `HOG + SVM (RBF)`
- `HOG + SVM (RBF, balanced class weights)`
- `HOG + SVM (linear)`
- `Raw pixels + SVM`
- baseline CNN
- upgraded CNN

## Dataset and Split

Data loading happens in `entrypoint/load.py`.

- Source: `tanganke/stanford_cars` from Hugging Face
- Split strategy: combine original train and test, then resplit into `70/15/15`
- Seeds: `85` for the first split and `42` for the second split
- In-memory outputs: `train_ds`, `val_ds`, `test_ds`

## Pipeline Overview

### 1. Exploratory Analysis

Notebook: `notebooks/EDA.ipynb`

Purpose:

- inspect the dataset
- check image sizes and class balance
- visualize example images
- examine HOG features

### 2. HOG Feature Extraction

Script: `src/hog.py`

Inputs:

- `train_ds`
- `val_ds`
- `test_ds`

Behavior:

- resizes images to `128x128`
- converts to grayscale
- extracts HOG features
- caches features to disk

Outputs:

- `data/hog_features_16x16.npz` with the current `HOG_PARAMS`

### 3. HOG + SVM (RBF)

Script: `src/svm.py`

Inputs:

- HOG features from `src/hog.py`

Behavior:

- runs an RBF-kernel SVM grid search over `C` and `gamma`
- selects the best model using validation Top-1 accuracy
- evaluates the best model on the test set

Outputs:

- `data/best_svm.pkl`
- `data/svm_results.json`

### 4. HOG + SVM (RBF, Balanced)

Script: `src/svm_balanced.py`

Inputs:

- HOG features from `src/hog.py`

Behavior:

- runs the same RBF-kernel HOG SVM search
- sets `class_weight="balanced"`
- evaluates on the same validation and test splits

Outputs:

- `data/best_svm_balanced.pkl`
- `data/svm_balanced_results.json`

### 5. HOG + SVM (Linear)

Script: `src/svm_linear.py`

Inputs:

- HOG features from `src/hog.py`

Behavior:

- trains a linear-kernel SVM
- searches over `C`
- evaluates the best model on validation and test

Outputs:

- `data/best_svm_linear.pkl`
- `data/svm_linear_results.json`

### 6. Raw Pixel Feature Extraction

Script: `src/raw_pixels.py`

Inputs:

- `train_ds`
- `val_ds`
- `test_ds`

Behavior:

- resizes images to `64x64`
- converts to grayscale
- flattens raw pixels
- caches features to disk

Outputs:

- `data/raw_pixel_features_64x64.npz`

### 7. Raw Pixels + SVM

Script: `src/svm_raw.py`

Inputs:

- raw-pixel features from `src/raw_pixels.py`

Behavior:

- standardizes the raw-pixel features
- reduces dimensionality with PCA before the SVM
- runs an RBF SVM search on the reduced features
- uses a smaller default search space so the comparison is practical to run
- evaluates the best model on the test set

Outputs:

- `data/best_svm_raw.pkl`
- `data/svm_raw_results.json`

Recommended commands:

Default faster run:

```powershell
python src/svm_raw.py
```

Full heavier run:

```powershell
python src/svm_raw.py --full-grid
```

Quick experiment on subsets:

```powershell
python src/svm_raw.py --train-limit 4000 --val-limit 1000 --test-limit 1000
```

Performance note:

- `src/svm_raw.py` uses scikit-learn `SVC`, so it is CPU-bound and will not significantly use the GPU.
- PCA is included because an RBF SVM on raw pixel vectors is otherwise too slow for a practical comparison run.
- The raw-pixel pipeline uses `64x64` grayscale images, which reduces each feature vector to `4096` values.

### 8. Baseline CNN

Script: `src/cnn.py`

Behavior:

- trains a pretrained `ResNet18`
- freezes the backbone by default
- saves the best checkpoint by validation Top-1 accuracy
- can optionally evaluate on the test split

Outputs:

- `data/best_cnn.pt`
- `data/cnn_history.json`
- `data/cnn_results.json`
- `data/cnn_test_predictions.npz` when test evaluation is run

### 9. Upgraded CNN

Script: `src/cnn_improved.py`

Behavior:

- trains a pretrained `ResNet34`
- uses stronger augmentation
- uses label smoothing, `AdamW`, and cosine LR decay
- fine-tunes the whole model by default
- saves the best checkpoint by validation Top-1 accuracy
- can optionally evaluate on the test split

Outputs:

- `data/best_cnn_improved.pt`
- `data/cnn_improved_history.json`
- `data/cnn_improved_results.json`
- `data/cnn_improved_test_predictions.npz` when test evaluation is run

### 10. Comparison Notebook

Notebook: `notebooks/cnn_comparison.ipynb`

Compares:

- `HOG + SVM (RBF)`
- `HOG + SVM (RBF, balanced)`
- `HOG + SVM (linear)`
- `Raw pixels + SVM`
- baseline CNN
- upgraded CNN

Reads:

- `data/svm_results.json`
- `data/svm_balanced_results.json`
- `data/svm_linear_results.json`
- `data/svm_raw_results.json`
- `data/cnn_history.json`
- `data/cnn_results.json`
- `data/cnn_improved_history.json`
- `data/cnn_improved_results.json`

Optional diagnostic inputs:

- `data/cnn_test_predictions.npz`
- `data/cnn_improved_test_predictions.npz`

Produces:

- CNN training curves
- full six-model accuracy comparison bar chart
- HOG kernel / class-weight comparison chart
- model hyperparameter summary
- per-class CNN accuracy plots when prediction files exist

## Recommended Run Order

### First full pass

```powershell
python src/svm.py
python src/svm_balanced.py
python src/svm_linear.py
python src/svm_raw.py
python src/cnn.py --epochs 10 --batch-size 32
python src/cnn_improved.py --epochs 10 --batch-size 32
jupyter lab
```

### Fairer final comparison run

```powershell
python src/svm.py
python src/svm_balanced.py
python src/svm_linear.py
python src/svm_raw.py
python src/cnn.py --eval-test-only
python src/cnn_improved.py --eval-test-only
jupyter lab
```

After that, open `notebooks/cnn_comparison.ipynb` and run all cells.

## Notes

- All pipelines use the same train, validation, and test split from `entrypoint/load.py`.
- HOG and raw-pixel features are cached so they do not need to be recomputed every run.
- CNN checkpoints are selected by validation Top-1 accuracy.
- The notebook currently uses test metrics when they exist and falls back to validation metrics otherwise.
- Install `tqdm` in the environment if you want the full progress bars shown by the long-running scripts.
