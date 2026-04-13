# Project Workflow

This document is a cleaner runbook for the main project pipeline.

The project compares four approaches on the same Stanford Cars split:

- `SVM + HOG`
- `SVM + raw pixels`
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

- `data/hog_features_8x8.npz` or `data/hog_features_16x16.npz`, depending on `HOG_PARAMS`

### 3. SVM on HOG Features

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

### 4. Raw Pixel Feature Extraction

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

### 5. SVM on Raw Pixels

Script: `src/svm_raw.py`

Inputs:

- raw-pixel features from `src/raw_pixels.py`

Behavior:

- standardizes the raw-pixel features
- reduces dimensionality with PCA before the SVM
- runs the same RBF-kernel SVM grid search as the HOG pipeline
- uses a smaller default search space so the comparison is practical to run
- selects the best model using validation Top-1 accuracy
- evaluates the best model on the test set

Outputs:

- `data/best_svm_raw.pkl`
- `data/svm_raw_results.json`

Recommended commands:

Default faster run:

```powershell
python src/svm_raw.py
```

This uses:

- `pca_components=256`
- `C in [1.0, 10.0]`
- `gamma in ["scale", 1e-3]`

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
- The raw-pixel pipeline now uses `64x64` grayscale images, which reduces each feature vector to `4096` values and makes the comparison more realistic on this hardware.

### 6. Baseline CNN

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

### 7. Upgraded CNN

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

### 8. Comparison Notebook

Notebook: `notebooks/cnn_comparison.ipynb`

Compares:

- `SVM + HOG`
- `SVM + raw pixels`
- baseline CNN
- upgraded CNN

Reads:

- `data/svm_results.json`
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
- final accuracy comparison bar chart
- per-class CNN accuracy plots when prediction files exist

## Recommended Run Order

### First full pass

```powershell
python src/svm.py
python src/svm_raw.py
python src/cnn.py --epochs 5 --batch-size 32
python src/cnn_improved.py --epochs 10 --batch-size 32
jupyter lab
```

### Final comparison run

```powershell
python src/svm.py
python src/svm_raw.py
python src/cnn.py --epochs 10 --batch-size 32 --fine-tune --lr 0.0001 --eval-test
python src/cnn_improved.py --epochs 10 --batch-size 32 --eval-test
jupyter lab
```

After that, open `notebooks/cnn_comparison.ipynb` and run all cells.

## Notes

- All pipelines use the same train, validation, and test split from `entrypoint/load.py`.
- HOG and raw-pixel features are cached so they do not need to be recomputed every run.
- CNN checkpoints are selected by validation Top-1 accuracy.
- The comparison notebook can still display partial results if some artifacts are missing, but the full comparison needs all four pipelines to be run.
- Install `tqdm` in the environment if you want the full progress bars shown by the long-running scripts.
