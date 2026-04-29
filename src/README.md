# `src/` overview

This folder holds the runnable project code for the Stanford Cars experiments.

Most scripts assume the dataset split has already been defined in `entrypoint/load.py`, which exposes `train_ds`, `val_ds`, and `test_ds`.

## Main scripts

### SVM pipeline

- `hog.py`
  Extracts HOG features from the train, validation, and test splits. Images are resized to `128x128`, converted to grayscale, and cached under `data/`.

- `svm.py`
  Trains the main HOG + SVM baseline with an RBF kernel. It runs a small grid search over `C` and `gamma`, picks the best model by validation Top-1 accuracy, then evaluates on the test set.

- `svm_balanced.py`
  Same HOG setup as `svm.py`, but uses `class_weight="balanced"` to test whether class reweighting helps.

- `svm_linear.py`
  Uses the same HOG features, but swaps the RBF kernel for a linear one.

- `raw_pixels.py`
  Builds a raw-pixel feature cache. Images are resized to `64x64`, converted to grayscale, flattened, and saved to `data/raw_pixel_features_64x64.npz`.

- `svm_raw.py`
  Trains an SVM on raw pixel features instead of HOG. The pipeline standardizes the features, optionally applies PCA, then fits an RBF SVM.

### CNN pipeline

- `cnn.py`
  Baseline CNN built on a pretrained `ResNet18`. By default it freezes the backbone and trains a new final classifier layer.

- `cnn_improved.py`
  Stronger CNN baseline built on a pretrained `ResNet34`. This version uses heavier augmentation, `AdamW`, cosine LR decay, label smoothing, and fine-tunes the full network by default.

## Helper scripts

- `predict.py`
  Loads the saved HOG + SVM model and predicts the class for one image path. It also prints the top 5 class guesses.

- `evaluate_test.py`
  Small standalone evaluator for a saved SVM model. This script assumes a specific model/cache layout and is less flexible than the main training scripts.

- `visualize.py`
  Generates a confusion matrix and per-class accuracy plot for the saved SVM model. Like `evaluate_test.py`, it assumes a specific saved HOG cache path.

- `progress_utils.py`
  Shared progress-bar wrapper used by longer-running scripts.

## Typical order

For the HOG + SVM runs:

1. Run `python src/hog.py`
2. Run one or more of:
   `python src/svm.py`
   `python src/svm_balanced.py`
   `python src/svm_linear.py`

For the raw-pixel SVM:

1. Run `python src/raw_pixels.py`
2. Run `python src/svm_raw.py`

For the CNNs:

1. Run `python src/cnn.py`
2. Run `python src/cnn_improved.py`

## Outputs

Most trained models and metric summaries are written to `data/`, including:

- saved `.pkl` SVM models
- saved `.pt` CNN checkpoints
- `.json` metric summaries
- cached feature files
- `.npz` prediction dumps for test evaluation

For the fuller project runbook, see `docs/workflow.md`.
