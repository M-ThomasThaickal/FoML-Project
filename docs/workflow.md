
### Preliminary Work
- Run `EDA.ipynb`

### Mid-project update

**Data loading** (`entrypoint/load.py`)
- Input: downloads `tanganke/stanford_cars` from HuggingFace (live, no local cache)
- Output: `train_ds`, `val_ds`, `test_ds` in memory (70/15/15 split, seeds 85/42)

**HOG feature extraction** (`src/hog.py`)
- Input: `train_ds`, `val_ds`, `test_ds` (from `load.py`)
- Output: `data/hog_features_8x8.npz` (or `hog_features_16x16.npz`, depending on `HOG_PARAMS`)

**SVM training** (`src/svm.py`)
- Input: `data/hog_features_8x8.npz` (loaded via `hog.py`)
- Output: `data/best_svm.pkl`

**Test evaluation** (`src/evaluate_test.py`)
- Input: `data/best_svm.pkl` + `data/hog_features.npz`
- Output: prints test Top-1 / Top-5
- Note: hardcodes `hog_features.npz` (no suffix) — may need to be updated to match actual cache filename

**Grid search visualization** (`notebooks/update.ipynb`)
- Input: hardcoded results from the `svm.py` grid search run (no external files needed)
- Output: `notebooks/svm_grid_search.png` — side-by-side Top-1/Top-5 heatmaps over the (C, gamma) grid
- HOG config used: orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2)
- Best params highlighted: C=10.0, gamma='scale' — Val Top-1: 0.0964, Top-5: 0.2208

**Visualizations** (`src/visualize.py`)
- Input: `data/best_svm.pkl` + `data/hog_features_8x8.npz`
- Output: `data/confusion8x8_matrix.png`, `data/per_class_accuracy8x8.png`

**Single-image prediction** (`src/predict.py`)
- Input: `data/best_svm.pkl` + a single image path (CLI argument)
- Output: prints predicted class name + Top-5 predictions

### Running SVM on raw pixels

**Raw pixel feature extraction + SVM training** (`src/svm_raw.py`)
- Input: `train_ds`, `val_ds`, `test_ds` (from `load.py`, loaded via `raw_pixels.py`)
- Intermediate cache: `data/raw_pixel_features.npz` (created on first run)
- Output: `data/best_svm_raw.pkl`
- Run: `python svm_raw.py` (extracts and caches raw pixel features on first run, then trains SVM)
