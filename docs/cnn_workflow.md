# CNN Workflow

This guide explains how to run the CNN side of the project and how to compare it against all current SVM baselines.

The comparison now includes:

- `HOG + SVM (RBF)`
- `HOG + SVM (RBF, balanced class weights)`
- `HOG + SVM (linear)`
- `Raw pixels + SVM`
- baseline CNN from `src/cnn.py`
- upgraded CNN from `src/cnn_improved.py`

## 1. Start From the Repo Root

Open a terminal in the project folder:

```powershell
cd C:\Users\Manu\repositories\FoML-Project
```

All commands below assume you are running from the repo root.

## 2. Create and Activate a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PyTorch does not install for your Python version, use Python 3.10, 3.11, or 3.12.

## 3. Install Dependencies

Install PyTorch first.

For an NVIDIA GPU build:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

For CPU-only:

```powershell
pip install torch torchvision torchaudio
```

Then install the rest of the project dependencies:

```powershell
pip install datasets scikit-learn scikit-image matplotlib jupyterlab numpy pillow joblib tqdm
```

`tqdm` enables the progress bars used by the longer-running feature extraction, SVM, and CNN scripts.

## 4. Verify PyTorch

Run:

```powershell
python -c "import torch; print(torch.__version__); print('cuda available:', torch.cuda.is_available())"
```

If CUDA is working, `cuda available` should print `True`. CPU-only training still works, but it will be slower.

## 5. Baseline CNN

Run the baseline CNN:

```powershell
python src/cnn.py --epochs 10 --batch-size 32
```

Saved artifacts:

- `data/best_cnn.pt`
- `data/cnn_history.json`
- `data/cnn_results.json`

To fine-tune the whole baseline model:

```powershell
python src/cnn.py --epochs 10 --batch-size 32 --fine-tune --lr 0.0001
```

## 6. Upgraded CNN

Run the stronger CNN comparison model:

```powershell
python src/cnn_improved.py --epochs 10 --batch-size 32
```

This model uses:

- pretrained `ResNet34`
- stronger augmentation
- label smoothing
- `AdamW`
- cosine learning-rate decay
- full fine-tuning by default

Saved artifacts:

- `data/best_cnn_improved.pt`
- `data/cnn_improved_history.json`
- `data/cnn_improved_results.json`

## 7. Evaluate CNNs on the Test Set

Evaluate the baseline CNN:

```powershell
python src/cnn.py --eval-test-only
```

This updates:

- `data/cnn_results.json`
- `data/cnn_test_predictions.npz`

Evaluate the upgraded CNN:

```powershell
python src/cnn_improved.py --eval-test-only
```

This updates:

- `data/cnn_improved_results.json`
- `data/cnn_improved_test_predictions.npz`

For a fairer final notebook comparison, run test evaluation for both CNNs so the notebook can show CNN test metrics instead of validation-best metrics.

## 8. Run the SVM Baselines

HOG + SVM (RBF):

```powershell
python src/svm.py
```

Saved artifacts:

- `data/best_svm.pkl`
- `data/svm_results.json`

HOG + SVM (RBF, balanced class weights):

```powershell
python src/svm_balanced.py
```

Saved artifacts:

- `data/best_svm_balanced.pkl`
- `data/svm_balanced_results.json`

HOG + SVM (linear):

```powershell
python src/svm_linear.py
```

Saved artifacts:

- `data/best_svm_linear.pkl`
- `data/svm_linear_results.json`

Raw-pixel SVM:

```powershell
python src/svm_raw.py
```

Saved artifacts:

- `data/best_svm_raw.pkl`
- `data/svm_raw_results.json`

Default behavior:

- standardizes raw-pixel features
- uses `64x64` grayscale raw-pixel features
- applies PCA before the SVM
- runs a smaller default search so the raw-pixel baseline is practical to finish

If you want the heavier original-style search:

```powershell
python src/svm_raw.py --full-grid
```

If you want a quick subset experiment:

```powershell
python src/svm_raw.py --train-limit 4000 --val-limit 1000 --test-limit 1000
```

Note:

- `src/svm_raw.py` uses scikit-learn `SVC`, so it is CPU-bound and will not make significant use of the GPU.

## 9. Open the Comparison Notebook

Start Jupyter:

```powershell
jupyter lab
```

Open:

```text
notebooks/cnn_comparison.ipynb
```

The notebook reads:

- `data/svm_results.json`
- `data/svm_balanced_results.json`
- `data/svm_linear_results.json`
- `data/svm_raw_results.json`
- `data/cnn_history.json`
- `data/cnn_results.json`
- `data/cnn_improved_history.json`
- `data/cnn_improved_results.json`

If test predictions exist, it also reads:

- `data/cnn_test_predictions.npz`
- `data/cnn_improved_test_predictions.npz`

The notebook produces:

- baseline vs upgraded CNN training curves
- full six-model accuracy comparison
- HOG kernel / class-weight comparison
- model hyperparameter summary
- per-class Top-1 accuracy plots for available CNN test predictions

## 10. Suggested Run Order

For a first complete pass:

```powershell
python src/svm.py
python src/svm_balanced.py
python src/svm_linear.py
python src/svm_raw.py
python src/cnn.py --epochs 10 --batch-size 32
python src/cnn_improved.py --epochs 10 --batch-size 32
jupyter lab
```

For a stronger final comparison:

```powershell
python src/svm.py
python src/svm_balanced.py
python src/svm_linear.py
python src/svm_raw.py
python src/cnn.py --eval-test-only
python src/cnn_improved.py --eval-test-only
jupyter lab
```

Then rerun `notebooks/cnn_comparison.ipynb` from top to bottom.
