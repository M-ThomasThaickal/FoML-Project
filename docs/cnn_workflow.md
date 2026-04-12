# CNN Workflow

This guide explains how to run the CNN portion of the FoML project from a fresh checkout.

## 1. Start From the Repo Root

Open a terminal in the project folder:

```powershell
cd C:\Users\Manu\repositories\FoML-Project
```

All commands below assume you are running them from the repo root.

## 2. Create and Activate a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PyTorch does not install for your Python version, create the environment with Python 3.10, 3.11, or 3.12 instead.

## 3. Install Dependencies

Install PyTorch first. For an NVIDIA GPU build:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

If you do not have an NVIDIA GPU or do not want CUDA:

```powershell
pip install torch torchvision torchaudio
```

Then install the rest of the project dependencies:

```powershell
pip install datasets scikit-learn scikit-image matplotlib jupyterlab numpy pillow joblib
```

## 4. Verify PyTorch

Run:

```powershell
python -c "import torch; print(torch.__version__); print('cuda available:', torch.cuda.is_available())"
```

If CUDA is working, `cuda available` should print `True`. CPU-only training can still run, but it will be much slower.

## 5. Train the CNN

Start with a small baseline run:

```powershell
python src/cnn.py --epochs 5 --batch-size 32
```

The script uses the same Stanford Cars train/validation/test split as the SVM pipeline. It trains a pretrained ResNet-18 classifier head by default and saves:

- `data/best_cnn.pt`: best checkpoint by validation Top-1 accuracy
- `data/cnn_history.json`: epoch-by-epoch train/validation loss, Top-1 accuracy, and Top-5 accuracy
- `data/cnn_results.json`: best validation result and output paths

The checkpoint file is ignored by Git because it can be large.

## 6. Fine-Tune the CNN

After the baseline works, try fine-tuning the whole model with a smaller learning rate:

```powershell
python src/cnn.py --epochs 10 --batch-size 32 --fine-tune --lr 0.0001
```

Use validation accuracy to decide whether the fine-tuned run is better than the frozen-backbone baseline.

## 7. Evaluate on the Test Set

Only run test evaluation once you are happy with validation performance:

```powershell
python src/cnn.py --eval-test-only
```

This loads `data/best_cnn.pt`, evaluates on the held-out test split, and saves:

- `data/cnn_results.json`: test loss, Top-1 accuracy, and Top-5 accuracy
- `data/cnn_test_predictions.npz`: true labels, Top-1 predictions, and Top-5 predictions

You can also train and evaluate in one command:

```powershell
python src/cnn.py --epochs 10 --batch-size 32 --fine-tune --lr 0.0001 --eval-test
```

## 8. Open the Comparison Notebook

Start Jupyter:

```powershell
jupyter lab
```

Open:

```text
notebooks/cnn_comparison.ipynb
```

Run the notebook cells from top to bottom. The notebook reads `data/cnn_history.json`, `data/cnn_results.json`, and `data/cnn_test_predictions.npz` if they exist. It produces:

- CNN training loss curves
- CNN train/validation Top-1 and Top-5 accuracy curves
- HOG + SVM vs CNN accuracy comparison
- CNN per-class Top-1 accuracy plot after test evaluation

## 9. Suggested Run Order

For a first complete pass:

```powershell
python src/cnn.py --epochs 5 --batch-size 32
jupyter lab
```

Then inspect `notebooks/cnn_comparison.ipynb`.

For a stronger final run:

```powershell
python src/cnn.py --epochs 10 --batch-size 32 --fine-tune --lr 0.0001 --eval-test
jupyter lab
```

Then rerun `notebooks/cnn_comparison.ipynb` to update the comparison figures.
