"""
Evaluate the saved best SVM model on the held-out test set.
Run this only once, after hyperparameters are finalized.

Usage:
    python evaluate_test.py
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score

MODEL_PATH = Path("../data/best_svm.pkl")
HOG_CACHE  = Path("../data/hog_features.npz")


def top5_accuracy(model, X, y):
    decisions = model.decision_function(X)
    top5_preds = np.argsort(decisions, axis=1)[:, -5:]
    return np.mean([y[i] in top5_preds[i] for i in range(len(y))])


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run svm.py first.")

if not HOG_CACHE.exists():
    raise FileNotFoundError(f"HOG cache not found at {HOG_CACHE}. Run hog.py first.")

model = joblib.load(MODEL_PATH)
data  = np.load(HOG_CACHE)
X_test, y_test = data["X_test"], data["y_test"]

top1 = accuracy_score(y_test, model.predict(X_test))
top5 = top5_accuracy(model, X_test, y_test)

print(f"Test — Top-1: {top1:.4f}, Top-5: {top5:.4f}")
