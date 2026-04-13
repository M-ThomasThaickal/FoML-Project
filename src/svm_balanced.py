import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from hog import X_test, X_train, X_val, y_test, y_train, y_val
from progress_utils import progress_bar

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = DATA_DIR / "best_svm_balanced.pkl"
RESULTS_PATH = DATA_DIR / "svm_balanced_results.json"


def top5_accuracy(model, X, y):
    decisions = model.decision_function(X)
    top5_preds = np.argsort(decisions, axis=1)[:, -5:]
    return np.mean([y[i] in top5_preds[i] for i in range(len(y))])


def evaluate(model, X, y, split):
    top1 = accuracy_score(y, model.predict(X))
    top5 = top5_accuracy(model, X, y)
    print(f"{split} - Top-1: {top1:.4f}, Top-5: {top5:.4f}")
    return top1, top5


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


C_values = [0.1, 1.0, 10.0, 100.0]
gamma_values = [1e-4, 1e-3, 1e-2, "scale"]

param_grid = [(C, gamma) for C in C_values for gamma in gamma_values]
best_val_top1 = -1.0
best_val_top5 = -1.0
best_params = {}
best_model = None

for combo, (C, gamma) in enumerate(progress_bar(param_grid, desc="Balanced SVM grid"), start=1):
    print(f"\n[{combo}/{len(param_grid)}] Training balanced SVM - C={C}, gamma={gamma}...")
    svm = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        decision_function_shape="ovr",
        class_weight="balanced",
    )
    svm.fit(X_train, y_train)
    print(f"[{combo}/{len(param_grid)}] Evaluating on val set...")
    val_top1, val_top5 = evaluate(svm, X_val, y_val, "Val")

    marker = " <-- best so far" if val_top1 > best_val_top1 else ""
    print(
        f"[{combo}/{len(param_grid)}] C={C}, gamma={gamma} - "
        f"Top-1: {val_top1:.4f}, Top-5: {val_top5:.4f}{marker}"
    )

    if val_top1 > best_val_top1:
        best_val_top1 = val_top1
        best_val_top5 = val_top5
        best_params = {"C": C, "gamma": gamma, "class_weight": "balanced"}
        best_model = svm

print(f"\nBest params: {best_params} - Val Top-1: {best_val_top1:.4f}")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, MODEL_PATH)
print(f"Saved best model to {MODEL_PATH}")

test_top1, test_top5 = evaluate(best_model, X_test, y_test, "Test")
save_json(
    RESULTS_PATH,
    {
        "feature_pipeline": "hog",
        "kernel": "rbf",
        "best_params": best_params,
        "val": {"top1": best_val_top1, "top5": best_val_top5},
        "test": {"top1": test_top1, "top5": test_top5},
        "model_path": str(MODEL_PATH),
    },
)
print(f"Saved balanced SVM metrics to {RESULTS_PATH}")
