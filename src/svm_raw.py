import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from progress_utils import progress_bar
from raw_pixels import X_test, X_train, X_val, y_test, y_train, y_val

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = DATA_DIR / "best_svm_raw.pkl"
RESULTS_PATH = DATA_DIR / "svm_raw_results.json"

DEFAULT_C_VALUES = [1.0, 10.0]
DEFAULT_GAMMA_VALUES = ["scale", 1e-3]
FULL_C_VALUES = [0.1, 1.0, 10.0, 100.0]
FULL_GAMMA_VALUES = [1e-4, 1e-3, 1e-2, "scale"]


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


def maybe_limit_split(X, y, limit, split_name):
    if limit is None or limit >= len(y):
        return X, y

    rng = np.random.default_rng(42)
    indices = rng.choice(len(y), size=limit, replace=False)
    print(f"Using a {split_name} subset: {limit}/{len(y)} examples")
    return X[indices], y[indices]


def build_model(args, C, gamma):
    steps = [("scale", StandardScaler())]
    if args.pca_components > 0:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=args.pca_components,
                    svd_solver="randomized",
                    random_state=42,
                ),
            )
        )

    steps.append(
        (
            "svm",
            SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                decision_function_shape="ovr",
                verbose=args.verbose_fit,
            ),
        )
    )
    return Pipeline(steps)


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate an SVM on raw pixel features.")
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Use the original 4x4 grid instead of the faster default grid.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=256,
        help="Number of PCA components before the SVM. Use 0 to disable PCA.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional cap on the number of training examples for quicker experiments.",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Optional cap on the number of validation examples for quicker experiments.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Optional cap on the number of test examples for quicker experiments.",
    )
    parser.add_argument(
        "--verbose-fit",
        action="store_true",
        help="Print libsvm optimization output while fitting each SVM.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    C_values = FULL_C_VALUES if args.full_grid else DEFAULT_C_VALUES
    gamma_values = FULL_GAMMA_VALUES if args.full_grid else DEFAULT_GAMMA_VALUES

    X_train_use, y_train_use = maybe_limit_split(X_train, y_train, args.train_limit, "train")
    X_val_use, y_val_use = maybe_limit_split(X_val, y_val, args.val_limit, "val")
    X_test_use, y_test_use = maybe_limit_split(X_test, y_test, args.test_limit, "test")

    print(
        f"Raw-pixel SVM setup | train={X_train_use.shape}, val={X_val_use.shape}, "
        f"test={X_test_use.shape}, pca_components={args.pca_components}"
    )

    param_grid = [(C, gamma) for C in C_values for gamma in gamma_values]
    total = len(param_grid)
    best_val_top1 = -1.0
    best_val_top5 = -1.0
    best_params = {}
    best_model = None

    for combo, (C, gamma) in enumerate(progress_bar(param_grid, desc="Raw SVM grid"), start=1):
        print(f"\n[{combo}/{total}] Training raw-pixel SVM - C={C}, gamma={gamma}...")
        svm = build_model(args, C, gamma)
        start_time = time.perf_counter()
        print(
            f"[{combo}/{total}] Starting fit "
            f"(scale -> PCA({args.pca_components}) -> SVM)..."
        )
        svm.fit(X_train_use, y_train_use)
        fit_seconds = time.perf_counter() - start_time
        print(f"[{combo}/{total}] Finished fit in {fit_seconds:.1f}s")
        print(f"[{combo}/{total}] Evaluating on val set...")
        val_start = time.perf_counter()
        val_top1, val_top5 = evaluate(svm, X_val_use, y_val_use, "Val")
        val_seconds = time.perf_counter() - val_start

        marker = " <-- best so far" if val_top1 > best_val_top1 else ""
        print(
            f"[{combo}/{total}] C={C}, gamma={gamma} - "
            f"Top-1: {val_top1:.4f}, Top-5: {val_top5:.4f}, "
            f"fit {fit_seconds:.1f}s, val {val_seconds:.1f}s{marker}"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_val_top5 = val_top5
            best_params = {"C": C, "gamma": gamma}
            best_model = svm

    print(f"\nBest params: {best_params} - Val Top-1: {best_val_top1:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")

    test_top1, test_top5 = evaluate(best_model, X_test_use, y_test_use, "Test")
    save_json(
        RESULTS_PATH,
        {
            "feature_pipeline": "raw_pixels",
            "preprocessing": {"pca_components": args.pca_components},
            "search_space": {"C_values": C_values, "gamma_values": gamma_values},
            "best_params": best_params,
            "val": {"top1": best_val_top1, "top5": best_val_top5},
            "test": {"top1": test_top1, "top5": test_top5},
            "model_path": str(MODEL_PATH),
        },
    )
    print(f"Saved raw-pixel SVM metrics to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
