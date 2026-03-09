import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from hog import X_train, y_train, X_val, y_val, X_test, y_test

MODEL_PATH = Path("../data/best_svm.pkl")


def top5_accuracy(model, X, y):
    decisions = model.decision_function(X)
    top5_preds = np.argsort(decisions, axis=1)[:, -5:]
    return np.mean([y[i] in top5_preds[i] for i in range(len(y))])


def evaluate(model, X, y, split):
    top1 = accuracy_score(y, model.predict(X))
    top5 = top5_accuracy(model, X, y)
    print(f"{split} — Top-1: {top1:.4f}, Top-5: {top5:.4f}")
    return top1, top5


# Grid search over C and gamma using val set
C_values     = [0.1, 1.0, 10.0, 100.0]
gamma_values = [1e-4, 1e-3, 1e-2, "scale"]

total         = len(C_values) * len(gamma_values)
best_val_top1 = -1
best_params   = {}

for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):
        combo = i * len(gamma_values) + j + 1
        print(f"\n[{combo}/{total}] Training SVM — C={C}, gamma={gamma}...")
        svm = SVC(kernel="rbf", C=C, gamma=gamma, decision_function_shape="ovr")
        svm.fit(X_train, y_train)
        print(f"[{combo}/{total}] Evaluating on val set...")
        val_top1, val_top5 = evaluate(svm, X_val, y_val, "Val")

        marker = " <-- best so far" if val_top1 > best_val_top1 else ""
        print(f"[{combo}/{total}] C={C}, gamma={gamma} — Top-1: {val_top1:.4f}, Top-5: {val_top5:.4f}{marker}")

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_params   = {"C": C, "gamma": gamma}
            best_model    = svm

print(f"\nBest params: {best_params} — Val Top-1: {best_val_top1:.4f}")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, MODEL_PATH)
print(f"Saved best model to {MODEL_PATH}")

# # Test (run only after hyperparameters are finalized)
# test_top1, test_top5 = evaluate(best_model, X_test, y_test, "Test")
