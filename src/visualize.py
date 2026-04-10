"""
Generate visualizations from the saved SVM model and HOG feature cache.

Usage:
    python visualize.py

Outputs (saved to ../data/):
    confusion_matrix.png   — test-set confusion matrix
    per_class_accuracy.png — Top-1 accuracy per class bar chart
"""
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix

MODEL_PATH = Path("../data/best_svm.pkl")
HOG_CACHE  = Path("../data/hog_features_8x8.npz")
OUT_DIR    = Path("../data")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run svm.py first.")
if not HOG_CACHE.exists():
    raise FileNotFoundError(f"HOG cache not found at {HOG_CACHE}. Run hog.py first.")

model = joblib.load(MODEL_PATH)
data  = np.load(HOG_CACHE)
X_test, y_test = data["X_test"], data["y_test"]

print("Running predictions on test set...")
y_pred = model.predict(X_test)

# --- Confusion matrix ---
print("Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(20, 18))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Confusion Matrix — HOG + SVM (Test Set)", fontsize=14)
ax.set_xlabel("Predicted class", fontsize=11)
ax.set_ylabel("True class", fontsize=11)
ax.tick_params(axis="both", labelsize=5)
plt.tight_layout()
out = OUT_DIR / "confusion8x8_matrix.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
plt.close(fig)

# --- Top-1 accuracy per class bar chart ---
print("Generating per-class accuracy bar chart...")
classes = np.unique(y_test)
per_class_acc = []
for c in classes:
    mask = y_test == c
    acc  = np.mean(y_pred[mask] == y_test[mask])
    per_class_acc.append(acc)

per_class_acc = np.array(per_class_acc)
sort_idx = np.argsort(per_class_acc)
sorted_classes = classes[sort_idx]
sorted_acc     = per_class_acc[sort_idx]

fig, ax = plt.subplots(figsize=(22, 6))
ax.bar(range(len(sorted_classes)), sorted_acc, width=0.8)
ax.set_xticks(range(len(sorted_classes)))
ax.set_xticklabels(sorted_classes, rotation=90, fontsize=5)
ax.set_xlabel("Class (sorted by accuracy)", fontsize=11)
ax.set_ylabel("Top-1 Accuracy", fontsize=11)
ax.set_title("Per-Class Top-1 Accuracy — HOG + SVM (Test Set)", fontsize=14)
ax.axhline(per_class_acc.mean(), color="red", linestyle="--", linewidth=1,
           label=f"Mean: {per_class_acc.mean():.3f}")
ax.legend()
plt.tight_layout()
out = OUT_DIR / "per_class_accuracy8x8.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
plt.close(fig)

print("Done.")
