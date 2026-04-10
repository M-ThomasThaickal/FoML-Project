"""
Predict the car model for a given image using the saved HOG+SVM model.

Usage:
    python predict.py <path_to_image>
"""
import sys
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
from datasets import load_dataset

# Must match hog.py exactly
IMG_SIZE = (128, 128)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
)

MODEL_PATH = Path("../data/best_svm.pkl")


def extract_single(image_path):
    img = np.array(Image.open(image_path).convert("RGB"))
    img = resize(img, IMG_SIZE, anti_aliasing=True)
    img_gray = rgb2gray(img)
    features = hog(img_gray, **HOG_PARAMS)
    return features.reshape(1, -1)


def get_label_names():
    # Pull class names from the dataset info
    ds = load_dataset("tanganke/stanford_cars", split="train")
    return ds.features["label"].names


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Run svm.py first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    label_names = get_label_names()

    features = extract_single(image_path)
    pred_idx = model.predict(features)[0]
    print(f"Predicted class index: {pred_idx}")
    print(f"Predicted class name:  {label_names[pred_idx]}")

    # Top-5
    decisions = model.decision_function(features)
    top5_indices = np.argsort(decisions, axis=1)[0, -5:][::-1]
    print("\nTop-5 predictions:")
    for rank, idx in enumerate(top5_indices, 1):
        print(f"  {rank}. {label_names[idx]}")
