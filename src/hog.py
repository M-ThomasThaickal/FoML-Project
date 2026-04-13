import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "entrypoint"))
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
from load import train_ds, val_ds, test_ds
from progress_utils import progress_bar

# HOG parameters
IMG_SIZE = (128, 128)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
)

_ppc = HOG_PARAMS["pixels_per_cell"][0]
CACHE_PATH = DATA_DIR / f"hog_features_{_ppc}x{_ppc}.npz"


def extract_hog(dataset, split_name):
    X, y = [], []
    for sample in progress_bar(dataset, desc=f"HOG {split_name}", total=len(dataset)):
        img = np.array(sample["image"])
        # Handle grayscale images (H, W) -> (H, W, 3)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = resize(img, IMG_SIZE, anti_aliasing=True)
        img_gray = rgb2gray(img)
        features = hog(img_gray, **HOG_PARAMS)
        X.append(features)
        y.append(sample["label"])
    return np.array(X), np.array(y)


if CACHE_PATH.exists():
    print("Loading HOG features from cache...")
    data = np.load(CACHE_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
else:
    print("Extracting HOG features (this may take a few minutes)...")
    X_train, y_train = extract_hog(train_ds, "train")
    X_val,   y_val   = extract_hog(val_ds, "val")
    X_test,  y_test  = extract_hog(test_ds, "test")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, X_train=X_train, y_train=y_train,
                         X_val=X_val,     y_val=y_val,
                         X_test=X_test,   y_test=y_test)
    print(f"Saved HOG features to {CACHE_PATH}")

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
