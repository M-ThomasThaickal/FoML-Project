import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "entrypoint"))
from skimage.color import rgb2gray
from skimage.transform import resize
from load import train_ds, val_ds, test_ds
from progress_utils import progress_bar

IMG_SIZE   = (64, 64)
CACHE_PATH = DATA_DIR / "raw_pixel_features_64x64.npz"


def extract_pixels(dataset, split_name):
    X, y = [], []
    for sample in progress_bar(dataset, desc=f"Raw pixels {split_name}", total=len(dataset)):
        img = np.array(sample["image"])
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = resize(img, IMG_SIZE, anti_aliasing=True)
        img_gray = rgb2gray(img)          # 64x64 -> 4096 dims
        X.append(img_gray.flatten())
        y.append(sample["label"])
    return np.array(X), np.array(y)


if CACHE_PATH.exists():
    print("Loading raw pixel features from cache...")
    data = np.load(CACHE_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
else:
    print("Extracting raw pixel features...")
    X_train, y_train = extract_pixels(train_ds, "train")
    X_val,   y_val   = extract_pixels(val_ds, "val")
    X_test,  y_test  = extract_pixels(test_ds, "test")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, X_train=X_train, y_train=y_train,
                         X_val=X_val,     y_val=y_val,
                         X_test=X_test,   y_test=y_test)
    print(f"Saved raw pixel features to {CACHE_PATH}")

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
