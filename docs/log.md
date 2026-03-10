# Work Log

This file contains documentation/a log of a what's been done for the project. Information here might be moved to different .md files in the "docs" folder later. Each section is distinguished by a '##' header.

## Loading in the dataset

### PyTorch/torchvision:

Pros:

- Integrates directly with DataLoader, PyTorch training loops
- Built-in transform pipeline via torchvision.transforms
- Returns (image, label) tuples natively

Cons:

- Auto-download is broken — the original Stanford URL is dead. You must manually download and place files in the correct directory structure
- Requires scipy (pip install scipy) to parse .mat annotation files
- More setup friction

### HuggingFace

Pros:

- Works out of the box — data is hosted on HF Hub, no manual download
- Returns structured dicts with image (PIL) and label fields
- Easy filtering, mapping, and batching with .map() / .filter()
- Compatible with HF Trainer and also adaptable to PyTorch via .with_format("torch")

Cons:

- Multiple community-hosted versions (tanganke/stanford_cars, HuggingFaceM4/Stanford-Cars) with slight differences — verify split sizes
- Slightly more overhead to adapt to a raw PyTorch DataLoader

### Decision

Chose to go with Hugging Face over PyTorch since HuggingFace is easier right now: the torchvision auto-download is broken. HuggingFace also returns structured dicts wiwth `image` (PIL) and `label` fields and has easy filtering, mapping, and batching with `.map()`/`.filter()`. It is compatible with HF `Trainer` and also adaptable to PyTorch via `.with_format("torch")`. The cons are multiple community-hosted versions (`tanganke/stanford_cars`, `HuggingFaceM4/Stanford-Cars`) with slight differences (split sizes), and slightly mor eoverhead to adapt to a raw PyTorch `DataLoader`.

## Splitting data

Considerations:

- how many data samples to use (full 16k or half)
- train/test/validate split

### Data Samples:

- 8k: lets us compare against published papers on this benchmark, since they all use the official 8144 train / 8144 test split. Standard and easy to justify academically.
- 16k: more training data -> better models, especialy for CNN. More test data -> more reliable per-class accuracy estimates. Fine if we don't care about benchmark comparability.

Since our primary question is SVM vs. CNN rather than if we can beat the state of the art, we don't really care about benchmark comparabiliity, so using 16k is probably the better choice.

### Split Size

70/15/15 is used. 15% of 16k is ~2400 images, which is enough for tuning hyperparameters and testing.

As for the random seed: "85 because that's the BPM of Valentine" - Channing.

## HOG Configuration & Cache Strategy

HOG features are cached to disk in `data/` to avoid recomputation. The cache filename is automatically derived from the `pixels_per_cell` setting in `HOG_PARAMS` (e.g. `hog_features_16x16.npz`, `hog_features_8x8.npz`), so different configurations coexist without overwriting each other.

To switch between configurations, change `pixels_per_cell` in `HOG_PARAMS` inside `src/hog.py`:
- `(16, 16)` — coarser features, smaller vector (~3780 dims), faster SVM training
- `(8, 8)` — finer features, larger vector (~15120 dims), slower SVM training

If the cache file for the selected config already exists it will be loaded; otherwise it will be extracted and saved automatically. Running `src/svm.py` after `src/hog.py` will always use whichever config is currently set.

**Current config: 8x8** (switched from 16x16 to capture finer spatial detail).
The original 16x16 cache is preserved as `data/hog_features_16x16.npz`.

## HOG + SVM Results (16x16 pixels_per_cell)

Grid search over C ∈ {0.1, 1.0, 10.0, 100.0} and gamma ∈ {1e-4, 1e-3, 1e-2, scale} using an RBF kernel SVM. Best params found: **C=10.0, gamma=scale**.

Test set evaluation (run once after hyperparameters finalized):
- Top-1 accuracy: **9.10%**
- Top-5 accuracy: **21.71%**

Low accuracy is expected given HOG's limitations on a 196-class fine-grained dataset — it captures shape/edge information but loses colour, texture, and fine detail that distinguishes car models. The CNN side of the project should significantly outperform this baseline.

The best model is saved to `data/best_svm.pkl` and can be evaluated on new images using `src/predict.py`.

## Dependencies

| Package                            | Role                                                                     |
| ---------------------------------- | ------------------------------------------------------------------------ |
| `torch` `torchvision` `torchaudio` | CNN training, GPU acceleration, image transforms                         |
| `datasets`                         | Load Stanford Cars dataset from HuggingFace Hub                          |
| `scikit-learn`                     | SVM classifier, train/val/test splitting, evaluation metrics             |
| `scikit-image`                     | HOG feature extraction                                                   |
| `matplotlib`                       | Plotting accuracy curves, confusion matrices, misclassification analysis |
| `jupyterlab`                       | Interactive notebooks for exploration and visualization                  |

Commands needed:

- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
- `pip install datasets scikit-learn scikit-image matplotlib jupyterlab`

A requirements.txt file will be set up so no commands will have to be run EXCEPT Pytorch. Installing PyTorch is manual. This means the setup looks like:

1. Step 1 - create virtual environment and activate  
   For linux:  
   `python3 -m venv .venv`  
   `source .venv/bin/activate`

1. Step 2 - PyTorch (manual, always)  
   `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`

1. step 3 - everything else  
   `pip install -r requirements.txt`
