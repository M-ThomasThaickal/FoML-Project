# FoML Project

This repository contains our Stanford Cars classification project for Foundations of Machine Learning.

Link to Slide Presentation: https://docs.google.com/presentation/d/1NSTjKjHbLSs4XhLwYnHQLGozTOz36pf9hPSKtF0lvAU/edit?usp=sharing 

The main comparison in the project is between classical SVM pipelines and transfer-learned CNNs:

- `HOG + SVM (RBF)`
- `HOG + SVM (RBF, balanced class weights)`
- `HOG + SVM (linear)`
- `Raw pixels + SVM`
- baseline CNN
- improved CNN

## Repository layout

- [`src/`](src/README.md)  
  Main project code for feature extraction, SVM training, CNN training, prediction, and visualization.

- [`docs/`](docs/README.md)  
  Project writeups, workflow notes, report material, and supporting documentation.

- `notebooks/`  
  Exploratory analysis and comparison notebooks.

- `entrypoint/`  
  Dataset loading and split setup used by the training scripts.

- `data/`  
  Cached features, saved models, checkpoints, and metric outputs.

## Where to start

If you want the full project run order, start with:

- [`docs/workflow.md`](docs/workflow.md)

If you only want the code overview for the scripts in `src/`, see:

- [`src/README.md`](src/README.md)

If you are focused on the CNN side, see:

- [`docs/cnn_workflow.md`](docs/cnn_workflow.md)

## Notes

- The dataset loading logic lives in `entrypoint/load.py`.
- Most outputs are written under `data/`.
- Some helper scripts in `src/` assume specific saved model or cache filenames, so the workflow docs are the safer reference if something looks inconsistent.
