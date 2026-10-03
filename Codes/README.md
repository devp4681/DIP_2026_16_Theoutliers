# Codes

This directory contains the code used for the CBIR experiments in the DIP project.

## 📂 Subdirectories

- `data/` — Data loading and dataset-related code.
- `features/` — Feature extraction code for HSV, LBP, and HOG descriptors.
- `retrieval/` — Similarity calculation, ranking, and CBIR retrieval code.
- `utils/` — Utility functions used by the project.

## 📦 Dependencies

The required Python packages are listed in `requirements.txt`.

Install them using:

```bash
pip install -r requirements.txt


## 🔬 Week 1

Week 1 implemented a classical-CV CBIR baseline using:

- HSV color histogram
- LBP texture descriptor
- HOG shape/edge descriptor
- Feature fusion
- Similarity-based image ranking
- Class-based and multi-label evaluation

Feature caching was used to avoid repeated feature extraction.
