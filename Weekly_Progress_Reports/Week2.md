# Week 2 Progress Report

**Date Range**: September 14 – September 20, 2026  
**Milestone Focus**: Dataset Exploratory Analysis & Multi-Feature Extraction Pipeline (Color, Texture, Shape)

---

## 1. Objectives for the Week
1. Parse and inspect the AID Multi-Label Dataset annotations (3,000 images, 17 multi-label categories, 600×600 pixel aerial resolution).
2. Develop handcrafted feature extractors for classical DIP representations:
   - **Color**: 2D HSV Hue-Saturation histogram ($16 \times 16 = 256$ bins) & color moments.
   - **Texture**: Local Binary Patterns (LBP) with uniform pattern encoding ($P=8, R=1 \rightarrow 10$ bins).
   - **Shape**: Histogram of Oriented Gradients (HOG) with cell size $16 \times 16$, 9 orientation bins.
3. Build independent similarity matching functions and benchmark each feature individually.

---

## 2. Key Implementations & Progress

### A. Dataset Loading & Label Analysis
- Loaded `multilabel .csv` containing binary indicator vectors for 17 classes: `airplane`, `bare-soil`, `buildings`, `cars`, `chaparral`, `court`, `dock`, `field`, `grass`, `mobile-home`, `pavement`, `sand`, `sea`, `ship`, `tanks`, `trees`, `water`.
- Observed multi-label co-occurrence patterns: e.g., `buildings` heavily co-occurs with `cars` and `pavement`; `dock` co-occurs with `water` and `ship`.

### B. Feature Extraction Modules
- **Color Module (`Codes/features/color.py`)**:
  - Implemented 2D Hue-Saturation normalized histogram using OpenCV's `calcHist` in HSV color space.
  - Implemented first 3 color moments (Mean, Standard Deviation, Skewness) per channel across RGB and HSV spaces.
- **Texture Module (`Codes/features/texture.py`)**:
  - Implemented Uniform LBP descriptor with $P=8, R=1$, providing rotation-invariant micro-texture features.
  - Added Gray-Level Co-occurrence Matrix (GLCM) feature extraction capturing Contrast, Dissimilarity, Homogeneity, Energy, and Correlation.
- **Shape Module (`Codes/features/shape.py`)**:
  - Implemented Edge Direction Histogram (EDH) via Sobel gradient operators ($G_x, G_y$).
  - Implemented standard HOG descriptor on resized $128 \times 128$ aerial patches.

---

## 3. Preliminary Experimental Findings

Each feature was evaluated independently using histogram correlation / cosine similarity on test queries:
- **Color (HSV alone)**:
  - Strong retrieval for chromatically homogeneous scenes (e.g., `water`, `sea`, `sand`, `bare-soil`).
  - Struggles when different scene categories share identical vegetation or ground color palettes (e.g., confusing `court` with `field` or `grass`).
- **Texture (LBP alone)**:
  - Successfully identifies repetitive structural textures like roof density, agricultural furrows, and road grids.
  - Insensitive to color variations, but lacks global structural context.
- **Shape (HOG alone)**:
  - Captures dominant directional geometry, but sensitive to aerial rotation and scale shifts without color/texture cues.

---

## 4. Challenges & Solutions
- **JPEG Truncation Warnings**: Direct decoding with OpenCV occasionally raised truncation warnings on specific aerial images. Resolved by introducing `safe_load_image` via PIL before converting to NumPy arrays.
- **Dimensionality Balance**: HOG feature dimensions are substantially larger than LBP or HSV. Normalization ($L_2$-norm) was enforced across all descriptors.

---

## 5. Team Task Split & Contributions

| Member | Focus & Contributions This Week |
|---|---|
| **devp4681 (Dev)** | Dataset annotation parsing, PIL safe-loader implementation, exploratory class co-occurrence analysis. |
| **dhruvif16 (Dhruvi)** | Implementation and verification of HSV color histogram, color moments, and Uniform LBP extraction. |
| **aastha280406 (Aastha)** | Implementation of HOG descriptor and Sobel edge direction histograms. |
| **Aashilearnstocode (Aashi)** | Experimental logging, feature dimensionality analysis, and documentation. |

---

## 6. Next Steps for Week 3
- Establish a rigorous evaluation protocol (stratified 80/20 train/validation split) to prevent data leakage.
- Implement comprehensive multi-label evaluation metrics: Precision@K, Multi-Label Precision@K, Jaccard Similarity / IoU @ K, and mAP.
