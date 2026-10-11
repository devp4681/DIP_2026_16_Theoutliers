# Week 1 — Project Overview & Baseline Retrieval Documentation

## 1. Dataset & Terminology Distinction

To avoid ambiguity throughout the project lifecycle, note the clear distinction between scene classes and semantic labels:
* **SCENE CLASS (30 Categories)**: The original AID aerial dataset contains 30 distinct scene categories (`airport`, `bareland`, `baseballfield`, `beach`, `bridge`, `center`, `church`, `commercial`, `denseresidential`, `desert`, `farmland`, `forest`, `industrial`, `meadow`, `mediumresidential`, `mountain`, `park`, `parking`, `playground`, `pond`, `port`, `railwaystation`, `resort`, `river`, `school`, `sparseresidential`, `square`, `stadium`, `storagetanks`, `viaduct`). Each image belongs to exactly one primary scene class.
* **SEMANTIC LABELS (17 Categories)**: The multi-label aerial benchmark provides multi-hot binary ground-truth vectors (`multilabel .csv`, 3000 images $\times$ 17 semantic attributes) indicating the co-presence of 17 object and land-cover categories: `airplane`, `bare-soil`, `buildings`, `cars`, `chaparral`, `court`, `dock`, `field`, `grass`, `mobile-home`, `pavement`, `sand`, `sea`, `ship`, `tanks`, `trees`, `water`.

### Split Statistics:
* Total images: 3,000 (600 $\times$ 600 pixels)
* Training gallery: 2,400 images (80 images per scene class)
* Test queries: 600 images (20 images per scene class)

---

## 2. Handcrafted Feature Descriptors

Three baseline handcrafted visual descriptors were extracted and cached:
1. **Color (HSV Histogram)**: $16 \times 16$ 2D histogram of Hue and Saturation channels (256-D, normalized to unit sum). Measured via Pearson correlation distance.
2. **Texture (LBP)**: Uniform Local Binary Patterns ($P=8, R=1$, 10 bins). Normalized to unit sum and compared via Pearson correlation distance.
3. **Shape / Structure (HOG)**: Histogram of Oriented Gradients computed on resized $128 \times 128$ grayscale images ($9$ orientations, $16 \times 16$ cells, $2 \times 2$ blocks per cell, 1764-D). Normalized with L2-Hys and compared via Cosine similarity.

---

## 3. Baseline Fusion Configurations & "Week 1 Best"

In Week 1, eight linear fusion combinations were evaluated against the 600 test queries. The top-performing baseline on the test set was:
* **Week 1 Best**: $\text{HSV} = 0.50, \text{LBP} = 0.25, \text{HOG} = 0.25$
* Test Performance:
  * Class P@5 = 0.3687, Class P@10 = 0.3345, Class mAP = 0.1699
  * ML P@5 = 0.3603, ML P@10 = 0.3483, ML mAP = 0.2296
  * Jaccard@10 = 0.6273, nDCG@10 = 0.6527

---

## 4. Methodological Note on Evaluation Protocol

As documented across project reports, intermediate evaluation on the test set in Week 1 selected the "Week 1 best" baseline. Subsequent development (Week 2 V1 and Week 4 V2) built upon these baselines. The final frozen test results in `Week4/final_test/final_overall_results.csv` reflect empirical comparisons across the final models rather than untouched holdout generalization.
