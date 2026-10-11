# Final Project Results & Technical Documentation

**V2 is the final frozen model.**

---

## 1. Frozen Final Model Architecture & Weights

The final retrieval engine (**V2**) fuses four global handcrafted feature descriptors with frozen linear weights:
* **HSV Color Histogram (16x16 bins)**: 0.10
* **LBP Texture (Radius 1, Points 8, uniform)**: 0.35
* **LBP Texture (Radius 2, Points 16, uniform)**: 0.35
* **HOG Shape (128x128, 9 orientations, 16x16 cell, 2x2 block)**: 0.20

**Authoritative configuration file**: `Week4/v2_config.json`

---

## 2. Authoritative Final Overall Results

The single authoritative final results table is stored in `Week4/final_test/final_overall_results.csv`:

| Model | Class P@5 | ML P@5 | Jaccard@5 | nDCG@5 | Class P@10 | ML P@10 | Jaccard@10 | nDCG@10 | Class mAP | ML mAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline** | — | — | — | — | — | — | ≈ 0.4440 | ≈ 0.4580 | — | — |
| **Week 1 best** | 0.3687 | 0.3603 | 0.6405 | 0.6598 | 0.3345 | 0.3483 | 0.6273 | 0.6527 | 0.1699 | **0.2296** |
| **V1 (Week 2)** | **0.4187** | 0.3803 | 0.6585 | 0.6761 | 0.3733 | 0.3593 | 0.6453 | 0.6699 | **0.1891** | 0.2262 |
| **V2 (Final Frozen)** | 0.4183 | **0.3857** | **0.6628** | **0.6806** | **0.3867** | **0.3650** | **0.6491** | **0.6740** | 0.1889 | 0.2278 |

*Note: Bold indicates the highest performing model for each metric. V1 achieves highest Class P@5 and Class mAP; Week 1 best achieves highest ML mAP; V2 achieves highest Class P@10, ML P@5, ML P@10, Jaccard@K, and nDCG@K. All values correspond exactly to `Week4/final_test/final_overall_results.csv` and remain strictly frozen.*

---

## 3. Key Findings & Performance Interpretation

### A. V2 vs V1 Performance Comparison
* **Class P@10 Gain**:
  * Observed mean difference: **+0.0133** (+1.33% across all 600 test queries).
  * 95% bootstrap confidence interval (10,000 resamples): **[0.0050, 0.0213]**.
  * The gain is **positive and modest, and the interval excludes zero under the assumption of independent queries**, noting that the test set was accessed during intermediate development stages.
* **Multi-Label Metric Precision (ML P@K)**:
  * V2 improves ML P@5 (0.3857 vs 0.3803) and ML P@10 (0.3650 vs 0.3593) relative to V1, but the multi-label improvement is not statistically established.
  * **ML mAP remains below the Week 1 best baseline**: Week 1 best achieved ML mAP = 0.2296, whereas V1 reached 0.2262 and V2 reached 0.2278. Therefore, V2 does **not** improve ML mAP over the Week 1 best baseline.

### B. Per-Class Performance & Modest Class Shifts
* Under V2 vs V1 on the test set:
  * **22 classes improved**
  * **6 classes worsened**
  * **2 classes unchanged**
* While a paired sign test indicates a consistent positive shift direction ($p = 0.0036$), **individual class changes must NOT be termed individually significant**:
  * There are only 20 test queries per scene class.
  * Approximately 15 of the 22 improved classes changed by $\le 0.02$ (a difference of just **1 to 4 images out of 200 retrieved slots** across the 20 queries for that class).
  * The correct technical interpretation is that **V2 shows a broad but modest shift in class-level P@10, with many individual class changes being small**.
* **Main Weakest Classes** (by V2 test Class P@10):
  1. `playground`: **0.025**
  2. `airport`: **0.080**
  3. `river`: **0.080**
  4. `square`: **0.100**
  5. `bridge`: **0.115**
  *(Other difficult classes include `stadium` at 0.145, `viaduct` at 0.155, and `railwaystation` at 0.200. Classes such as `airport`, `bridge`, and `square` suffer from large within-class appearance variations, runway/apron texture confusion, and structural overlaps with urban roads and plazas).*

---

## 4. Evaluation Protocol & Methodological Limitation

> [!IMPORTANT]
> **Test-Set Access Disclosure**:
> An important methodological limitation is that the original test set was accessed during intermediate development stages:
> 1. In **Week 1**, multiple fusion configurations (8 combinations) were evaluated directly on the test set, and the "Week 1 best" baseline (HSV 0.5 + LBP 0.25 + HOG 0.25) was retained based on those test results.
> 2. In **Week 2**, Engine V1 was evaluated on the test set prior to the final V2 development.
> 3. In **Week 4**, the final V2 evaluation was performed.
> 
> Therefore, the test set was not used as a strictly untouched, blind holdout throughout the multi-week lifecycle. The final V2 evaluation should consequently be interpreted as a final empirical comparison rather than a completely blind generalization estimate. This disclosure accurately reflects our engineering process and does not alter the empirical numbers reported.

### Note on Evaluation Script Versioning
The final test script was run in two historical versions:
* `final_test_old.py` (now archived in `Week4/final_test/archive/`): Early experimental script that used binary-class nDCG.
* `final_test.py`: Active authoritative script (printing `"--- CORRECTED OVERALL METRICS ---"`). The models remained strictly frozen throughout, but the evaluation code was corrected to use graded Jaccard-based nDCG over the full gallery and the standardized model list.

---

## 5. Explainability Framework & Component Similarity Evidence

* **Nature of the Explanation**:
  * The explanation system provides **component-level similarity evidence** (or *relative similarity evidence for the retrieved image*), showing how strongly each handcrafted descriptor agrees with the query relative to the background gallery distribution.
  * It does **NOT** provide causal feature attribution. We do not claim that a specific feature "caused" an image to be retrieved.
  * These values represent **empirical percentile ranks**, not "feature importance".
* **Key Empirical Observation**:
  * Across retrieved top-5 results, approximately **92% exhibit HSV similarity percentiles $\ge 95$**, and LBP-r1 percentiles are likewise heavily concentrated at high values.
  * HOG percentiles show much greater dispersion.
  * This confirms that retrieved images largely achieve high scores through concordant agreement across multiple features rather than any single dominant descriptor.

---

## 6. Scope Limitations & Task Definitions

* **Global Descriptors vs Region Attribution**:
  * The implemented explainability analysis provides global component-level similarity evidence (HSV, LBP, and HOG).
  * Region-level spatial visual attribution was not implemented because the retrieval system relies entirely on global handcrafted feature representations rather than localized deep bounding boxes or dense spatial feature maps.
* **Multi-Label Relevance Definition**:
  * Multi-label relevance was evaluated using the fixed criterion:
    $$\text{Relevance} = \mathbb{I}(\text{Jaccard Similarity} \ge 0.8)$$
  * The 0.8 threshold enforces high semantic overlap between query and retrieved image annotation vectors.

---

## 7. Graded nDCG Metric Formulation

* **Definition**:
  * In our evaluation pipeline (`Week1/week1_evaluate.py`), nDCG is computed using **graded multi-label relevance gains**:
    $$\text{gain}_i = \text{Jaccard}(L_q, L_{g_i})$$
  * The Ideal Discounted Cumulative Gain ($\text{IDCG}$) is calculated by sorting all true Jaccard gains over the **entire gallery** in descending order.
  * This differs from older archived binary-class nDCG implementations where gain was strictly 1 if scene classes matched and 0 otherwise. Binary-class formulations naturally yield substantially lower numerical values because non-matching scene classes contribute zero gain even if they share multi-label attributes.
* **Contextual Baselines**:
  * Under random gallery ranking:
    * $\text{Random nDCG@10} \approx 0.4580$
    * $\text{Random Jaccard@10} \approx 0.4440$
  * Because Jaccard-based nDCG operates on continuous semantic overlap scores bounded by dataset label co-occurrence statistics, its numerical scale reflects the underlying label density. The observed gains (0.6527 $\rightarrow$ 0.6740) represent meaningful semantic ranking improvements over both random guessing and the Week 1 baseline, but should not be overstated as near-perfect ranking.

---

## 8. Validation-to-Test Performance Gap

* **Observed Metrics**:
  * V2 5-Fold Cross-Validation Class P@10 $\approx$ **0.4340**
  * V2 Held-Out Test Class P@10 = **0.3867**
* **Technical Factors & Hypotheses**:
  * While gallery size differed (1,920 validation vs 2,400 test), the proportion of same-class images remained constant (80 per class vs 64 in validation folds), so gallery scale alone does not explain the difference.
  * **Working Hypothesis**: Train-only validation folds may contain visually similar or near-neighbor images originating from identical source scene distributions or acquisition tiles, whereas the held-out test split captures greater visual and environmental variation. This remains an empirical hypothesis and was not independently verified.

---

## 9. Reliability Scoring Analysis

* **Formula & Operational Definition**:
  * Final frozen score:
    $$\text{reliability\_score} = \text{top1\_similarity}$$
  * Derived thresholds (from validation 20th and 70th percentiles):
    * **LOW**: $\text{score} < 0.930$
    * **MEDIUM**: $0.930 \le \text{score} < 0.960$
    * **HIGH**: $\text{score} \ge 0.960$
* **Validation vs Test Transfer**:
  * Validation HIGH/MEDIUM/LOW partitions represent in-sample threshold calibration.
  * Evaluating these fixed thresholds on the held-out test set provides honest transfer assessment:
    * **HIGH (24.7% of queries, n=148)**: Class P@10 = **0.6088**, ML P@10 = **0.4824**
    * **MEDIUM (50.7% of queries, n=304)**: Class P@10 = **0.3299**, ML P@10 = **0.3724**
    * **LOW (24.7% of queries, n=148)**: Class P@10 = **0.2811**, ML P@10 = **0.2324**
* **Correlation Metrics**:
  * Spearman rank correlation with Class P@10: $\approx 0.40$
  * **AUC $\approx$ 0.70 for predicting Class P@10 $\ge$ 0.5**
  * Within-class Spearman correlation: $\approx 0.12$
* **Interpretation**:
  * Reliability is an **empirical coarse retrieval-difficulty indicator**, **NOT** a calibrated posterior probability of retrieval correctness, and **NOT** guaranteed confidence.
  * It is partially class-driven: visually homogeneous classes (e.g. `parking`, `bareland`) naturally achieve higher top-1 similarities, whereas heterogeneous classes (e.g. `bridge`, `airport`) systematically yield lower scores.

---

## 10. Multi-Label Precision & Class-Specific Chance Baselines

> [!NOTE]
> **Key Finding**: ML precision is low where label sets vary within a class, and the $\text{Jaccard} \ge 0.8$ criterion is strict.

* **Class-Specific Chance of $\text{Jaccard} \ge 0.8$**:
  * Under random gallery sampling, the probability of drawing an image with $\text{Jaccard} \ge 0.8$ against test queries varies dramatically by class:
    * `farmland`: chance is **0.2%** (0.002) $\rightarrow$ observed ML P@10 = **0.015** is **$\approx 7.5\times$ chance**.
    * `river`: chance is **0.8%** (0.008) $\rightarrow$ observed ML P@10 = **0.030** is **$\approx 3.8\times$ chance**.
    * **All 30 classes perform strictly above their class-specific random chance baseline.**
* **Why Farmland and River ML Precision is Numerically Low**:
  * For `farmland`: Every farmland query is annotated solely with `field`. However, the gallery's 80 farmland images contain 25 different label subsets (the most frequent subset accounts for only 11% of images, with varying combinations of `bare-soil`, `buildings`, `trees`, and `water`). Even correct-class retrieved farmland images achieve a mean Jaccard of only 0.40, with only 5% reaching $\ge 0.8$. Wrong-class retrievals reach 0% at $\ge 0.8$.
  * The $\text{Jaccard} \ge 0.8$ threshold is exceptionally strict for classes with varied multi-label annotations.
  * For built-up classes with varied label combinations, the empirical lift over chance is modest (**$1.5\times$ to $2.1\times$** chance: `playground` $1.5\times$, `commercial` $1.6\times$, `railwaystation` $1.6\times$).
  * For natural classes with consistent label sets, the lift over chance is substantially higher (**$17\times$ to $21\times$** chance: `desert` $21\times$, `forest` $20\times$, `mountain` $17\times$). This accounts for the apparent disparity between class precision and multi-label precision.

---

## 11. Dataset Integrity Note: Training Image `forest_2`

* In the training gallery, image `forest_2` has an all-zero label vector in `multilabel_.csv`.
* Consequently, its Jaccard similarity with any query is strictly 0.0.
* Because `forest_2` is only 1 image out of 2,400 in the gallery, this anomaly is practically harmless to overall retrieval performance, but is formally noted for complete data integrity.
