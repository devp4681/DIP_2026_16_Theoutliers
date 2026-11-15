# Workstream 4: Statistical Validation & Verification Report

**Project:** Multi-Label Content-Based Image Retrieval (CBIR) on AID Dataset  
**Phase:** Week 8 Technical Validation & Closure  
**Status:** Verified, Mathematically Reconciled, and Audited  

---

## 1. Executive Summary

This report establishes the statistical foundation of the CBIR project evaluation. It reconciles a critical discrepancy between headline Multi-Label Mean Average Precision (ML mAP) and the query-level average of `ml_ap`, audits the paired bootstrap confidence interval methodology (10,000 resamples), recalculates relevant confidence intervals, incorporates per-class Class AP confidence intervals for weak classes, compares metric trajectories between LOFO validation and final test evaluation, identifies improvements whose confidence intervals include zero, and outlines the formal statistical limitations of the study.

**Key Findings:**
1. **Mathematical Reconciliation of ML mAP:** In `week1_evaluate.py`, headline ML mAP is defined as `np.nanmean(ap)`, computing the mean exclusively over queries that possess at least one relevant gallery document ($N_{\text{rel}} > 0$). In the 600-query test set, exactly **12 queries have $N_{\text{rel}} = 0$** (Jaccard similarity with all gallery images is $< 0.8$). The headline ML mAP evaluates over the 588 valid queries, whereas simple per-query averages divide by 600. The relationship:
   $$\text{Headline ML mAP} = \text{Per-Query Mean} \times \frac{600}{588}$$
   holds with exact numerical precision ($10^{-8}$) across V2, V3, and V4.
2. **Statistically Significant Improvements (95% CI Excludes Zero):**
   * V4 vs V3 Class P@10 on Test: $+0.0315$, 95% CI $[+0.0180, +0.0450]$
   * V4 vs V3 Class mAP on Test: $+0.0653$, 95% CI $[+0.0581, +0.0728]$
   * V4 vs V3 ML mAP on Test: $+0.0152$, 95% CI $[+0.0118, +0.0187]$ (or $+0.0156$ on 588 queries)
   * V4 vs V2 on all metrics (Class P@10, ML P@10, Class mAP, ML mAP).
3. **Improvements Consistent with Zero (95% CI Includes Zero):**
   * **V4 vs V3 ML P@10 on Test:** $-0.0022$, 95% CI $[-0.0140, +0.0097]$ (**CI includes zero**).
   * **V4 vs V3 nDCG@10 on LOFO:** $+0.0012$, improved on only 2 of 5 folds (**consistent with noise**).
   * **Weak Class P@10 on Test (V4 vs V3):** Airport ($[-0.0100, +0.1200]$), Bridge ($[-0.0700, +0.0250]$), and Square ($[-0.0250, +0.1550]$) all have 95% confidence intervals spanning zero due to the small sample size ($N=20$).

---

## 2. Reconciliation of Headline ML mAP vs Per-Query `ml_ap`

### 2.1. The Root Cause of the Discrepancy
When comparing `final_overall_results_v2_v3_v4.csv` with `per_query_v2_v3_v4.csv` and `paired_bootstrap.csv`, a numerical divergence was identified:
* In `final_overall_results_v2_v3_v4.csv`, V4 ML mAP is reported as **0.2716**.
* In `per_query_v2_v3_v4.csv`, the mean of column `(V4, ml_ap)` is **0.2662**.
* In `paired_bootstrap.csv`, the observed difference for V4 - V3 `ml_ap` is reported as **0.015244**, whereas the difference between the headline values ($0.271605 - 0.256050$) is **0.015555**.

### 2.2. Mathematical Proof of Equivalence
Inspection of `average_precision` in `Week1/week1_evaluate.py` reveals:
```python
def average_precision(rel_sorted):
  r = rel_sorted.astype(np.float64)
  cum = np.cumsum(r, axis=1)
  prec = cum / np.arange(1, r.shape[1] + 1)
  n_rel = r.sum(1)
  ok = n_rel > 0
  ap = np.full(r.shape[0], np.nan)
  ap[ok] = (prec[ok] * r[ok]).sum(1) / n_rel[ok]
  return np.nanmean(ap)
```
1. In information retrieval evaluation, if a query image has zero relevant items in the gallery ($N_{\text{rel}} = 0$), Precision at any recall level is undefined. The standard TREC / PASCAL VOC convention marks these queries as `NaN` and omits them from the macro-average denominator.
2. In the AID test set (600 queries vs 2,400 gallery images), multi-label relevance is defined as Jaccard similarity $J(L_q, L_g) \ge 0.8$. Exactly **12 queries** have no gallery images satisfying this condition:
   * `bareland_81`, `bareland_84`, `bareland_92`, `bareland_95`, `bareland_96` (5 queries)
   * `farmland_91`, `farmland_92` (2 queries)
   * `storagetanks_86`, `storagetanks_94` (2 queries)
   * `bridge_90` (1 query)
   * `center_93` (1 query)
   * `playground_85` (1 query)
3. In `per_query_v2_v3_v4.csv`, these 12 undefined queries were assigned an imputed score of `0.0` instead of `NaN`.
4. When a simple average is computed across all 600 rows of `per_query_v2_v3_v4.csv`, the sum is divided by 600 instead of 588:

$$\text{Mean}_{600} = \frac{1}{600} \sum_{i=1}^{588} \text{AP}_i + \frac{1}{600} \sum_{j=1}^{12} 0.0 = \frac{588}{600} \cdot \text{Headline ML mAP}$$

$$\text{Headline ML mAP} = \text{Mean}_{600} \times \frac{600}{588} = \text{Mean}_{600} \times \frac{50}{49}$$

### Numerical Verification:

| Model | Sum of per-query `ml_ap` | Mean across all 600 queries | Mean across 588 valid queries | Headline ML mAP | Absolute Difference |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **V2** | 133.930269 | 0.223217 | **0.227773** | 0.227773 | $0.00000000$ |
| **V3** | 150.557466 | 0.250929 | **0.256050** | 0.256050 | $0.00000000$ |
| **V4** | 159.703695 | 0.266173 | **0.271605** | 0.271605 | $0.00000000$ |

The two metrics are mathematically identical under their respective denominator definitions. Both definitions are methodologically valid, but they must be explicitly labeled:
* **Headline ML mAP (Macro-Average over Queries with Relevant Items):** Divides by $N=588$.
* **Imputed Query-Level Mean (All Queries):** Divides by $N=600$.

---

## 3. Audited Confidence Intervals (10,000 Paired Bootstrap Iterations)

### 3.1. Final Test Set: V4 vs V3 and V4 vs V2 ($N=600$ Queries)

The table below presents the verified 95% paired bootstrap confidence intervals (percentile method, 10,000 resamples, seed 0):

| Comparison | Metric | Observed Difference | 95% CI Lower | 95% CI Upper | Queries Improved | Queries Worsened | CI Excludes Zero? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V4 - V3** | **Class P@10** | **+0.0315** | **+0.0180** | **+0.0450** | 179 | 138 | **Yes (Significant)** |
| | **Class mAP** | **+0.0653** | **+0.0581** | **+0.0728** | 416 | 184 | **Yes (Significant)** |
| | **ML P@10** | **-0.0022** | **-0.0140** | **+0.0097** | 171 | 175 | **No (Includes Zero)** |
| | **ML mAP (600q)**| **+0.0152** | **+0.0118** | **+0.0187** | 327 | 260 | **Yes (Significant)** |
| | *ML mAP (588q)*| *+0.0156* | *+0.0120* | *+0.0191* | 327 | 260 | *Yes (Significant)* |
| **V4 - V2** | **Class P@10** | **+0.0800** | **+0.0628** | **+0.0973** | 253 | 119 | **Yes (Significant)** |
| | **Class mAP** | **+0.1039** | **+0.0925** | **+0.1153** | 449 | 151 | **Yes (Significant)** |
| | **ML P@10** | **+0.0353** | **+0.0210** | **+0.0498** | 220 | 152 | **Yes (Significant)** |
| | **ML mAP (600q)**| **+0.0430** | **+0.0378** | **+0.0486** | 491 | 97 | **Yes (Significant)** |

### 3.2. 5-Fold LOFO Cross-Validation ($N=2,400$ Pooled Queries)

| Comparison | Metric | Observed Difference | 95% CI Lower | 95% CI Upper | Queries Improved | Queries Worsened | CI Excludes Zero? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V4 - V3** | **Class P@10** | **+0.0292** | **+0.0226** | **+0.0360** | 840 | 579 | **Yes (Significant)** |
| | **Class mAP** | **+0.0659** | **+0.0620** | **+0.0698** | 1706 | 694 | **Yes (Significant)** |
| | **ML P@10** | **+0.0070** | **+0.0014** | **+0.0129** | 762 | 720 | **Barely above zero** |
| | **ML mAP** | **+0.0198** | **+0.0175** | **+0.0221** | 1362 | 1016 | **Yes (Significant)** |
| **V4 - V2** | **Class P@10** | **+0.1025** | **+0.0935** | **+0.1117** | 1233 | 465 | **Yes (Significant)** |
| | **Class mAP** | **+0.1083** | **+0.1027** | **+0.1140** | 1860 | 540 | **Yes (Significant)** |
| | **ML P@10** | **+0.0546** | **+0.0473** | **+0.0619** | 1048 | 539 | **Yes (Significant)** |
| | **ML mAP** | **+0.0522** | **+0.0490** | **+0.0554** | 1984 | 395 | **Yes (Significant)** |

---

## 4. Per-Class Confidence Intervals for Weak Classes

For the three historically weak categories identified in Weeks 6–7 (**Airport, Bridge, Square**), 95% bootstrap confidence intervals were computed individually for both Class P@10 and Class AP across LOFO (80 queries) and Final Test (20 queries):

### 4.1. Final Test Set ($N=20$ Queries per Class)

| Weak Class | Metric | V3 Mean | V4 Mean | Delta (V4 - V3) | 95% CI Lower | 95% CI Upper | CI Excludes Zero? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Airport** | **Class P@10** | 0.0700 | 0.1200 | **+0.0500** | -0.0100 | +0.1200 | **No (Includes Zero)** |
| | **Class AP** | 0.0580 | 0.0683 | **+0.0103** | +0.0004 | +0.0229 | **Yes (Borderline)** |
| **Bridge** | **Class P@10** | 0.1050 | 0.0850 | **-0.0200** | -0.0700 | +0.0250 | **No (Includes Zero)** |
| | **Class AP** | 0.0415 | 0.0459 | **+0.0044** | +0.0008 | +0.0083 | **Yes (Small gain)** |
| **Square** | **Class P@10** | 0.1500 | 0.2050 | **+0.0550** | -0.0250 | +0.1550 | **No (Includes Zero)** |
| | **Class AP** | 0.0668 | 0.0820 | **+0.0152** | +0.0019 | +0.0309 | **Yes (Significant)** |

### 4.2. LOFO Cross-Validation ($N=80$ Queries per Class)

| Weak Class | Metric | V3 Mean | V4 Mean | Delta (V4 - V3) | 95% CI Lower | 95% CI Upper | CI Excludes Zero? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Airport** | **Class P@10** | 0.3038 | 0.3638 | **+0.0600** | +0.0250 | +0.0963 | **Yes (Significant)** |
| | **Class AP** | 0.1514 | 0.1945 | **+0.0432** | +0.0311 | +0.0566 | **Yes (Significant)** |
| **Bridge** | **Class P@10** | 0.1562 | 0.1738 | **+0.0175** | -0.0050 | +0.0412 | **No (Includes Zero)** |
| | **Class AP** | 0.0673 | 0.0844 | **+0.0171** | +0.0126 | +0.0219 | **Yes (Significant)** |
| **Square** | **Class P@10** | 0.2125 | 0.2262 | **+0.0138** | -0.0113 | +0.0412 | **No (Includes Zero)** |
| | **Class AP** | 0.0778 | 0.0889 | **+0.0110** | +0.0066 | +0.0156 | **Yes (Significant)** |

### Critical Analytical Insights:
1. **P@10 Uncertainty on Weak Classes:** On the final test set, **not a single weak class achieved a Class P@10 confidence interval that excludes zero**. For Bridge, the observed delta is actually negative ($-0.0200$, CI $[-0.0700, +0.0250]$). On LOFO, both Bridge and Square also include zero on Class P@10. Claims that V4 systematically resolves precision for Bridge and Square are **statistically unsupported**.
2. **AP Stability:** In contrast, Class AP shows consistent, statistically significant positive intervals across all three classes on both LOFO and Test. This confirms that $k$-reciprocal re-ranking successfully moves true class exemplars up from distant ranks into intermediate ranks, even when they do not break into the top 10.

---

## 5. Identification of Improvements Consistent with Zero

To prevent scientific overclaiming during the final presentation and viva, the following experimental deltas are explicitly identified as **not statistically significant** (confidence interval includes zero or cross-fold sign test fails):

1. **V4 vs V3 ML P@10 on Final Test:** $\Delta = -0.0022$, 95% CI $[-0.0140, +0.0097]$. 171 queries improved, 175 worsened. The effect is pure noise.
2. **V4 vs V3 nDCG@10 on LOFO:** Mean delta $= +0.0012$. Positive on only 2 of 5 folds ($p = 0.50$, sign test). V4 does not improve multi-label ranking discount over V3.
3. **V4 vs V3 Top-5 Ranking Metrics (Jaccard@5, nDCG@5):**
   * Jaccard@5 LOFO delta: $-0.0016$ (improved on only 1 of 5 folds).
   * nDCG@5 LOFO delta: $-0.0036$ (improved on 0 of 5 folds).
   * Jaccard@5 Test delta: $-0.0075$.
   * nDCG@5 Test delta: $-0.0071$.
   * At $K=5$, V4 is slightly *inferior* to V3 because reciprocal re-ranking occasionally displaces a relevant multi-label neighbor at rank 1–3 in favor of a verified scene-class match.
4. **Weak Class P@10 Gains on Final Test:**
   * Airport P@10 delta $+0.0500$ (CI $[-0.0100, +0.1200]$)
   * Bridge P@10 delta $-0.0200$ (CI $[-0.0700, +0.0250]$)
   * Square P@10 delta $+0.0550$ (CI $[-0.0250, +0.1550]$)
   All three intervals span zero.

---

## 6. Formal Statistical Limitations

1. **Sample Size Limitations:**
   * 5-fold LOFO cross-validation evaluates on 5 data partitions. While aggregating 2,400 queries provides strong query-level statistical power, the 5 fold-level samples provide limited power for fold-level $t$-tests ($df = 4$).
   * The final test partition contains only 20 queries per class. Class-level test metrics are susceptible to high sampling variance ($\pm 5\%$ per single image).
2. **Query Independence Assumption:**
   * Bootstrap confidence intervals treat queries as independent and identically distributed (i.i.d.). In practice, queries evaluated against a shared gallery exhibit weak covariance through shared competition for gallery items. The reported bootstrap intervals are therefore approximate.
3. **Repeated Test Set History:**
   * The 600-image test set was evaluated in Week 1 (early feature checks) and Week 4 (frozen V2). While V3 and V4 parameters were tuned strictly on LOFO training data, the test set cannot be claimed as a completely blind holdout for the multi-week lifecycle.
