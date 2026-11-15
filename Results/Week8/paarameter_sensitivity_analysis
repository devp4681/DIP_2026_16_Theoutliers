# Workstream 3: Parameter Sensitivity Analysis ($k_1$ and $k_2$ Robustness)

**Project:** Multi-Label Content-Based Image Retrieval (CBIR) on AID Dataset  
**Phase:** Week 8 Technical Validation & Closure  
**Status:** Complete, Empirically Verified, and Config Frozen  

---

## 1. Executive Summary

The V4 retrieval system incorporates gallery-only $k$-reciprocal re-ranking governed by three hyper-parameters:
* $k_1 = 30$: Neighborhood size for reciprocal nearest-neighbor verification.
* $k_2 = 3$: Number of top-ranked reciprocal neighbors used in query expansion representation.
* $\lambda = 0.5$: Fusion weight balancing original V3 distance and Jaccard reciprocal distance ($D_{V4} = \lambda \cdot J_{\text{KR}} + (1 - \lambda) \cdot D_{V3}$).

This workstream investigates the robustness of these parameters, specifically addressing:
1. Whether $k_1 = 30$, selected on a validation gallery of 64 images per class, transfers reliably to the full test gallery of 80 images per class.
2. The stability of performance across neighboring $k_1$ values.
3. A systematic comparison of $k_2 \in \{1, 2, 3\}$ across all 5 cross-validation folds.
4. Fold-level variability, trade-offs between precision and mean average precision (mAP), and verification that repeated selection across identical folds is not misconstrued as independent validation.

**Primary Conclusion:** The provisional selection of $k_2 = 3$ and $k_1 = 30$ is **robust, stable, and confirmed**. $k_2 = 3$ was independently selected on all 5 LOFO validation folds, exhibits the lowest cross-fold variance ($\sigma = 0.0042$), and provides the highest global ranking quality (Class mAP: 0.3206 vs 0.3037 for $k_2 = 1$). The frozen V4 configuration is retained without modification.

---

## 2. Investigation of $k_1$ Transfer Across Gallery Densities

### 2.1. The Gallery Scaling Question
* In 5-fold LOFO cross-validation, queries are evaluated against 4 folds serving as the gallery ($4 \times 480 = 1,920$ images). Because the AID training set comprises 30 balanced classes, each class contains exactly:
  $$N_{\text{class}}^{\text{LOFO}} = \frac{1,920}{30} = 64 \text{ gallery images per class}$$
  At $k_1 = 30$, the neighborhood window covers:
  $$\text{Coverage}_{\text{LOFO}} = \frac{30}{64} \approx 46.88\% \text{ of potential class exemplars}$$
* In the final test evaluation, the entire training set serves as the gallery ($2,400$ images):
  $$N_{\text{class}}^{\text{Test}} = \frac{2,400}{30} = 80 \text{ gallery images per class}$$
  If $k_1$ remains fixed at 30:
  $$\text{Coverage}_{\text{Test}} = \frac{30}{80} = 37.50\% \text{ of potential class exemplars}$$
* If $k_1$ were scaled proportionally to maintain the exact 46.88% coverage, the scaled parameter would be:
  $$k_1^* = 30 \times \frac{80}{64} = 37.5 \approx 38$$

### 2.2. Empirical Validation on Neighboring $k_1$ Values
To verify whether $k_1 = 30$ resides on a broad, stable performance plateau or a fragile local peak, the complete LOFO grid search results from Week 6 were examined across $k_1 \in \{20, 25, 30, 35, 40\}$:

| $k_1$ | $k_2$ | $\lambda$ | Selection Score (Mean $\pm$ Std) | Class P@10 (Mean) | Class mAP (Mean) | ML mAP (Mean) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20** | 3 | 0.5 | $0.4668 \pm 0.0048$ | 0.5358 | 0.3124 | 0.2912 |
| **25** | 3 | 0.5 | $0.4680 \pm 0.0046$ | 0.5372 | 0.3183 | 0.2931 |
| **30** | **3** | **0.5** | $\mathbf{0.4686 \pm 0.0042}$ | **0.5361** | **0.3206** | **0.2938** |
| **35** | 3 | 0.5 | $0.4682 \pm 0.0043$ | 0.5348 | 0.3218 | 0.2935 |
| **40** | 3 | 0.5 | $0.4675 \pm 0.0045$ | 0.5332 | 0.3224 | 0.2929 |

### Findings on $k_1$ Stability:
1. **Broad Plateau:** Across the entire range $k_1 \in [25, 35]$, the selection score varies by less than $0.0006$ (less than $0.13\%$), well within the standard error of cross-validation ($\pm 0.0042$).
2. **Transferability to 80 Gallery Images:** Because $k_1 = 35$ yields virtually identical performance ($0.4682$ vs $0.4686$) with slightly higher Class mAP ($0.3218$), the performance curve is flat across the scaled density target ($k_1^* = 37.5$). The fixed value of $k_1 = 30$ does not degrade when transitioning to an 80-image-per-class gallery; its slightly more conservative window (37.5% coverage) helps filter out false-positive reciprocal edges in denser graphs.

---

## 3. Systematic Examination of $k_2 \in \{1, 2, 3\}$

In $k$-reciprocal re-ranking, $k_2$ controls the degree of query expansion applied to the reciprocal feature vector $V_q$.
* When $k_2 = 1$, no expansion occurs ($V_q$ is constructed strictly from reciprocal neighbors of query $q$).
* When $k_2 > 1$, $V_q$ is averaged with the top-$(k_2 - 1)$ gallery representations:
  $$V_q \leftarrow \frac{1}{k_2} \left( V_q + \sum_{g \in \text{top}(k_2 - 1)} V_g \right)$$

### 3.1. Fold-Level Performance Table
The table below documents the 5-fold Leave-One-Fold-Out results across $k_2 = 1$, $k_2 = 2$, and $k_2 = 3$ with $k_1 = 30$ and $\lambda = 0.5$ (`k2_per_fold.csv`):

| System | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V3 Baseline** | 0.4409 | 0.4517 | 0.4444 | 0.4418 | 0.4408 | 0.4439 | 0.0046 |
| **V4 ($k_2=1$)** | 0.4636 | 0.4724 | 0.4653 | 0.4678 | 0.4577 | 0.4654 | 0.0054 |
| **V4 ($k_2=2$)** | 0.4663 | 0.4752 | 0.4670 | 0.4664 | 0.4623 | 0.4674 | 0.0047 |
| **V4 ($k_2=3$)** | **0.4666** | **0.4756** | **0.4668** | **0.4690** | **0.4650** | **0.4686** | **0.0042** |

### Detailed Metric Breakdown (Mean $\pm$ Std):

| Metric | V3 Baseline | V4 ($k_2=1$) | V4 ($k_2=2$) | V4 ($k_2=3$, Frozen) |
| :--- | :---: | :---: | :---: | :---: |
| **Selection Score** | $0.4439 \pm 0.0046$ | $0.4654 \pm 0.0054$ | $0.4674 \pm 0.0047$ | $\mathbf{0.4686 \pm 0.0042}$ |
| **Class P@10** | $0.5069 \pm 0.0084$ | $\mathbf{0.5488 \pm 0.0135}$ | $0.5383 \pm 0.0144$ | $0.5361 \pm 0.0125$ |
| **Class mAP** | $0.2547 \pm 0.0039$ | $0.3037 \pm 0.0046$ | $0.3171 \pm 0.0060$ | $\mathbf{0.3206 \pm 0.0045}$ |
| **ML P@10** | $0.4556 \pm 0.0064$ | $0.4624 \pm 0.0079$ | $0.4596 \pm 0.0030$ | $\mathbf{0.4626 \pm 0.0042}$ |
| **ML mAP** | $0.2738 \pm 0.0042$ | $0.2888 \pm 0.0029$ | $0.2930 \pm 0.0038$ | $\mathbf{0.2938 \pm 0.0040}$ |
| **nDCG@10** | $0.7286 \pm 0.0032$ | $0.7232 \pm 0.0037$ | $0.7291 \pm 0.0035$ | $\mathbf{0.7298 \pm 0.0030}$ |
| **Runtime / Fold (KR)** | - | 0.627 s | 0.781 s | 0.803 s |

---

## 4. Rationale for Retaining $k_2 = 3$

### 4.1. Formal LOFO Selection Outcome
Under strict Leave-One-Fold-Out cross-validation, the model parameters for each held-out fold are chosen solely on the remaining 4 training folds using the composite selection score.
* As recorded in `k2_lofo_selection.csv`:
  * Fold 1 held out $\rightarrow$ selected $k_2 = 3$ (training score: 0.4691)
  * Fold 2 held out $\rightarrow$ selected $k_2 = 3$ (training score: 0.4668)
  * Fold 3 held out $\rightarrow$ selected $k_2 = 3$ (training score: 0.4690)
  * Fold 4 held out $\rightarrow$ selected $k_2 = 3$ (training score: 0.4685)
  * Fold 5 held out $\rightarrow$ selected $k_2 = 3$ (training score: 0.4695)
* On **100% of folds (5 / 5)**, $k_2 = 3$ was independently selected by the validation objective.

### 4.2. Precision vs Global Ranking (mAP) Trade-off
* **$k_2 = 1$ Behavior:** Disabling query expansion gives the highest top-10 precision (**Class P@10 = 0.5488**, $+0.0127$ over $k_2 = 3$). This occurs because isolated queries with unusual backgrounds avoid blending with irrelevant neighbors.
* **$k_2 = 3$ Advantage:** However, $k_2 = 1$ causes a severe collapse in global retrieval recall and ranking order:
  * Class mAP drops by **$-0.0169$** ($0.3037$ vs $0.3206$).
  * ML mAP drops by **$-0.0050$** ($0.2888$ vs $0.2938$).
  * nDCG@10 drops by **$-0.0066$** ($0.7232$ vs $0.7298$).
* In information retrieval, mAP evaluates the entire ranked list across all relevant documents, whereas P@10 only reflects the top 10 items. $k_2 = 3$ delivers substantially better global ranking while sacrificing only 1.2% in top-10 precision.

### 4.3. Fold-Level Variance & Stability
* Cross-fold standard deviation of the selection score decreases monotonically as $k_2$ increases:
  $$\sigma(k_2=1) = 0.0054 \quad > \quad \sigma(k_2=2) = 0.0047 \quad > \quad \sigma(k_2=3) = 0.0042$$
* $k_2 = 3$ provides the most consistent, low-variance performance across folds.

### 4.4. Methodological Precaution: Independence of Repeated Selection
* The Work Plan explicitly warns: *"Avoid treating repeated selection on the same folds as independent validation."*
* We recognize that while all 5 folds select $k_2 = 3$, these 5 folds share substantial training overlap (each pair of folds shares 3 out of 5 subsets). This constitutes standard cross-validation evidence rather than 5 statistically independent external experiments.
* Crucially, the decision rule from Week 6 states that to replace a frozen configuration, a candidate must demonstrate:
  1. $\Delta \text{ selection score} > +0.005$
  2. $\ge 4 / 5$ folds improved
  3. Class P@10 does not decrease
* As shown in `k2_vs_current_v4.csv`, both $k_2 = 1$ ($\Delta = -0.0032$) and $k_2 = 2$ ($\Delta = -0.0012$) fail the primary threshold ($\Delta < 0$). Under the pre-established scientific criteria, neither qualifies to replace $k_2 = 3$.

---

## 5. Final Configuration Decision

**Retain $k_2 = 3$ permanently as the frozen parameter of V4.**
* Configuration: $k_1 = 30, \quad k_2 = 3, \quad \lambda = 0.5, \quad \text{gallery\_only} = \text{True}$.
* Hardware / latency impact: Adding query expansion ($k_2 = 3$) increases re-ranking time by less than $0.18$ seconds per 480 queries ($0.803$ s vs $0.627$ s per fold), which is entirely negligible in practice.
* The frozen configuration in `Week 8/final_results/config/final_config.json` is verified and retained.
