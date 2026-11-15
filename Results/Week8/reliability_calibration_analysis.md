# Workstream 2: Reliability Calibration & Verification Analysis

**Project:** Multi-Label Content-Based Image Retrieval (CBIR) on AID Dataset  
**Phase:** Week 8 Technical Validation & Closure  
**Status:** Validated, Reconciled, and Frozen  

---

## 1. Executive Summary

Reliability estimation provides users with a confidence indicator for each retrieval query before inspecting the results. This workstream resolves historical formula discrepancies across weeks, verifies the exact mathematical definition of the reliability score for V4, analyzes the calibration of the decision thresholds on training data, evaluates the transfer of these thresholds to the final test set, examines the structural impact of gallery density on graph-based distance scores, and provides a justified decision regarding recalibration.

**Key Decisions & Findings:**
1. **Confirmed Score Formula:** For the frozen V4 system, the reliability score is uniquely defined as $S_{\text{rel}} = 1 - D_{V4}(q, \text{top-1})$, where $D_{V4} = 0.5 \cdot J_{\text{KR}} + 0.5 \cdot D_{V3}$.
2. **Resolution of Discrepancies:** Historical Week 4 thresholds ($\text{LOW} < 0.930, \text{HIGH} \ge 0.960$) applied strictly to V2's unnormalized similarity scale and are invalid for V4. The authoritative V4 thresholds are $\text{LOW} < 0.6419$ and $\text{HIGH} \ge 0.8078$, calibrated on the 20th and 70th percentiles of the 2,400 LOFO held-out queries.
3. **Gallery Density Effect:** Moving from the LOFO gallery ($N=1,920$) to the test gallery ($N=2,400$) increases graph density in $k$-reciprocal space, slightly shifting the distance distribution upward. This causes the test partition shares to shift from 30/50/20% to 20.8% HIGH, 51.3% MEDIUM, and 27.8% LOW.
4. **Preserved Monotonicity:** Despite this density shift, precision stratification remains strictly preserved on unseen test data: $\text{HIGH } (0.854) > \text{MEDIUM } (0.448) > \text{LOW } (0.211)$.
5. **Decision on Recalibration:** **No recalibration is permitted.** Tuning thresholds on test queries constitutes data leakage and violates evaluation integrity. The frozen thresholds are retained permanently.

---

## 2. Mathematical Definition & Discrepancy Resolution

### 2.1. Historical Evolution of the Score
Across project phases, different retrieval architectures operated on distinct numerical scales:
* **Week 4 (V2 Pipeline):** V2 computed weighted similarities using Pearson correlation and cosine similarity ($S_{V2} \in [-1, 1]$, typically $[0.88, 0.99]$ for aerial imagery). The reliability score was defined as raw top-1 similarity ($S_{\text{rel}} = S_{V2}(q, \text{top-1})$), with empirical thresholds of:
  $$\text{LOW} < 0.930, \quad \text{HIGH} \ge 0.960$$
* **Week 5–6 (V3 & V4 Pipelines):** V3 introduced Hellinger distance for HSV and LBP histograms. V4 added gallery-only $k$-reciprocal Jaccard distance ($J_{\text{KR}} \in [0, 1]$) blended with V3 distance:
  $$D_{V4}(q, g) = \lambda \cdot J_{\text{KR}}(q, g) + (1 - \lambda) \cdot D_{V3}(q, g), \quad \lambda = 0.5$$
  Because $D_{V4}$ is a distance metric ($0$ = identical, positive = dissimilar), distance must be transformed to similarity:
  $$S_{\text{rel}} = 1 - D_{V4}(q, \text{top-1})$$

### 2.2. Resolving Configuration Discrepancies
* In `Week4/reliability_config.json`, the thresholds are recorded as $0.930$ and $0.960$.
* In `Week 8/final_results/config/reliability_config_v4.json` and `Week 8/final_results/config/final_config.json`, the authoritative V4 thresholds are:
  $$\text{LOW} < 0.641873, \quad \text{HIGH} \ge 0.807825$$
* **Clarification:** The Week 4 thresholds belong exclusively to the obsolete V2 feature space. Applying V2 thresholds to V4 would classify nearly 100% of queries as LOW because $1 - D_{V4}$ rarely exceeds $0.90$. `reliability_config_v4.json` correctly supersedes `reliability_config.json`.

---

## 3. Threshold Calibration Methodology (LOFO Training Data)

To guarantee complete independence from the final test set, thresholds were established exclusively on training data using the 5-fold Leave-One-Fold-Out (LOFO) evaluation:
1. For each of the 5 folds, 480 held-out training queries were evaluated against the 1,920 gallery images using frozen V4 parameters ($k1=30, k2=3, \lambda=0.5$).
2. Across all 5 folds, top-1 reliability scores for all 2,400 training queries were pooled.
3. The thresholds were set to target fixed quantiles:
   * **LOW threshold:** 20th percentile of LOFO scores = **0.641873** (rounded to 0.6419).
   * **HIGH threshold:** 70th percentile of LOFO scores = **0.807825** (rounded to 0.8078).
   * **MEDIUM category:** $[0.6419, 0.8078)$, targeting the middle 50% of queries.

### Validation Statistics on LOFO (2,400 Held-Out Queries):
* **Spearman Rank Correlation:** The correlation between $S_{\text{rel}}$ and per-query Class P@10 is **$\rho = 0.5625$** ($p < 10^{-180}$). This is substantially stronger than the V2 correlation ($\rho = 0.3970$), confirming that V4's reciprocal distance provides a much more predictive confidence signal.

---

## 4. LOFO vs Test Reliability Distribution Comparison

When the frozen LOFO thresholds are applied directly to the 600 unseen test queries evaluated against the 2,400-image training gallery, the resulting distributions and retrieval performances compare as follows:

| Category | Dataset Split | Queries ($N$) | Query Share (%) | Class P@10 (Mean) | ML P@10 (Mean) | Class AP (Mean) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HIGH** | **LOFO (Train)** | 723 | 30.13% | 0.8136 | 0.6134 | 0.4900 |
| ($\ge 0.8078$) | **Final Test** | 125 | 20.83% | **0.8536** | **0.5496** | **0.5425** |
| | *Delta (Test - LOFO)* | *-598* | *-9.30%* | *+0.0400* | *-0.0638* | *+0.0525* |
| **MEDIUM** | **LOFO (Train)** | 1,199 | 49.96% | 0.4726 | 0.4257 | 0.2844 |
| $[0.6419, 0.8078)$ | **Final Test** | 308 | 51.33% | **0.4481** | **0.4104** | **0.2846** |
| | *Delta (Test - LOFO)* | *-891* | *+1.37%* | *-0.0245* | *-0.0153* | *+0.0002* |
| **LOW** | **LOFO (Train)** | 478 | 19.92% | 0.2759 | 0.3272 | 0.1552 |
| ($< 0.6419$) | **Final Test** | 167 | 27.83% | **0.2114** | **0.2701** | **0.1211** |
| | *Delta (Test - LOFO)* | *-311* | *+7.91%* | *-0.0645* | *-0.0571* | *-0.0341* |

### Key Analytical Findings:
1. **Monotonic Precision is Flawlessly Preserved:**
   * On the test set, HIGH reliability queries achieve **0.8536 Class P@10** and **0.5425 Class AP**.
   * MEDIUM reliability queries achieve **0.4481 Class P@10** and **0.2846 Class AP**.
   * LOW reliability queries achieve **0.2114 Class P@10** and **0.1211 Class AP**.
   * The stratification is clean, steep, and strictly monotonic ($\text{HIGH} \gg \text{MEDIUM} \gg \text{LOW}$). A user presented with a HIGH flag can trust that >85% of top-10 results belong to the target scene class.
2. **Category Proportions Shifted:**
   * The HIGH category share decreased from 30.1% to 20.8% (-9.3%).
   * The LOW category share increased from 19.9% to 27.8% (+7.9%).
   * The MEDIUM category remained stable at ~51% (50.0% LOFO vs 51.3% Test).

---

## 5. Investigation of the Gallery Density Effect

Why did the test distribution shift toward lower scores, reducing the HIGH share from 30% to 21%?

### Technical Mechanism:
1. **Gallery Scale Difference:**
   * In LOFO cross-validation, queries are searched against **1,920 gallery images** (64 images per class).
   * In final test evaluation, queries are searched against **2,400 gallery images** (80 images per class), an increase of $+25\%$ in gallery volume.
2. **Impact on $k$-Reciprocal Neighborhoods:**
   * In $k$-reciprocal re-ranking, candidate $g$ is a reciprocal neighbor of query $q$ only if $q$ ranks within the top-$k_1$ nearest neighbors of $g$ in the gallery.
   * As gallery size expands from 1,920 to 2,400 with $k_1$ fixed at 30, the competition for nearest-neighbor slots within the gallery intensifies. The gallery graph becomes denser, and each gallery node's top-$k_1$ set covers a smaller fraction of the dataset (37.5% of class exemplars vs 46.9%).
   * Consequently, Jaccard distance values shift slightly higher across the board, leading to a modest downward translation in $S_{\text{rel}} = 1 - D_{V4}$.
3. **Purity Enhancement in the HIGH Tier:**
   * Because the HIGH threshold ($0.8078$) became more selective under the denser gallery, only the most unambiguous query-gallery matches surpassed it.
   * As a direct result, test-set HIGH queries achieved **higher Class P@10 (0.8536 vs 0.8136)** and **higher Class AP (0.5425 vs 0.4900)** than observed during LOFO!
   * The increased conservatism improved tier precision at the cost of coverage.

---

## 6. Evaluation of Recalibration & Final Decision

### Arguments for Recalibration:
* Re-adjusting thresholds to the 20th and 70th percentiles of the test distribution would restore the 30% / 50% / 20% proportion split.

### Arguments against Recalibration:
1. **Validation Integrity:** Adjusting thresholds based on test set scores is a severe violation of machine learning protocol (test-set snooping / post-hoc calibration).
2. **Practical Utility:** In an operational deployment, test query distributions are unknown in advance. A model must rely on thresholds calibrated during offline validation.
3. **Monotonicity & Quality:** The existing thresholds maintain excellent separation ($0.854 \rightarrow 0.448 \rightarrow 0.211$). Modifying thresholds is unnecessary because the classification remains highly effective.

### Final Technical Decision:
**RETAIN THE FROZEN LOFO THRESHOLDS UNCHANGED.**
* Formula: $S_{\text{rel}} = 1 - D_{V4}(q, \text{top-1})$
* Low threshold: $0.6419$
* High threshold: $0.8078$
* The shift in tier proportions (30/50/20% $\rightarrow$ 21/51/28%) is documented transparently in all tables and notes as a known, natural consequence of gallery density scaling in graph-based re-ranking.
