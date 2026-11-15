# Final Technical Report: Validation, Discrepancy Investigation, and Project Closure (Week 8)

**Authors:** Group 16 — Content-Based Image Retrieval (CBIR) Project  
**Benchmark:** AID Multi-Label Dataset (3,000 images, 30 scene classes, 17 multi-labels)  
**Deliverable:** Week 8 Final Consolidated Package  
**Date:** November 2026  

---

## 1. Project Overview & Frozen V4 Retrieval System

Content-Based Image Retrieval (CBIR) on high-resolution aerial imagery poses unique challenges due to complex multi-object spatial compositions, varied scene scales, and semantic overlap across land-use categories. Over Weeks 1–7, this project systematically engineered, validated, and optimized an aerial image retrieval pipeline, progressing from single-descriptor baselines to a frozen multi-modal fusion architecture enhanced with contextual graph re-ranking.

### 1.1. Architectural Evolution (V1 through V4)
* **V1 (Baseline Fusion, Week 1–2):** Linear combination of Global HSV color histograms ($16 \times 16$ bins), uniform Local Binary Patterns (LBP, radius 1, 8 neighbors), and Histogram of Oriented Gradients (HOG, $128 \times 128$, $16 \times 16$ cells). Evaluated using Pearson correlation and Cosine similarities.
* **V2 (Multi-Scale LBP Fusion, Week 3–4):** Introduced multi-scale texture modeling by adding LBP with radius 2, 16 neighbors ($18$ bins). Frozen weights: $\text{HSV}(0.10) + \text{LBP-r1}(0.35) + \text{LBP-r2}(0.35) + \text{HOG}(0.20)$.
* **V3 (Metric-Tuned Distance Fusion, Week 5):** Replaced heuristic similarity metrics with probability-based Hellinger distance for histogram features (HSV, LBP-r1, LBP-r2) and Cosine distance for HOG. Eliminated unstable z-score normalizations in favor of raw L1/L2-Hys normalization. Re-optimized fusion weights on 5-fold Leave-One-Fold-Out (LOFO) validation: $\text{HSV}(0.20) + \text{LBP-r1}(0.40) + \text{LBP-r2}(0.20) + \text{HOG}(0.20)$.
* **V4 (Frozen Final System, Week 6–8):** Layered label-free, gallery-only $k$-reciprocal nearest-neighbor re-ranking on top of V3 distances. For each query $q$ and gallery candidate $g$:
  $$D_{V4}(q, g) = \lambda \cdot J_{\text{KR}}(q, g) + (1 - \lambda) \cdot D_{V3}(q, g)$$
  with frozen parameters:
  $$k_1 = 30, \quad k_2 = 3, \quad \lambda = 0.5, \quad \text{gallery\_only} = \text{True}$$
  Reliability estimation formula:
  $$S_{\text{rel}} = 1 - D_{V4}(q, \text{top-1}), \quad \text{LOW} < 0.6419, \quad \text{HIGH} \ge 0.8078$$

### 1.2. Week 8 Objectives & Protocol
As specified in the *Week 8 Technical Work Plan*, Week 8 does not pursue a "V5" or introduce new feature descriptors or heuristics. Instead, Week 8 rigorously establishes the trustworthiness, statistical soundness, and reproducibility of the frozen V4 system.

---

## 2. Workstream 1: LOFO vs Test Discrepancy Investigation

During the final held-out test evaluation (600 test queries against 2,400 training gallery images), four scene categories showed marked divergence from their 5-fold LOFO cross-validation performance:
* **Playground:** LOFO Class P@10 dropped from **0.6650** to **0.0350** on Test.
* **River:** LOFO Class P@10 dropped from **0.5250** to **0.0600** on Test.
* **Meadow:** LOFO Class P@10 dropped from **0.7388** to **0.3050** on Test.
* **Pond:** LOFO Class P@10 surged from **0.3425** to **0.8800** on Test.

### 2.1. Audit of the Software Pipeline
To confirm that these discrepancies were not caused by indexing bugs or data leakage, five rigorous checks were performed:
1. **Path & Cache Index Alignment:** Evaluated whether alphabetical sorting in `list_images()` induced index drift. All 600 test query paths and all 2,400 gallery paths match `multilabel_.csv`, feature caches, and result CSVs with 100% alignment.
2. **File Integrity & Hashing:** MD5 hashes computed across all 3,000 AID images verified zero corrupted files and zero duplicates. All images are identically sized at $600 \times 600 \times 3$.
3. **Gallery Construction Consistency:** The proportion of true-class images in the gallery is identical across both evaluation regimes ($64 / 1,920 = 3.333\%$ in LOFO; $80 / 2,400 = 3.333\%$ in Final Test), ruling out class prior shifts.

### 2.2. Empirical Demonstration of Distribution Shift
1. **River (Semantic Shift):** Ground-truth multi-label inspection revealed that training river images (1–80) are predominantly **urban/harbor riverfronts** (77.5% buildings, 77.5% pavement, 71.2% cars, 48.8% docks, 45.0% ships). Test river images (81–100) are **rural/wilderness waterways** (0% docks, 0% ships, 5% pavement, 65% chaparral). Consequently, test river queries retrieve mountain, desert, church, and farmland scenes.
2. **Playground (Scale Mismatch & Context Dominance):** Playgrounds occupy $<15\%$ of the image surface, with the remainder dominated by residential homes and roads. Empirical visual distance measurements confirmed that test playgrounds are **+0.1367 more similar to SparseResidential gallery images than to gallery Playground images** ($0.6813$ vs $0.5446$). Global features represent the residential matrix, retrieving SparseResidential (44) and MediumResidential (35).
3. **Pond (Positive Shift):** Test ponds are clean, high-contrast water bodies with well-defined tree borders, matching training exemplars with higher intra-class similarity ($0.6172$) than training ponds exhibit among themselves ($0.5929$), enabling precision to surge to 88%.

---

## 3. Workstream 2: Reliability Calibration & Verification

### 3.1. Reconciling Score Definitions
Historical Week 4 thresholds ($\text{LOW} < 0.930, \text{HIGH} \ge 0.960$) applied to V2's unnormalized similarity scale. V4 operates in distance space; its score is $S_{\text{rel}} = 1 - D_{V4}(q, \text{top-1})$. The authoritative thresholds ($\text{LOW} < 0.6419, \text{HIGH} \ge 0.8078$) were established on the 20th and 70th percentiles of all 2,400 LOFO held-out queries. The Spearman correlation with Class P@10 is $\rho = 0.5625$ ($p < 10^{-180}$).

### 3.2. Gallery Density Effect on Threshold Transfer
When evaluated on the 2,400-image test gallery (vs 1,920 in LOFO), increased graph density shifts distance scores slightly lower. As a result, the share of HIGH queries decreased from 30.1% to 20.8%, while LOW queries increased from 19.9% to 27.8%.
* Crucially, **monotonic precision is strictly preserved**:
  * Final Test HIGH: **0.8536 Class P@10**, **0.5425 Class AP**
  * Final Test MEDIUM: **0.4481 Class P@10**, **0.2846 Class AP**
  * Final Test LOW: **0.2114 Class P@10**, **0.1211 Class AP**
* **Decision:** Retain frozen thresholds without post-hoc test re-tuning.

---

## 4. Workstream 3: Parameter Sensitivity ($k_1$ and $k_2$ Robustness)

### 4.1. $k_1 = 30$ Density Transfer
On LOFO, $k_1 = 30$ spans 46.9% of a class (30/64). On the final test gallery (80 images/class), it spans 37.5%. Cross-validation grid results over $k_1 \in [20, 40]$ show a flat performance plateau: $k_1=25$ (0.4680), $k_1=30$ (0.4686), and $k_1=35$ (0.4682) differ by $<0.0006$. Fixing $k_1=30$ is robust.

### 4.2. $k_2 \in \{1, 2, 3\}$ Evaluation
* All 5 LOFO validation folds independently selected $k_2 = 3$ (`k2_lofo_selection.csv`).
* $k_2 = 3$ exhibits the lowest cross-fold standard deviation ($\sigma = 0.0042$ vs $0.0047$ for $k_2=2$ and $0.0054$ for $k_2=1$).
* While $k_2 = 1$ offers slightly higher top-10 precision (+0.0127 Class P@10), $k_2 = 3$ provides superior global ranking quality: Class mAP is **+0.0169 higher** ($0.3206$ vs $0.3037$) and ML mAP is **+0.0050 higher** ($0.2938$ vs $0.2888$).
* **Decision:** Retain $k_2 = 3$ permanently.

---

## 5. Workstream 4: Statistical Validation & Metric Reconciliation

### 5.1. Mathematical Reconciliation of ML mAP
In `week1_evaluate.py`, `average_precision` computes the mean over queries that have at least one relevant gallery item ($N_{\text{rel}} > 0$). In the test set, exactly 12 queries have $N_{\text{rel}} = 0$ (Jaccard $\ge 0.8$). The headline ML mAP evaluates over 588 queries, while per-query CSVs divide by 600:
$$\text{Headline ML mAP} = \text{Per-Query Mean} \times \frac{600}{588}$$
This mathematical identity holds to $10^{-8}$ precision across all models.

### 5.2. Verified Paired Bootstrap Confidence Intervals (10,000 Iterations)
* **V4 vs V3 Class P@10:** $+0.0315$, 95% CI $[+0.0180, +0.0450]$ (Significant gain)
* **V4 vs V3 Class mAP:** $+0.0653$, 95% CI $[+0.0581, +0.0728]$ (Large significant gain)
* **V4 vs V3 ML mAP:** $+0.0152$, 95% CI $[+0.0118, +0.0187]$ (Significant gain)
* **V4 vs V3 ML P@10:** $-0.0022$, 95% CI $[-0.0140, +0.0097]$ (**CI includes zero / noise**)
* **Weak Class P@10 on Test:** Airport ($[-0.0100, +0.1200]$), Bridge ($[-0.0700, +0.0250]$), and Square ($[-0.0250, +0.1550]$) all span zero due to $N=20$ sample limitations.

---

## 6. Summary of Rejected Experiments & Scientific Criteria

In accordance with strict empirical decision rules established in Week 6, candidate methods were rejected if they failed to achieve $\Delta \text{ score} > +0.005$, $\ge 4/5$ folds positive, and no drop in Class P@10:

| Candidate Experiment | Week | Reference | $\Delta$ Selection Score | Outcome | Technical Reason for Rejection |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Average Query Expansion (AQE) on V3** | 6 | V3 | $+0.0023$ | Rejected (Ablation) | Gain $+0.0023 < +0.005$ threshold; Class P@10 fell by $-0.0042$ (negative on 4/5 folds). |
| **Combined AQE $\rightarrow$ KR (Order 1)** | 6 | V4 | $-0.0026$ | Rejected | Inferior to gallery-only $k$-reciprocal alone. |
| **Transductive $k$-Reciprocal** | 6 | V4 | $+0.0012$ | Replaced by Gallery-Only | Gallery-only protocol avoids cross-query interactions and test leakage. |
| **Per-Query Z-Score Normalization** | 5 | V2 Raw | $-0.0157$ | Rejected | Degraded raw distance fusion; variance distortion across sparse histograms. |
| **Neighbourhood Expansion (NE) on V4** | 7 | V4 | $-0.0007$ | Rejected (Ablation) | Negative delta on 5/5 folds; all 9 grid configs failed to improve V4. |
| **V4 with $k_2 = 1$** | 7 | V4 ($k_2=3$) | $-0.0032$ | Not Adopted | Class mAP dropped by $-0.0169$; rejected on all 5 LOFO folds. |
| **V4 with $k_2 = 2$** | 7 | V4 ($k_2=3$) | $-0.0012$ | Not Adopted | Inferior selection score across all 5 folds. |

---

## 7. Limitations & Future Directions

1. **Global Descriptor Limitations:** Histograms and global HOG cannot capture localized spatial arrangements, causing false matches when background context dominates (e.g. Playground confused with Residential).
2. **Small Test Sample ($N=20$):** Class-level test metrics exhibit wide confidence intervals ($\pm 0.08$), making per-class test comparisons sensitive to outlier queries.
3. **Future Work:** Transitioning from hand-crafted global descriptors to deep spatial feature extractors (e.g., Vision Transformers or Mask-RCNN region features) to decouple foreground targets from background context.

---

## 8. Final Consolidated Evaluation Table

### LOFO Cross-Validation (Mean $\pm$ Std across 5 Folds, 2,400 Queries)
| System | Class P@10 | Class mAP | ML P@10 | ML mAP | nDCG@10 | Selection Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V2** | $0.4336 \pm 0.0034$ | $0.2123 \pm 0.0014$ | $0.4080 \pm 0.0081$ | $0.2411 \pm 0.0035$ | $0.6901 \pm 0.0057$ | $0.3970 \pm 0.0033$ |
| **V3** | $0.5069 \pm 0.0084$ | $0.2547 \pm 0.0039$ | $0.4556 \pm 0.0064$ | $0.2738 \pm 0.0042$ | $0.7286 \pm 0.0032$ | $0.4439 \pm 0.0046$ |
| **V4** | $\mathbf{0.5361 \pm 0.0125}$ | $\mathbf{0.3206 \pm 0.0045}$ | $\mathbf{0.4626 \pm 0.0042}$ | $\mathbf{0.2938 \pm 0.0040}$ | $\mathbf{0.7298 \pm 0.0030}$ | $\mathbf{0.4686 \pm 0.0042}$ |

### Final Held-Out Test Evaluation (600 Queries vs 2,400 Gallery Images, Run Once)
| Model | Class P@10 | Class mAP | ML P@10 | ML mAP | nDCG@10 | Selection Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V2** | 0.3867 | 0.1889 | 0.3650 | 0.2278 | 0.6740 | 0.3685 |
| **V3** | 0.4352 | 0.2275 | 0.4025 | 0.2561 | 0.7060 | 0.4054 |
| **V4** | **0.4667** | **0.2928** | **0.4003** | **0.2716** | **0.7030** | **0.4269** |
| *V4 - V3* | *+0.0315* | *+0.0653* | *-0.0022* | *+0.0152* | *-0.0030* | *+0.0214* |
| *V4 - V2* | *+0.0800* | *+0.1039* | *+0.0353* | *+0.0430* | *+0.0290* | *+0.0584* |
