# Week 6 - Label-Free Neighborhood Context on Frozen V3: Experimental Evaluation

## 1. Objective
The objective of Week 6 is to evaluate whether label-free neighborhood context can improve the frozen V3 visual retrieval baseline beyond cross-validation variability on the AID Multi-Label aerial image dataset. Two primary methodologies were investigated:
1. **Average Query Expansion (AQE)** in similarity space.
2. **k-reciprocal neighborhood re-ranking** adapted from Zhong et al. (2017) under a strict gallery-only, single-query deployment protocol.

## 2. Frozen V3 Baseline
The V3 baseline configuration remained strictly frozen throughout all experiments:
- **Descriptors**:
  - HSV (16x16 histogram, L1-normalized, Hellinger similarity)
  - LBP-r1 (radius 1, 8 neighbors, L1-normalized, Hellinger similarity)
  - LBP-r2 (radius 2, 16 neighbors, L1-normalized, Hellinger similarity)
  - HOG (128x128 input, 9 orientations, 16x16 cell, 2x2 block, L2-Hys norm, Cosine similarity)
- **Fusion Weights**: HSV = 0.20, LBP-r1 = 0.40, LBP-r2 = 0.20, HOG = 0.20.
- **Normalization**: Raw fusion (no Z-score normalization).
- **Reproduction Audit**: Frozen V3 reproduced known benchmark metrics exactly across all 5 folds:
  - Class P@10 = 0.5069 +/- 0.0084
  - Class mAP = 0.2547 +/- 0.0039
  - ML P@10 = 0.4556 +/- 0.0064
  - ML mAP = 0.2738 +/- 0.0042
  - nDCG@10 = 0.7286 +/- 0.0032
  - Selection Score = 0.4439 +/- 0.0046

## 3. Dataset and Split Protocol
The experiments utilized the 5-fold cross-validation partition established in `Week2/folds_5.json`:
- Total training images: 2,400 across 30 aerial scene categories (80 images per category).
- For each outer cross-validation fold: 480 held-out evaluation queries and 1,920 gallery images.
- Query self-matches and gallery self-neighbors were strictly excluded.
- The 600-image test set was not used during Week 6 experiments or parameter selection.

## 4. Label-Free Constraint and Gallery-Only Single-Query Protocol
No image labels were used for retrieval, neighborhood construction, query expansion, re-ranking, or parameter selection. Labels were used only after retrieval to compute evaluation metrics.

### Inductive Single-Query Guarantee:
- The reciprocal neighborhood graph is constructed strictly from the gallery (1,920 images).
- Each held-out query interacts with the gallery only.
- No query-query distance matrix (D_qq) is used.
- No held-out query participates in any other query's graph, reciprocal neighborhood, or re-ranking.

## 5. Method A - Average Query Expansion (AQE)
AQE was evaluated in similarity space over m in {1, 2, 3, 5, 8}, alpha in {0.25..4.0}, and three decay weightings.
- Selected parameters on outer folds: m in {5, 8}, alpha in {0.5, 1.0}, rank-decay.
- Selection Score Delta vs V3: +0.0023 (below the +0.005 threshold).
- Class P@10 Delta vs V3: -0.0042 (negative on 4 of 5 folds).
- Status: Retained only as an ablation showing no meaningful effect.

## 6. Method B - Gallery-Only k-reciprocal Re-Ranking
The gallery-only implementation constructs reciprocal neighborhoods from the gallery and evaluates queries one at a time.
The parameter grid was extended to:
- k1 in {10, 15, 20, 25, 30, 40}
- k2 in {3, 5, 8}
- lambda in {0.2, 0.3, 0.5, 0.7, 0.8, 0.9}

The LOFO selection chose the same parameter setting on all five outer folds:
**`k1 = 30, k2 = 3, lambda = 0.5`**

## 7. Overall Comparative Results
The table below compares the Frozen V3 control, AQE, the previous transductive ablation, and the new gallery-only k-reciprocal method across all six evaluation metrics:

| System | Class P@10 | Class mAP | ML P@10 | ML mAP | nDCG@10 | Selection Score | Delta Selection vs V3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen V3 Baseline** | 0.5069 +/- 0.0084 | 0.2547 +/- 0.0039 | 0.4556 +/- 0.0064 | 0.2738 +/- 0.0042 | 0.7286 +/- 0.0032 | **0.4439 +/- 0.0046** | +0.0000 |
| **AQE (Ablation)** | 0.5028 +/- 0.0082 | 0.2618 +/- 0.0036 | 0.4586 +/- 0.0025 | 0.2774 +/- 0.0037 | 0.7305 +/- 0.0033 | **0.4462 +/- 0.0027** | +0.0023 |
| **Transductive KR (Ablation)** | 0.5474 +/- 0.0126 | 0.3231 +/- 0.0041 | 0.4564 +/- 0.0095 | 0.2938 +/- 0.0043 | 0.7283 +/- 0.0043 | **0.4698 +/- 0.0060** | +0.0259 |
| **Gallery-Only KR (V4 Proposed)** | 0.5361 +/- 0.0125 | 0.3206 +/- 0.0045 | 0.4626 +/- 0.0042 | 0.2938 +/- 0.0040 | 0.7298 +/- 0.0030 | **0.4686 +/- 0.0042** | **+0.0247** |
| **Combined (Order 1: AQE->KR)** | 0.5298 +/- 0.0128 | 0.3178 +/- 0.0046 | 0.4598 +/- 0.0069 | 0.2937 +/- 0.0041 | 0.7286 +/- 0.0047 | **0.4659 +/- 0.0050** | +0.0220 |

### Interpretation of Retrieval Gains:
- **Full-Ranking & Class Metrics**: Gallery-only k-reciprocal produces substantial gains in Class mAP (+0.0659, a +25.9% relative gain) and Class P@10 (+0.0292), alongside a notable improvement in ML mAP (+0.0200).
- **Top-10 Multi-Label Metrics**: In contrast, gains in ML P@10 (+0.0070) and nDCG@10 (+0.0012) are more modest. k-reciprocal primarily refines category-level coherence and broad manifold recall across the full rank list, rather than strictly altering fine-grained multi-label rank order in the top 10.

## 8. Per-Fold Deltas Across All Six Metrics
Out-of-fold differences (Method - V3) for each of the five folds:

### Gallery-Only k-reciprocal Deltas
| Fold | Class P@10 Delta | Class mAP Delta | ML P@10 Delta | ML mAP Delta | nDCG@10 Delta | Selection Delta | Selected Parameters |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fold 1** | +0.0325 | +0.0677 | +0.0073 | +0.0209 | -0.0002 | **+0.0256** | {'k1': 30, 'k2': 3, 'lambda': 0.5} |
| **Fold 2** | +0.0333 | +0.0682 | +0.0000 | +0.0179 | -0.0001 | **+0.0239** | {'k1': 30, 'k2': 3, 'lambda': 0.5} |
| **Fold 3** | +0.0235 | +0.0614 | +0.0092 | +0.0184 | -0.0005 | **+0.0224** | {'k1': 30, 'k2': 3, 'lambda': 0.5} |
| **Fold 4** | +0.0338 | +0.0642 | +0.0144 | +0.0190 | +0.0046 | **+0.0272** | {'k1': 30, 'k2': 3, 'lambda': 0.5} |
| **Fold 5** | +0.0229 | +0.0680 | +0.0044 | +0.0237 | +0.0021 | **+0.0242** | {'k1': 30, 'k2': 3, 'lambda': 0.5} |
| **Mean** | **+0.0292** | **+0.0659** | **+0.0070** | **+0.0200** | **+0.0012** | **+0.0247** | — |

- Paired effect ratio for Selection Score delta: 13.53
- Paired effect ratio for Class mAP delta: 21.82

## 9. Per-Class Analysis
Retrieval performance across distinct aerial categories, highlighting historically weak/problematic classes:

| Class | V3 Class P@10 | KR Class P@10 | Delta Class P@10 | V3 Class mAP | KR Class mAP | Delta Class mAP | Category Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **playground** | 0.5837 | 0.6650 | +0.0812 | 0.3497 | 0.5128 | +0.1631 | Weak / Problematic |
| **airport** | 0.3038 | 0.3638 | +0.0600 | 0.1514 | 0.1945 | +0.0432 | Weak / Problematic |
| **river** | 0.5413 | 0.5250 | -0.0163 | 0.2196 | 0.2628 | +0.0433 | Weak / Problematic |
| **square** | 0.2125 | 0.2262 | +0.0137 | 0.0778 | 0.0889 | +0.0110 | Weak / Problematic |
| **bridge** | 0.1562 | 0.1737 | +0.0175 | 0.0673 | 0.0844 | +0.0171 | Weak / Problematic |

Key observations:
- Across the five focus classes: playground (P@10 +0.0812, mAP +0.1631); airport (P@10 +0.0600, mAP +0.0432); river (P@10 -0.0163, mAP +0.0433); square (P@10 +0.0137, mAP +0.0110); bridge (P@10 +0.0175, mAP +0.0171).
- Four of the five focus classes exhibit positive Class P@10 improvements, with playground (+0.0812) and airport (+0.0600) showing notable gains.
- River exhibits a slight decrease in Class P@10 (-0.0163) but gains +0.0433 in Class mAP, indicating improved broad recall across the gallery.

## 10. Query-Level Changes
Behavior across the complete pool of 2,400 queries under gallery-only k-reciprocal:
- Fraction of queries whose Top-10 ranking changed: 100.00% (2400/2,400)
- Number of query-level wins (higher Class P@10): 840 (35.00%)
- Number of query-level losses (lower Class P@10): 579 (24.12%)
- Number of unchanged queries: 981 (40.88%)
- Query win/loss ratio: 1.45x

The method produced more query-level wins than losses, with an overall win-to-loss ratio of 1.45x.

## 11. Parameter-Surface Analysis
The top 10 configurations in the extended grid display small differences near the upper region.
Selection score spans from 0.4686 down to 0.4677 (spread: 0.0009).
Parameter sensitivity appears limited in the neighborhood of k1 in [25, 30, 40], k2 in [3, 5], and lambda in [0.5, 0.7].

## 12. Decision
Applying the predefined adoption rule:
- **Criterion 1 (LOFO selection score improvement > 0.005)**: Delta = +0.0247 (Passes, exceeding threshold by 4.9x).
- **Criterion 2 (Positive improvement on at least 4 of 5 outer folds)**: 5 of 5 folds positive (+0.0256, +0.0239, +0.0224, +0.0272, +0.0242) (Passes).
- **Criterion 3 (Class P@10 does not decrease overall)**: Increases from 0.5069 to 0.5361 (+0.0292) (Passes).

Selected under the pre-set decision rule: **`V4 = V3 + k-reciprocal`**

## 13. Limitations
1. **Gallery Precomputation**: While individual query re-ranking executes in ~0.5 seconds, pairwise gallery relationships (O(N_g^2)) must be precomputed.
2. **Metric Divergence**: Improvements are concentrated in category-level precision and full-ranking mAP, with minimal impact on top-10 multi-label nDCG.

## 14. Mandatory Verification Statements
- **The 600-image test set was not used during Week 6 experiments or parameter selection.**
- **No image labels were used for retrieval, neighborhood construction, query expansion, re-ranking, or parameter selection. Labels were used only after retrieval to compute evaluation metrics.**
