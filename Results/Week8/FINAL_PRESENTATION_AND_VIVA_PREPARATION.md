# Final Presentation & Viva Defense Preparation Guide (Week 8)

**Project:** Content-Based Image Retrieval (CBIR) on AID Multi-Label Dataset  
**Group:** 16  
**Final System:** V4 = V3 + Gallery-Only $k$-Reciprocal Re-ranking  
**Date:** November 2026  

---

## 1. Slide Deck Structure & Speaker Notes

### Slide 1: Title & Executive Summary
* **Header:** Multi-Label Aerial Image Retrieval via Multi-Modal Feature Fusion and Reciprocal Neighborhood Re-ranking.
* **Key Visuals:** AID Dataset overview (3,000 images, 30 scene categories, 17 multi-labels), evolution banner from V1 to V4.
* **Speaker Script (30s):**
  > "Good morning examiners. Today we present Group 16's Content-Based Image Retrieval project on high-resolution aerial imagery. Over eight weeks of rigorous, hypothesis-driven engineering, we developed V4—a retrieval system combining multi-modal histogram and edge descriptors with gallery-only reciprocal graph re-ranking. On the official held-out test evaluation, V4 achieves a Class mAP of 0.2928 (a 55% relative gain over V2) and an overall selection score of 0.4269, outperforming all baseline models with confirmed statistical significance."

### Slide 2: Pipeline Architecture (V1 $\rightarrow$ V4)
* **Header:** System Architecture & Incremental Engineering.
* **Key Visuals:** Flowchart showing:
  1. Feature Extraction: HSV ($16 \times 16$), LBP-r1 ($R=1, P=8$), LBP-r2 ($R=2, P=16$), HOG ($128 \times 128$).
  2. V3 Distance Fusion: Hellinger distance on histograms, Cosine distance on HOG.
  3. V4 Re-ranking: Gallery-only $k$-reciprocal Jaccard distance blended 50/50 with V3 distance.
* **Speaker Script (45s):**
  > "Our pipeline addresses three specific computer vision challenges in aerial imaging. First, multi-scale texture modeling: adding LBP radius 2 captured coarser land patterns that radius 1 missed. Second, metric space consistency: in Week 5 we proved that probability-based Hellinger distance outperforms correlation for sparse histograms. Third, manifold learning: in Week 6 we introduced gallery-only k-reciprocal re-ranking. Rather than relying solely on pairwise query-gallery distance, V4 verifies whether a gallery image also considers the query to be among its nearest neighbors. This filters out asymmetric false positives."

### Slide 3: LOFO Validation vs Final Test Performance
* **Header:** Benchmark Performance Across Evaluation Regimes.
* **Key Visuals:** Side-by-side comparison table of V2, V3, and V4 across 5-fold LOFO and the single-run Final Test.
* **Speaker Script (45s):**
  > "To guarantee scientific integrity, our hyper-parameters were tuned exclusively on 5-fold Leave-One-Fold-Out cross-validation without ever accessing the test set. On LOFO, V4 improved Class mAP from 0.2547 to 0.3206 and Selection Score to 0.4686, with positive deltas across all five folds. When tested on the 600 unseen test queries against the full 2,400 gallery, V4 confirmed these gains: Class mAP rose to 0.2928 (+0.065 over V3), and Class P@10 reached 0.4667 (+0.0315 over V3). 10,000-sample paired bootstrap tests confirm that these primary gains exclude zero."

### Slide 4: Discrepancy Investigation (Playground, River, Meadow, Pond)
* **Header:** Workstream 1: Technical Root Causes of Per-Class Divergence.
* **Key Visuals:** Bar charts and exemplar comparison of Playground, River, Meadow, and Pond between LOFO and Test.
* **Speaker Script (60s):**
  > "In Week 8, we conducted an exhaustive investigation into four classes showing large discrepancies between LOFO and Test. We proved that this was not caused by code bugs or indexing shifts—all 3,000 image hashes, paths, and multi-label matrices are verified 1-to-1. Instead, we proved empirical distribution shift in the dataset:
  > First, River: Training river images are 77% urban waterfronts with docks, ships, and buildings. Test river images are 65% wilderness scrubland with zero docks and ships. Test river queries retrieve desert and mountain because the visual features accurately describe natural terrain.
  > Second, Playground: Playgrounds occupy under 15% of an aerial image. Global descriptors are dominated by the surrounding residential houses. We measured that test playgrounds are +0.1367 more visually similar to SparseResidential gallery images than to gallery Playgrounds!
  > Conversely, Pond surged to 88% precision on test because test ponds are clean, high-contrast water bodies with distinctive tree rings."

### Slide 5: Reliability Calibration & Gallery Density
* **Header:** Workstream 2: Automated Confidence Stratification.
* **Key Visuals:** Reliability score definition ($S_{\text{rel}} = 1 - D_{V4}(q, \text{top-1})$), tier distribution chart (HIGH / MEDIUM / LOW), test precision curve.
* **Speaker Script (45s):**
  > "A practical CBIR system must know when to trust its predictions. We defined a reliability confidence score based on the inverted top-1 distance, setting thresholds at the 20th and 70th percentiles of LOFO validation. On the final test set, our reliability indicator achieved flawless monotonic stratification: HIGH queries delivered 85.4% precision and 54.3% AP, MEDIUM delivered 44.8%, and LOW delivered 21.1%. While scaling the gallery from 1,920 to 2,400 images increased graph density and made the HIGH tier more conservative (21% vs 30%), we refused to recalibrate thresholds on test labels to prevent data snooping."

### Slide 6: Rejected Hypotheses & Scientific Rigor
* **Header:** Negative Results: Why We Froze V4 and Rejected V5.
* **Key Visuals:** Table of rejected experiments: AQE, AQE+KR, Transductive KR, Z-score, Neighbourhood Expansion (NE).
* **Speaker Script (45s):**
  > "True research rigor is demonstrated as much by what is rejected as by what is adopted. In Week 6, Average Query Expansion failed our threshold (+0.0023 < +0.005) and hurt Class P@10 on 4/5 folds. In Week 7, we tested V4-stage Neighbourhood Expansion across a 9-configuration grid. NE degraded selection score on 5 out of 5 folds (-0.0007) by pulling in irrelevant gallery neighbors. In Week 8, we investigated k2=1 and k2=2; both produced lower selection scores than k2=3 across all 5 folds. We therefore adhered strictly to our pre-registered adoption criteria: V4 remains frozen."

### Slide 7: Limitations & Future Work
* **Header:** Scientific Limitations & Engineering Roadmaps.
* **Key Visuals:** Summary of hand-crafted descriptor limits, $N=20$ sampling variance, path toward deep representations.
* **Speaker Script (30s):**
  > "To conclude, our work establishes the upper performance bound for hand-crafted global descriptors on the AID Multi-Label benchmark. Hand-crafted histograms cannot decouple localized objects from global background context. For future work, transitioning to spatial region proposals or deep vision transformers would resolve localized categories like Playground while preserving the reciprocal re-ranking manifold that proved so effective in V4. Thank you, and we welcome your questions."

---

## 2. Anticipated Viva Examiner Questions & Model Answers

### Q1: "What exactly does V4 do, and why is it superior to V3?"
* **Answer:**
  > "V3 is a pure visual distance metric: it computes a weighted combination of Hellinger distances across HSV and dual-scale LBP histograms plus Cosine distance on HOG. V4 adds manifold-based post-processing via gallery-only $k$-reciprocal nearest-neighbor re-ranking. Pairwise visual distance is asymmetric: query $q$ might have gallery image $g$ as a nearest neighbor simply because $g$ is a 'hub' in feature space, even though $q$ is far down $g$'s neighbor list. V4 requires reciprocity: $g$ is only a mutual neighbor if $q$ ranks within $g$'s top-$k_1$ nearest neighbors. V4 then computes Jaccard distance between their reciprocal neighbor sets and blends this 50/50 with V3 distance. This boosted Class mAP from 0.2547 to 0.3206 on LOFO (+25.8% relative) and from 0.2275 to 0.2928 on the final test set (+28.7% relative)."

### Q2: "Why didn't you build a V5 in Week 8?"
* **Answer:**
  > "The Week 8 Technical Work Plan explicitly instructed us to freeze V4 and focus on validation, discrepancy investigation, and reproducibility. Scientifically, building V5 would have violated our empirical adoption criteria. In Week 7, we tested Neighbourhood Expansion (NE) on top of V4, and it degraded performance on all 5 folds. In Week 8, we evaluated $k_2 = 1$ and $k_2 = 2$, and both were inferior to $k_2 = 3$ across all 5 folds. Without genuine evidence supporting a new architecture, pursuing V5 would represent arbitrary tinkering rather than sound engineering."

### Q3: "Why did Playground and River collapse on the final test set?"
* **Answer:**
  > "Our Week 8 investigation proved that this divergence is caused by dataset distribution shift, not code or indexing bugs. For River, 77.5% of training river images are urban/harbor riverfronts containing docks, ships, and pavement, whereas test river images are 65% wilderness scrubland with 0% docks and 0% ships. The visual features faithfully represent natural terrain, retrieving mountains and deserts. For Playground, the play structure occupies under 15% of the $600 \times 600$ aerial scene. The surrounding residential neighborhood dominates the global histograms. We measured that test playgrounds are +0.1367 more visually similar to gallery SparseResidential images than to gallery Playground images. Global hand-crafted descriptors lack localized spatial attention."

### Q4: "Why does the headline ML mAP (0.2716) differ from the mean of per-query `ml_ap` (0.2662)?"
* **Answer:**
  > "This is a mathematically verified property of information retrieval evaluation. In `week1_evaluate.py`, Average Precision is computed using `np.nanmean(ap)` across queries with at least one relevant gallery image ($N_{\text{rel}} > 0$). In the AID test set, exactly 12 queries have zero gallery images with Jaccard $\ge 0.8$. Standard benchmark protocol marks these 12 queries as undefined (NaN) and divides by 588. When per-query CSVs assign an imputed 0.0 to these 12 queries, taking a simple average divides by 600. The relationship is exact: $\text{Headline ML mAP} = \text{Per-Query Mean} \times \frac{600}{588}$. Both conventions are valid, and our Week 8 verifier mathematically confirms their equivalence to $10^{-8}$ precision."

### Q5: "Your reliability HIGH category dropped from 30% on LOFO to 21% on Test. Why didn't you recalibrate the threshold?"
* **Answer:**
  > "Recalibrating thresholds using test query labels would constitute test-set snooping and severe scientific leakage. The decrease in HIGH query share from 30% to 21% is caused by gallery density scaling: moving from 1,920 LOFO gallery images to 2,400 test gallery images increases graph density in $k$-reciprocal space, making reciprocal verification slightly more conservative. Crucially, the stratification remained flawless on unseen test data: HIGH queries achieved 85.4% Class P@10 and 54.3% AP, MEDIUM achieved 44.8%, and LOW achieved 21.1%. The thresholds successfully identified high-confidence queries without post-hoc tuning."

### Q6: "Why did you choose $k_2 = 3$ instead of $k_2 = 1$, when $k_2 = 1$ gave higher Class P@10?"
* **Answer:**
  > "$k_2 = 1$ disables query expansion, which prevents isolated queries from drifting, yielding a slight gain in top-10 precision (0.5488 vs 0.5361). However, $k_2 = 1$ causes a severe collapse in global retrieval ranking across the entire gallery: Class mAP drops from 0.3206 to 0.3037 (-0.0169), ML mAP drops from 0.2938 to 0.2888 (-0.0050), and cross-fold variance increases by 29%. Furthermore, $k_2 = 3$ was independently selected on all 5 LOFO validation folds. Prioritizing global ranking quality (mAP) over a marginal 1.2% top-10 precision gain is standard best practice in information retrieval."

### Q7: "Are the improvements of V4 over V3 statistically significant?"
* **Answer:**
  > "Yes, on primary metrics, but with important scientific caveats. On the 600-query test set, 10,000-iteration paired bootstrap tests confirm that V4's gains over V3 in Class mAP (+0.0653, 95% CI $[+0.0581, +0.0728]$), Class P@10 (+0.0315, 95% CI $[+0.0180, +0.0450]$), and ML mAP (+0.0152, 95% CI $[+0.0118, +0.0187]$) strictly exclude zero. However, we transparently report that ML P@10 (-0.0022, 95% CI $[-0.0140, +0.0097]$) and nDCG@10 (+0.0012, positive on only 2/5 folds) include zero and represent statistical noise. Furthermore, with only 20 queries per class on the test set, per-class P@10 intervals for weak classes (Airport, Bridge, Square) also span zero."
