# Workstream 1: LOFO versus Test Discrepancy Investigation Report

**Project:** Multi-Label Content-Based Image Retrieval (CBIR) on AID Dataset  
**Phase:** Week 8 Technical Validation & Closure  
**Status:** Complete & Empirically Verified  

---

## 1. Executive Summary

During the final held-out test evaluation (600 test queries against 2,400 training gallery images), four scene classes exhibited substantial performance discrepancies when compared to their 5-fold Leave-One-Fold-Out (LOFO) cross-validation benchmarks on the training set:
* **Playground:** Class P@10 dropped from **0.6650** (LOFO) to **0.0350** (Test). Class AP dropped from **0.5128** to **0.0490**.
* **River:** Class P@10 dropped from **0.5250** (LOFO) to **0.0600** (Test). Class AP dropped from **0.2628** to **0.0531**.
* **Meadow:** Class P@10 dropped from **0.7388** (LOFO) to **0.3050** (Test). Class AP dropped from **0.3669** to **0.1634**.
* **Pond:** Class P@10 surged from **0.3425** (LOFO) to **0.8800** (Test). Class AP surged from **0.1178** to **0.2991**.

A comprehensive technical investigation was conducted across five dimensions:
1. File paths, naming conventions, and cache indexing.
2. Label matrix alignment and image ID verification.
3. Image integrity, file hashing (MD5), and duplicate detection.
4. Gallery construction consistency and class prior stability.
5. Visual and multi-label feature distribution shift.

**Primary Finding:** The discrepancies are **not** caused by software bugs, cache misalignment, indexing errors, or gallery imbalance. They are caused by **genuine distribution shifts** in the AID dataset between the training partition (`images_tr`, images 1–80) and the test partition (`images_test`, images 81–100), combined with the fundamental limitation of global image descriptors (HSV, LBP, HOG) when describing localized semantic concepts within complex aerial scenes.

---

## 2. Performance Comparison: LOFO vs Final Test

The table below summarizes the retrieval performance across V2 (multi-scale LBP fusion), V3 (tuned feature weights), and V4 (gallery-only $k$-reciprocal re-ranking) for the four focal classes:

| Class | Stage | Queries | V2 Class P@10 | V3 Class P@10 | V4 Class P@10 | V4 Class AP | Zero P@10 Queries (V4) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Playground** | LOFO | 80 | 0.4962 | 0.5837 | **0.6650** | **0.5128** | 2 / 80 (2.5%) |
| | Test | 20 | 0.0250 | 0.0200 | **0.0350** | **0.0490** | 18 / 20 (90.0%) |
| | *Delta* | *-* | *-0.4712* | *-0.5637* | **-0.6300** | **-0.4638** | *+87.5%* |
| **River** | LOFO | 80 | 0.4800 | 0.5413 | **0.5250** | **0.2628** | 5 / 80 (6.2%) |
| | Test | 20 | 0.0800 | 0.0650 | **0.0600** | **0.0531** | 13 / 20 (65.0%) |
| | *Delta* | *-* | *-0.4000* | *-0.4763* | **-0.4650** | **-0.2097** | *+58.8%* |
| **Meadow** | LOFO | 80 | 0.5613 | 0.6738 | **0.7388** | **0.3669** | 4 / 80 (5.0%) |
| | Test | 20 | 0.2550 | 0.4000 | **0.3050** | **0.1634** | 1 / 20 (5.0%) |
| | *Delta* | *-* | *-0.3063* | *-0.2738* | **-0.4338** | **-0.2035** | *0.0%* |
| **Pond** | LOFO | 80 | 0.2800 | 0.2987 | **0.3425** | **0.1178** | 23 / 80 (28.8%) |
| | Test | 20 | 0.7000 | 0.6850 | **0.8800** | **0.2991** | 1 / 20 (5.0%) |
| | *Delta* | *-* | *+0.4200* | *+0.3863* | **+0.5375** | **+0.1813** | *-23.8%* |

### Observations:
1. The discrepancy pattern is present across **all three retrieval architectures** (V2, V3, and V4). In fact, V2 already showed test P@10 values of 0.025 for Playground, 0.080 for River, 0.255 for Meadow, and 0.700 for Pond in Week 4.
2. The degradation is not an artifact of $k$-reciprocal re-ranking. While V4 amplified the separation (improving LOFO Playground from 0.58 to 0.66, and test Pond from 0.68 to 0.88), the underlying divergence between splits was already present in V2 and V3.
3. For Playground, 18 out of 20 test queries returned zero correct scene items in the top-10.
4. For River, 13 out of 20 test queries returned zero correct scene items in the top-10.
5. For Pond, the discrepancy is strongly positive: 18 out of 20 test queries achieved P@10 $\ge 0.70$, with 12 queries reaching a perfect 1.00.

---

## 3. Systematic Verification of the Data & Evaluation Pipeline

To verify whether implementation bugs contributed to these differences, five strict verification checks were conducted:

### 3.1. Path & Index Alignment
* **Verification:** The image list generation function `list_images()` uses `sorted(Path(folder).rglob("*.jpg"))`. On both train (`images_tr`) and test (`images_test`) archives, this produces deterministic alphabetical ordering.
* **Result:** For test queries, the order is `airport_100.jpg`, `airport_81.jpg`, ..., `viaduct_99.jpg`. This matches the index of `per_query_v2_v3_v4.csv`, `multilabel_.csv`, and the precomputed feature caches exactly ($600 / 600$ matches).
* **Result:** For training gallery images, all 2,400 paths match `multilabel_.csv` and feature caches with zero index shifts ($2,400 / 2,400$ matches).

### 3.2. Image Integrity & Duplicate Analysis
* **Verification:** MD5 hashes were computed for all 3,000 images in the AID dataset across all 30 classes.
* **Result:** All 3,000 images have unique MD5 hashes. There are zero duplicate images between train and test, and zero duplicate images within classes.
* **Resolution Check:** All 3,000 images were confirmed to have exact dimensions of $600 \times 600$ pixels with 3 RGB channels (no corrupt files, truncated streams, or dimension mismatches).

### 3.3. Gallery Construction Consistency
* **LOFO Gallery:** Each LOFO fold uses 4 folds as gallery ($4 \times 480 = 1,920$ images). Because the 5 folds are strictly balanced across all 30 classes (16 queries and 64 gallery images per class), every LOFO gallery contains exactly **64 images per class** ($64 / 1,920 = 1/30 = 3.333\%$ prior).
* **Final Test Gallery:** The test evaluation uses the entire training set as gallery ($2,400$ images). This contains exactly **80 images per class** ($80 / 2,400 = 1/30 = 3.333\%$ prior).
* **Finding:** Gallery class priors are mathematically identical across both evaluation regimes ($3.333\%$). Gallery class imbalance is eliminated as a cause.

---

## 4. Root Cause Analysis: Evidence for Distribution Shift

Empirical evidence confirms two distinct mechanisms driving the observed discrepancies: **Semantic Content Shift** and **Context Dominance in Global Descriptors**.

### 4.1. River: Severe Multi-Label Semantic Shift
Analysis of the multi-label ground truth (`multilabel_.csv`) reveals a stark difference in what constitutes a "River" image between the train and test splits:

| Semantic Label | Train River (images 1–80) | Test River (images 81–100) | Semantic Delta |
| :--- | :---: | :---: | :---: |
| **water** | 100.0% | 100.0% | 0.0% |
| **bare-soil** | 98.8% | 100.0% | +1.2% |
| **grass** | 97.5% | 100.0% | +2.5% |
| **trees** | 77.5% | 80.0% | +2.5% |
| **buildings** | **77.5%** | **25.0%** | **-52.5%** |
| **pavement** | **77.5%** | **5.0%** | **-72.5%** |
| **cars** | **71.2%** | **20.0%** | **-51.2%** |
| **field** | **57.5%** | **30.0%** | **-27.5%** |
| **dock** | **48.8%** | **0.0%** | **-48.8%** |
| **ship** | **45.0%** | **0.0%** | **-45.0%** |
| **chaparral** | **2.5%** | **65.0%** | **+62.5%** |

* **Analysis:** The training set rivers are predominantly **urban and commercial riverfronts**, characterized by docks (48.8%), ships (45.0%), extensive pavement (77.5%), buildings (77.5%), and vehicles (71.2%).
* In contrast, the test set rivers are **rural, mountainous, and desert wilderness waterways**, completely devoid of docks (0.0%) and ships (0.0%), virtually free of pavement (5.0%), and heavily dominated by scrubland/chaparral (65.0% vs 2.5% in train).
* **Retrieval Consequence:** When a test query of a wilderness river is submitted against the urban-heavy training gallery, global color and texture descriptors correctly identify natural terrain features, retrieving:
  * Mountain: 28 retrievals
  * Church: 18 retrievals
  * Desert: 18 retrievals
  * Farmland: 17 retrievals
  * River: only 16 retrievals (out of 200 total top-10 slots).
  The visual features faithfully represent the scene content; the failure is caused by an unrepresentative split in the dataset curation.

### 4.2. Playground: Context Dominance & Feature Misalignment
Playgrounds represent a classic failure mode for global aerial image descriptors:
* An aerial playground consists of localized equipment (swings, slides, athletic markings) covering a small spatial fraction of a $600 \times 600$ image ($< 15\%$ surface area).
* The remainder of the image is dominated by the surrounding neighborhood matrix: residential houses, driveways, street pavement, lawns, and tree lines.

Quantitative visual similarity measurements using the frozen V3 feature representations ($0.20 \text{ HSV} + 0.40 \text{ LBP-r1} + 0.20 \text{ LBP-r2} + 0.20 \text{ HOG}$) reveal:
* **Intra-Train Same-Class Similarity (Playground $\leftrightarrow$ Playground):** $0.6542 \pm 0.0644$
* **Test-to-Train Same-Class Similarity (Test Playground $\leftrightarrow$ Train Playground):** $0.5446 \pm 0.0784$ (**Drop of -0.1095**)
* **Test Playground $\leftrightarrow$ Train SparseResidential Similarity:** **0.6813**

**Critical Finding:** Test playground queries are **+0.1367 more visually similar to SparseResidential gallery images than to gallery Playground images**.
Furthermore, test playgrounds have 50.0% bare-soil vs only 12.5% in training playgrounds (+37.5%). Consequently, the top retrieved classes for test playgrounds are:
* SparseResidential: 44 retrievals
* MediumResidential: 35 retrievals
* DenseResidential: 19 retrievals
* Park: 12 retrievals
* Meadow: 11 retrievals
* Playground: 7 retrievals (Class P@10 = 0.0350).

### 4.3. Meadow: Vegetation Overlap
Meadow images consist purely of grass (100%) and perimeter paths/pavement (100%).
* Visual similarity: Test-to-train similarity ($0.6956$) is slightly lower than intra-train ($0.7067$).
* However, because the global feature representation lacks spatial layout, uniform green pastures are indistinguishable from large residential lawns or golf courses in SparseResidential.
* Top retrievals for test meadow queries: SparseResidential (54), Meadow (51), Forest (24), Mountain (18).

### 4.4. Pond: Positive Distribution Shift
Unlike River and Playground, Pond experienced a dramatic **increase** in performance on the test set (+0.5375 on P@10, surging from 0.3425 to 0.8800).
* **Multi-label profile:** Test ponds exhibit higher vegetation density (trees 100% vs 88.8%) and higher pavement presence (90.0% vs 45.0%), reflecting standardized artificial park ponds with paved perimeters.
* **Visual similarity:** Test pond queries have a **higher** similarity to training ponds ($0.6172$) than training ponds have to one another ($0.5929$, a gain of $+0.0243$).
* In LOFO, pond suffered from confusion with port (0.090) and river (0.069). In the test set, the 20 test ponds are visually clean, high-contrast water bodies surrounded by distinctive tree rings, allowing V4 re-ranking to achieve $88\%$ precision.

---

## 5. Summary of Established Causes and Remaining Uncertainties

### Established Causes:
1. **No Implementation or Indexing Errors:** Image IDs, paths, feature caches, and multi-label ground truth matrices are 100% verified and aligned.
2. **Dataset Partitioning Bias:** AID splits 1–80 and 81–100 were not randomly stratified across environmental subclasses. River contains a severe geographic/land-use shift (urban waterfronts vs arid scrubland waterways).
3. **Descriptor Scale Mismatch:** Global histograms lack spatial localization. For classes defined by localized human-scale features (Playground), global descriptors capture background context (residential housing and roads), leading to systematic confusion with residential categories.
4. **Graph Re-ranking Amplification:** $k$-reciprocal re-ranking reinforces strong initial clusters. When initial visual similarity points toward SparseResidential (Playground) or Mountain/Desert (River), reciprocal verification solidifies the incorrect neighborhood.

### Remaining Uncertainties:
1. **Test Split Sample Size ($N=20$ per class):** With only 20 queries per class, individual queries exert a large influence on class-level metrics ($5\%$ per query). The 95% bootstrap confidence intervals for 20-query classes are wide (e.g., $\pm 0.06$ to $\pm 0.12$), meaning test metric estimates contain higher sampling variance than the 80-query LOFO estimates.

---

## 6. Recommendations & Technical Conclusion

1. **Retain Frozen V4 System:** Do not alter feature fusion weights or add ad-hoc post-processing to artificially "fix" River or Playground on the test set. Such adjustments would constitute test-set overfitting and violate validation integrity.
2. **Transparent Reporting:** The final report and viva presentation must explicitly present both LOFO and test numbers, using the multi-label and visual similarity evidence documented here to explain why global descriptors fail on these specific classes.
3. **Future Work:** State clearly that resolving classes like Playground and River requires either localized object detection (e.g. YOLO bounding boxes for playground equipment) or spatially-aware deep representations (e.g., ResNet/ViT feature maps) rather than global hand-crafted descriptors.
