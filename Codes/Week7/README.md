# Week 7 - Validate and freeze V4, then final test

Flow: **V4 -> validate -> freeze -> final test -> analyse -> document.** No tuning after the final test.

**Status.** V4 is validated and frozen on train/LOFO evidence (steps 1-3, LOFO part of 5, and 6 done).
The **final test (V2 vs V3 vs V4) has NOT been run** because the test-split feature caches were not available; the script is ready (step 4). Nothing below makes a claim about test-set performance of V3 or V4.

## 1. Frozen system (V4)
V3 = HSV(0.20) + LBP-r1(0.40) + LBP-r2(0.20) + HOG(0.20), Hellinger for the three histograms, cosine for HOG.
V4 = V3 + gallery-only k-reciprocal re-ranking, **k1=30, k2=3, lambda=0.5**: `D_V4 = 0.5 * Jaccard_KR + 0.5 * D_V3`. See `selected_parameters.json`.

## 2. Neighbourhood expansion: demoted to an ablation
Rejected (fails all three Week 6 criteria; selection score -0.0007, 0/5 folds). Everything moved to `ne_ablation/`, with its own README. It is not used anywhere in the final system.

## 3. k2 check (the one new model experiment)
Only k2 changed (k1=30, lambda=0.5 fixed), 5-fold LOFO on train data, Week 6 adoption rule applied unchanged. Week 6 had only tried k2 in {3, 5, 8}, so k2 = 1 and 2 were new. (k2 = 1 means no query expansion inside the re-ranking.)

| System | Class P@10 | ML P@10 | nDCG@10 | Class mAP | ML mAP | Selection score (mean +/- std) |
|---|---:|---:|---:|---:|---:|---:|
| V3 | 0.5069 | 0.4556 | 0.7286 | 0.2547 | 0.2738 | 0.4439 +/- 0.0046 |
| V4, k2=1 | 0.5488 | 0.4624 | 0.7232 | 0.3037 | 0.2888 | 0.4654 +/- 0.0054 |
| V4, k2=2 | 0.5383 | 0.4596 | 0.7291 | 0.3171 | 0.2930 | 0.4674 +/- 0.0047 |
| **V4, k2=3** | 0.5361 | 0.4626 | 0.7298 | 0.3206 | 0.2938 | **0.4686 +/- 0.0042** |

Selection score per fold: k2=3 is 0.4666 / 0.4756 / 0.4668 / 0.4690 / 0.4650. Versus k2=3, k2=1 is -0.0032 (0/5 folds up) and k2=2 is -0.0012 (1/5 up). Leave-one-fold-out chose k2=3 on all five folds.
**Decision: V4 unchanged, k2=3.** Note: k2=1 has the highest Class P@10 (+0.0127 over k2=3) but is worse on Class mAP, nDCG@10 and the selection score, so it fails the rule. Files: `k2_check/`.

## 4. V4 versus V3 on LOFO data (where V4 actually improves)
Paired fold differences (V4 - V3), 5 folds:

| Metric | Mean change | Relative | Folds improved | Fold SD of the change |
|---|---:|---:|---:|---:|
| Class mAP | +0.0659 | +25.9% | 5/5 | 0.0030 |
| Class P@10 | +0.0292 | +5.8% | 5/5 | 0.0055 |
| ML mAP | +0.0200 | +7.3% | 5/5 | 0.0024 |
| Selection score | +0.0247 | +5.6% | 5/5 | 0.0018 |
| ML P@10 | +0.0070 | +1.5% | 4/5 | 0.0054 |
| nDCG@10 | +0.0012 | +0.2% | 2/5 | 0.0022 |

Interpretation: V4 clearly improves the ranking-wide metrics (mAP) and class precision. The gain in ML P@10 is small, and **nDCG@10 is unchanged within noise**, so no improvement should be claimed for it.

## 5. Weak-class analysis, V3 vs V4 (2,400 held-out LOFO queries, 80 per class)
Query level: Class P@10 improved on 840 queries, worsened on 579, unchanged on 981. Class level: P@10 improved on 23 classes and worsened on 7; AP improved on 27 and worsened on 3.

| Class | P@10 V3 -> V4 | AP V3 -> V4 | Queries up / down (P@10) | Main confusions in the top 10 (V3 -> V4) |
|---|---|---|---|---|
| airport | 0.304 -> 0.364 (+0.060, 95% CI 0.025 to 0.096) | 0.151 -> 0.195 | 33 / 17 | railwaystation 14.5% -> 12.6%, commercial 11.4% -> 10.2% |
| bridge | 0.156 -> 0.174 (+0.017, CI -0.005 to 0.041) | 0.067 -> 0.084 | 20 / 16 | port 13.8% -> 16.6%, river 13.0% -> 12.1% |
| square | 0.212 -> 0.226 (+0.014, CI -0.011 to 0.041) | 0.078 -> 0.089 | 21 / 17 | denseresidential 20.1% -> 17.9% |

- **Airport** is the clear gain. **Bridge and square** improve slightly, but the P@10 intervals include zero (80 queries each; query-level bootstrap treats queries as independent), so only the AP gains (CI above zero) are reasonably supported. They remain among the weakest classes: V4 P@10 is 0.174 and 0.226, still about 28 and 26 of 80 queries with P@10 = 0.
- V4 does **not** fix all weak classes: stadium (-0.009), railwaystation (-0.015) and the large drops in park (-0.064), commercial (-0.059) and denseresidential (-0.030) show that re-ranking can hurt classes whose neighbourhoods are mixed. Largest AP gains: bareland, desert, playground, resort, parking.
- Files: `weak_class_analysis/` (`per_class_v3_vs_v4.csv`, `focus_class_top_confusions.csv`, confusion matrices for V3 and V4, query counts).

## 6. Reliability (HIGH / MEDIUM / LOW) recalibrated for V4 - LOFO part
The Week 4 rule (top-1 similarity, thresholds 0.930 / 0.960) was built for V2's similarity scale; applied to V3-style similarity it puts 89% of queries in LOW, so it cannot be reused. Same procedure as Week 4 on the 2,400 LOFO queries:
- Spearman with per-query Class P@10: `1 - D_V4(top-1)` **0.562**, V3 similarity of V4's top-1 0.342, top1-top2 gap -0.060. The first was chosen.
- Thresholds at the 20th / 70th percentile: LOW < 0.6419, HIGH >= 0.8078.
- Leave-fold-out check (thresholds from the other four folds, applied to the held-out fold):

| Label | Share of queries | Class P@10 | ML P@10 |
|---|---:|---:|---:|
| HIGH | 30.1% | 0.814 | 0.613 |
| MEDIUM | 50.0% | 0.473 | 0.426 |
| LOW | 19.9% | 0.276 | 0.327 |

HIGH > MEDIUM > LOW on all five folds. Evaluating these frozen categories on the test set is part of the final test and is **pending**. Files: `reliability_recalibration/`.

## 7. Final test (step 4) - pending
`final_test.py` does, in order: checks the test-split caches exist; reproduces V2 against the Week 4 result (overall table and per-query Class P@10) and stops if it does not match (a separate message if the query names do not line up with the Week 4 file); **writes the lock file `final_test/FINAL_TEST_DONE.json` as soon as that check passes** (so a crash later cannot lead to a silent second look at the test set); saves the V2/V3/V4 distance matrices to `final_test/test_distance_matrices_v2_v3_v4.npz` (so a crash in the analysis can be redone from disk, not by re-running the test); evaluates V2, V3 and V4 on the 600 test queries against the 2,400 gallery with the unchanged evaluator; paired query-level bootstrap (10,000 resamples) for V4-vs-V2 and V4-vs-V3 plus per-class sign tests; **weak-class reporting for square, bridge and airport** (per-class rows, a pooled 60-query bootstrap, and top-10 confusion tables; reporting only, since 20 queries per class is very noisy); and the frozen reliability categories.
Reliability note: the thresholds were set on 1,920-image LOFO galleries while the test gallery has 2,400 images, and the Jaccard part of `D_V4` depends on gallery density, so the HIGH/MEDIUM/LOW shares on test may differ from LOFO (about 30/50/20). They are reported, not re-tuned.
Status: `--dry-run` passes. It runs the whole reporting code (bootstrap, sign tests, weak-class pooled bootstrap, confusions, reliability) on LOFO fold 1 into a temporary folder, and V3/V4 values equal the LOFO values to 1e-9. V2 is only smoke-tested, and the V2 reproduction check can only run on the real test set. With the test caches missing, the script exits with the list of missing files and writes nothing.
**Needed:** `cache/test_hsv_16x16`, `test_lbp_r1_p8`, `test_lbp_r2_p16`, `test_hog_128_o9_c16_b2` (`.npy` + `_paths.json`), built with `Week1/cache_features.py` and `Week3/cache_test_r2.py`, which need the test images. `Week4/final_test/final_test.py` (the active Week 4 script) was not in my upload, so V2 is reproduced from `Week4/v2_config.json` and the Week 4 outputs.
Repo checks before the run: `Week 5/distances.py` must be committed (`cv_experiment.py` imports it); `week7_common.py` imports `neighborhood_context` and falls back to `neighbourhood_context` if your file is spelled that way; folders `Week 5` and `Week 6` (with spaces) are expected at the project root.

## 8. Honest note on test-set use
Per Week 4's own disclosure, the test set was already used in earlier weeks: eight Week 1 fusion configurations and V1 were evaluated on it, and V2 was evaluated in Week 4. The test set is therefore **not a fully blind holdout for the project as a whole**, and the final test is a final empirical comparison rather than a blind estimate. From the code supplied, V3 and V4 were selected only with train-set LOFO (the cache holds only train features), so they have not been tuned on the test set. All Week 7 choices (NE rejection, k2 = 3, reliability thresholds) used train/LOFO data only.

## 9. Not tested
Spatial, Gabor or LAB features were not tested (optional; they would need the image files and would have to precede the final test). V4 is frozen without them.

## 10. Reproduce
From the project root (folder containing `Week1 ... Week 7`, `cache/`, `multilabel_.csv`):

    python "Week 7/k2_check.py"
    python "Week 7/weak_class_analysis.py"
    python "Week 7/reliability_recalibration.py"
    python "Week 7/ne_ablation/ne_ablation.py"          # ablation only
    python "Week 7/final_test.py" --dry-run              # plumbing check, train data
    python "Week 7/final_test.py"                        # ONCE, after the test caches exist

Each script prints a log; saved logs are `*_run_log.txt`. Shared helpers: `week7_common.py`.

## 11. Limitations
Five folds are limited evidence; query-level bootstrap assumes independent queries; features are global (no spatial layout); only class labels were available for confusion analysis; no image files were supplied, so rankings are given as file names and classes.
