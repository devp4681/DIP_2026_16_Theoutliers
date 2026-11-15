# Final results folder (Week 8, task 1)

**Final-test status: completed.**

## Frozen system
V4 = V3 + gallery-only k-reciprocal re-ranking. V3 weights HSV 0.20 / LBP-r1 0.40 / LBP-r2 0.20 / HOG 0.20 (Hellinger, Hellinger, Hellinger, cosine; raw fusion). k1 = 30, k2 = 3, lambda = 0.5. Reliability score = 1 - D_V4(top-1), LOW < 0.6419, HIGH >= 0.8078. Full details: `config/final_config.json`.

## 1. V2 vs V3 vs V4 (LOFO, mean +/- std over 5 folds)
| System | Class P@5 | ML P@5 | Jaccard@5 | nDCG@5 | Class P@10 | ML P@10 | Jaccard@10 | nDCG@10 | Class mAP | ML mAP | selection_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V2 | 0.5047 +/- 0.0071 | 0.4447 +/- 0.0078 | 0.6864 +/- 0.0051 | 0.7067 +/- 0.0066 | 0.4336 +/- 0.0034 | 0.4080 +/- 0.0081 | 0.6594 +/- 0.0041 | 0.6901 +/- 0.0057 | 0.2123 +/- 0.0014 | 0.2411 +/- 0.0035 | 0.3970 +/- 0.0033 |
| V3 | 0.5832 +/- 0.0113 | 0.4940 +/- 0.0040 | 0.7235 +/- 0.0041 | 0.7433 +/- 0.0045 | 0.5069 +/- 0.0084 | 0.4556 +/- 0.0064 | 0.6982 +/- 0.0029 | 0.7286 +/- 0.0032 | 0.2547 +/- 0.0039 | 0.2738 +/- 0.0042 | 0.4439 +/- 0.0046 |
| V4 | 0.5926 +/- 0.0112 | 0.4960 +/- 0.0067 | 0.7219 +/- 0.0044 | 0.7397 +/- 0.0049 | 0.5361 +/- 0.0125 | 0.4626 +/- 0.0042 | 0.7022 +/- 0.0011 | 0.7298 +/- 0.0030 | 0.3206 +/- 0.0045 | 0.2938 +/- 0.0040 | 0.4686 +/- 0.0042 |

Each recorded Week 5/6 value was reproduced to 4 decimals (`lofo_comparison/reproduction_checks.csv`, 20 of 20).

## 2. Paired comparison
Fold-level deltas (5 folds):

| comparison | metric | mean_delta | relative_pct | folds_improved |
|---|---|---|---|---|
| V4 - V2 | Class P@10 | 0.1025 | 23.6379 | 5 |
| V4 - V2 | ML P@10 | 0.0546 | 13.3769 | 5 |
| V4 - V2 | nDCG@10 | 0.0397 | 5.7482 | 5 |
| V4 - V2 | Class mAP | 0.1083 | 51.0260 | 5 |
| V4 - V2 | ML mAP | 0.0527 | 21.8475 | 5 |
| V4 - V2 | selection_score | 0.0716 | 18.0210 | 5 |
| V4 - V3 | Class P@10 | 0.0292 | 5.7620 | 5 |
| V4 - V3 | ML P@10 | 0.0070 | 1.5456 | 4 |
| V4 - V3 | nDCG@10 | 0.0012 | 0.1618 | 2 |
| V4 - V3 | Class mAP | 0.0659 | 25.8668 | 5 |
| V4 - V3 | ML mAP | 0.0200 | 7.2940 | 5 |
| V4 - V3 | selection_score | 0.0247 | 5.5544 | 5 |

Query-level paired bootstrap (10,000 resamples over the 2,400 pooled held-out queries; queries are treated as independent, so the intervals are approximate):

| comparison | metric | mean_diff | ci95_low | ci95_high | queries_better | queries_worse |
|---|---|---|---|---|---|---|
| V4 - V2 | class_p10 | 0.1025 | 0.0935 | 0.1117 | 1233 | 465 |
| V4 - V2 | ml_p10 | 0.0546 | 0.0473 | 0.0619 | 1048 | 539 |
| V4 - V2 | class_ap | 0.1083 | 0.1027 | 0.1140 | 1860 | 540 |
| V4 - V2 | ml_ap | 0.0522 | 0.0490 | 0.0554 | 1984 | 395 |
| V4 - V3 | class_p10 | 0.0292 | 0.0226 | 0.0360 | 840 | 579 |
| V4 - V3 | ml_p10 | 0.0070 | 0.0014 | 0.0129 | 762 | 720 |
| V4 - V3 | class_ap | 0.0659 | 0.0620 | 0.0698 | 1706 | 694 |
| V4 - V3 | ml_ap | 0.0198 | 0.0175 | 0.0221 | 1362 | 1016 |

Caution on top-5 multi-label ranking: V4 is slightly LOWER than V3 on Jaccard@5 (-0.0016, 1/5 folds up) and nDCG@5 (-0.0036, 0/5 folds up); see `lofo_comparison/paired_fold_deltas.csv`. 
How to read this: V4 improves on V3 clearly in Class mAP, Class P@10 and ML mAP (all 5 folds, intervals far from zero). ML P@10 is a small gain (about +0.007, 4 of 5 folds, interval barely above zero) and nDCG@10 is **unchanged within noise** (+0.0012, 2 of 5 folds). A bootstrap interval above zero is not the same as a practically meaningful gain; five folds are limited evidence.

## 3. Weak classes (LOFO, 80 queries per class)
| class | V2_class_p10 | V2_class_ap | n_queries | V3_class_p10 | V3_class_ap | V4_class_p10 | V4_class_ap |
|---|---|---|---|---|---|---|---|
| airport | 0.223 | 0.116 | 80 | 0.304 | 0.151 | 0.364 | 0.195 |
| bridge | 0.142 | 0.084 | 80 | 0.156 | 0.067 | 0.174 | 0.084 |
| square | 0.189 | 0.072 | 80 | 0.212 | 0.078 | 0.226 | 0.089 |

Airport gains clearly from V3 to V4. Bridge and square gain slightly; their V3-to-V4 P@10 intervals include zero (`weak_classes/per_class_v3_vs_v4.csv`) and both remain among the weakest classes. Top confusions: `weak_classes/focus_class_top_confusions.csv`. V4 also made some classes worse (e.g. park, commercial, denseresidential).

## 4. Reliability (LOFO held-out; thresholds from the other four folds)
| label | n_queries | pct_queries | class_p10 | ml_p10 | class_ap |
|---|---|---|---|---|---|
| HIGH | 723.000 | 30.125 | 0.814 | 0.613 | 0.490 |
| MEDIUM | 1199.000 | 49.958 | 0.473 | 0.426 | 0.284 |
| LOW | 478.000 | 19.917 | 0.276 | 0.327 | 0.155 |

The thresholds were fixed on LOFO data with 1,920-image galleries; shares on the 2,400-image test gallery may differ and will be reported, not re-tuned.

## 5. Unsuccessful / not-adopted experiments
| experiment | week | reference | selection_score | delta_vs_reference | outcome | reason |
|---|---|---|---|---|---|---|
| Average Query Expansion (AQE) on V3 | 6 | V3 | 0.4462 | 0.0023 | rejected (ablation) | gain +0.0023 is below the +0.005 threshold; Class P@10 -0.0042, negative on 4 of 5 folds |
| AQE then k-reciprocal (Order 1) | 6 | V4 | 0.4659 | -0.0026 | rejected | lower than gallery-only k-reciprocal alone |
| Transductive k-reciprocal (earlier version) | 6 | V4 | 0.4698 | 0.0012 | replaced by gallery-only version | Week 6 kept it as an ablation; V4 uses the gallery-only single-query protocol instead |
| Per-query Z-score normalisation (on V2) | 5 | V2 raw fusion | 0.3813 | -0.0157 | rejected | lower than unnormalised V2 even after retuning (best 0.3954); recorded value, not rerun here |
| Neighbourhood expansion (NE) on V4 | 7 | V4 | 0.4679 | -0.0007 | rejected (ablation) | selection score -0.0007, 0/5 folds up; failed all three Week 6 criteria; every one of 9 settings below V4 |
| V4 with k2=1 | 7 | V4 (k2=3) | 0.4654 | -0.0032 | not adopted | gain < 0.005 needed to replace k2=3 (selection score lower); LOFO selected k2=3 on all folds |
| V4 with k2=2 | 7 | V4 (k2=3) | 0.4674 | -0.0012 | not adopted | gain < 0.005 needed to replace k2=3 (selection score lower); LOFO selected k2=3 on all folds |

## 6. FINAL TEST (600 queries vs 2,400 gallery, run once)
| Model | Class P@5 | ML P@5 | Jaccard@5 | nDCG@5 | Class P@10 | ML P@10 | Jaccard@10 | nDCG@10 | Class mAP | ML mAP | selection_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V2 | 0.4183 | 0.3857 | 0.6628 | 0.6806 | 0.3867 | 0.3650 | 0.6491 | 0.6740 | 0.1889 | 0.2278 | 0.3685 |
| V3 | 0.4777 | 0.4247 | 0.6940 | 0.7125 | 0.4352 | 0.4025 | 0.6799 | 0.7060 | 0.2275 | 0.2561 | 0.4054 |
| V4 | 0.4960 | 0.4083 | 0.6866 | 0.7054 | 0.4667 | 0.4003 | 0.6786 | 0.7030 | 0.2928 | 0.2716 | 0.4269 |

Paired bootstrap:

| comparison | metric | mean_diff | ci95_low | ci95_high | queries_better | queries_worse |
|---|---|---|---|---|---|---|
| V4 - V2 | class_p10 | 0.0800 | 0.0628 | 0.0973 | 253 | 119 |
| V4 - V2 | ml_p10 | 0.0353 | 0.0210 | 0.0498 | 220 | 152 |
| V4 - V2 | class_ap | 0.1039 | 0.0925 | 0.1153 | 449 | 151 |
| V4 - V2 | ml_ap | 0.0430 | 0.0378 | 0.0486 | 491 | 97 |
| V4 - V3 | class_p10 | 0.0315 | 0.0180 | 0.0450 | 179 | 138 |
| V4 - V3 | ml_p10 | -0.0022 | -0.0140 | 0.0097 | 171 | 175 |
| V4 - V3 | class_ap | 0.0653 | 0.0581 | 0.0728 | 416 | 184 |
| V4 - V3 | ml_ap | 0.0152 | 0.0118 | 0.0187 | 327 | 260 |

Reliability on test (frozen thresholds):

| label | n_queries | class_p10 | ml_p10 | class_ap | pct_queries |
|---|---|---|---|---|---|
| HIGH | 125 | 0.854 | 0.550 | 0.543 | 20.833 |
| MEDIUM | 308 | 0.448 | 0.410 | 0.285 | 51.333 |
| LOW | 167 | 0.211 | 0.270 | 0.121 | 27.833 |

Reading of the test result: V4 vs V3 is higher on Class P@5 (+0.0183), Class P@10 (+0.0315), Class mAP (+0.0653), ML mAP (+0.0152), selection_score (+0.0214) and lower on ML P@5 (-0.0163), Jaccard@5 (-0.0075), nDCG@5 (-0.0071), ML P@10 (-0.0022), Jaccard@10 (-0.0013), nDCG@10 (-0.0030). V4 vs V2 is higher on Class P@5, ML P@5, Jaccard@5, nDCG@5, Class P@10, ML P@10, Jaccard@10, nDCG@10, Class mAP, ML mAP, selection_score. Check `final_test/paired_bootstrap.csv` before calling any difference real: e.g. the V4 - V3 ML P@10 interval includes zero. The weak-class test numbers (20 queries per class) are noisy and differ from LOFO (bridge does not improve on test).

Note: the test set was also used in earlier weeks (Week 1 fusion configurations, V1 and V2), so it is not a fully blind holdout for the project as a whole; V3 and V4 were selected on train/LOFO data only.

Status note: this README was regenerated after the final test; sections 1-5 remain LOFO (training data) results.

## Folder map
`config/` frozen configurations | `lofo_comparison/` per-fold, summary, paired deltas, bootstrap, reproduction checks | `weak_classes/` | `reliability/` | `k2_check/` | `failed_experiments/` (AQE, NE and others, with NE grid files) | `figures/` | `final_test/` | `MANIFEST.sha256` (file hashes).

## Reproduce
```
python "Week 7/k2_check.py"
python "Week 7/weak_class_analysis.py"
python "Week 7/reliability_recalibration.py"
python "Week 8/build_final_results.py"
python "Week 8/verify_final_results.py"
```
