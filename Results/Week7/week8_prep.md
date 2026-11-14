# Week 8 preparation (final test numbers to be added after `final_test.py` is run)

## Report summary
V1 to V3: HSV, LBP (r1, r2) and HOG descriptors fused with tuned weights (V2 correlation similarities; V3 Hellinger/cosine). V4 adds gallery-only k-reciprocal re-ranking (k1=30, k2=3, lambda=0.5). On 5-fold LOFO, V4 improves selection score 0.4439 -> 0.4686 (5/5 folds), Class mAP 0.2547 -> 0.3206 (+25.9%), Class P@10 0.5069 -> 0.5361, ML mAP 0.2738 -> 0.2938; ML P@10 gains are small (+0.007) and nDCG@10 is unchanged. Week 7: neighbourhood expansion rejected (ablation); k2 in {1,2,3} tested and k2=3 kept; weak classes (airport improves clearly, bridge/square slightly, stadium/railwaystation not); reliability labels recalibrated for V4. Final test: pending. Test set was used in earlier weeks (disclose).

## Slide outline
1. Objective. 2. V1-V4 progression and V4 pipeline. 3. Week 7 plan: validate and freeze. 4. NE ablation (negative result). 5. k2 check. 6. V3 vs V4 on LOFO, with honest metric table. 7. Weak classes (airport, bridge, square, confusions). 8. Reliability calibration. 9. Final test V2 vs V3 vs V4 (add after run) and statistics. 10. Limitations: test-set disclosure, global features, remaining weak classes.

## Demo ideas (from LOFO outputs; images not supplied)
- Success: resort queries (10/10 resort).
- Difficult: beach_6, bridge_21 (P@10 = 0 for V3 and V4).
- Class that improves: airport (P@10 0.304 -> 0.364); class that does not: stadium / railwaystation.
- Reliability: show one HIGH, one LOW query (`final_test/per_query_v2_v3_v4.csv` after the final test).

## Viva Q&A
- **What does V4 do?** Fuses four descriptor distances (V3), then re-ranks with k-reciprocal neighbourhoods built from gallery-gallery distances, blended 50/50 with the original distance.
- **Why k-reciprocal?** Mutual nearest neighbours are more reliable than one-sided ones; it lifted Class mAP by about 26%.
- **What does k2 do?** The number of neighbours averaged for query expansion inside the re-ranking; k2=1 means none. k2=3 gave the best selection score on 5/5 LOFO folds.
- **Why was NE rejected?** It added a second neighbourhood step that re-used information V4 already uses; selection score -0.0007, 0/5 folds positive.
- **Did you avoid leakage?** All Week 7 choices used only train-set LOFO; labels enter only the metrics; thresholds for the held-out reliability check came from the other four folds.
- **Was the test set blind?** No: it was used in Weeks 1, 2 and 4 (disclosed). V3/V4 were selected without it, so the final test is a final comparison, not a blind estimate.
- **Is V4 better everywhere?** No. Large gains in mAP; nDCG@10 flat; stadium, railwaystation, park and commercial got worse in LOFO.
- **How is reliability defined?** `1 - D_V4(top-1)`; thresholds 0.6419 / 0.8078 from LOFO percentiles; HIGH queries had Class P@10 0.81 versus 0.28 for LOW on held-out folds.
