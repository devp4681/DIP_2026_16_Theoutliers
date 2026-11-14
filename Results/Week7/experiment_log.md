# Week 7 experiment log

All rows: 5-fold LOFO, train data only, test set not used. Run commands are from the project root.

| # | Experiment | Method / configuration | Command | Result (selection score, mean of 5 folds) | Decision | Reason |
|---|---|---|---|---|---|---|
| 7.0 | Baseline reproduction | V3 and V4 (k1=30, k2=3, lambda=0.5) | any script below | V3 0.4439, V4 0.4686 (all six metrics match Week 6 to 1e-4) | verified | - |
| 7.1 | Neighbourhood expansion (NE) | top-m expansion on V4's ranking; m {3,5,10} x beta {0.2,0.35,0.5} | `python "Week 7/ne_ablation/ne_ablation.py"` | 0.4679 (-0.0007, 0/5 folds up) | REJECTED, kept as ablation | failed all 3 Week 6 criteria |
| 7.2 | k2 check | k2 in {1,2,3}; k1=30, lambda=0.5 fixed | `python "Week 7/k2_check.py"` | k2=1 0.4654, k2=2 0.4674, k2=3 0.4686 | KEEP k2=3; V4 frozen | no alternative gains > 0.005; k2=1 and k2=2 are -0.0032 and -0.0012; LOFO picked k2=3 on all 5 folds |
| 7.3 | Weak-class analysis V3 vs V4 | pooled 2,400 held-out queries | `python "Week 7/weak_class_analysis.py"` | see README section 5 | analysis only | - |
| 7.4 | Reliability recalibration for V4 | score chosen by Spearman; thresholds at 20th / 70th percentile; held-out-fold check | `python "Week 7/reliability_recalibration.py"` | Spearman 0.562; held-out Class P@10 HIGH 0.814 / MEDIUM 0.473 / LOW 0.276 | frozen on train data | HIGH > MEDIUM > LOW on 5/5 folds |
| 7.5 | Final test V2 vs V3 vs V4 | 600 test queries, 2,400 gallery | `python "Week 7/final_test.py"` | NOT RUN | PENDING | test-split feature caches (`cache/test_*.npy`) were not supplied; script ready and dry-run verified |
| 7.6 | Spatial / Gabor / LAB features | - | - | NOT TESTED | not done | optional; would need image files and must be done before the final test |
