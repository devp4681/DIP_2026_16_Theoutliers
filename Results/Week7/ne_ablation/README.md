# Neighbourhood expansion (NE) - negative-result ablation

This was my first Week 7 experiment. It is **not** part of the final system; it is kept as a documented negative result.

**Method.** `D_cand = (1 - beta) * D_V4 + beta * mean(D_V3 between the top-m V4 results and every gallery image)`. `beta = 0` equals V4.
Grid: m in {3, 5, 10} x beta in {0.2, 0.35, 0.5}; leave-one-fold-out selection of the setting; same 5-fold LOFO protocol and metrics as V4. Train data only.

**Result (5 folds, mean).**

| Metric | V4 | V4 + NE | Change | Folds improved |
|---|---:|---:|---:|---:|
| Selection score | 0.4686 | 0.4679 | -0.0007 | 0/5 |
| Class P@10 | 0.5361 | 0.5358 | -0.0003 | 1/5 |
| Class mAP | 0.3206 | 0.3176 | -0.0030 | 0/5 |
| ML P@10 | 0.4626 | 0.4625 | -0.0002 | 2/5 |
| ML mAP | 0.2938 | 0.2936 | -0.0002 | 0/5 |
| nDCG@10 | 0.7298 | 0.7301 | +0.0002 | 4/5 |

All nine settings were below V4 on the selection score, and the loss grew with beta. The Week 6 rule failed on all three criteria, so NE was rejected. NE adds about 0.07 s per fold of 480 queries.
Likely reason (consistent with the evidence, not proven): V4's k-reciprocal step already uses gallery-neighbourhood structure, and when V4's top-m contains wrong-class images the expansion reinforces them.

**Files.** `ne_ablation.py` (code), `run_log.txt`, `week7_results.json`, `week7_comparison_table.csv`, `grid_*.csv`, `per_query_results.csv`, `per_class_delta.csv`, `error_analysis_examples.csv`.
Run: `python "Week 7/ne_ablation/ne_ablation.py"` from the project root.
