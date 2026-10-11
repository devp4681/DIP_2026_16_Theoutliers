# Week 2 – Stronger retrieval engine

Put these files in the same folder as your Week 1 code (`cache_features.py`, `week1_evaluate.py`),
next to `images_tr/`, `images_test/` and `multilabel .csv`.

    python cache_features.py            # Week 1 cache (skipped if already cached)
    python week2_spatial_features.py    # NEW: 2x2 spatial HSV + LBP cache
    python week2_experiments.py         # validation tuning -> best_weights.json (~5-8 min at step 0.1)
    python week2_visualize.py           # top-5 figures in ./figures
    python week2_experiments.py --final-test   # ONCE, at the very end (600 test queries)

Outputs: split_indices.json, validation_results.csv, feature_comparison.csv, best_weights.json, test_results.csv

Later weeks:  from retrieval_engine_v1 import RetrievalEngineV1
