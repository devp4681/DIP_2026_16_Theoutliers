# V1 vs V2 Decision

## Configurations
**V1 Configuration:**
- HSV: 0.10
- LBP: 0.70
- HOG: 0.20

**V2 Configuration (Multi-scale LBP):**
- HSV: 0.10
- LBP r=1, p=8: 0.35
- LBP r=2, p=16: 0.35
- HOG: 0.20

## Five-Fold Results
- V1 Selection Score: ~0.3933
- V2 Selection Score: ~0.3970
- V2 ranked 1st across all 5 folds compared to the other variants tested in Week 3.

## Paired Bootstrap Analysis (V2 - V1)
- **Observed Mean Class P@10 Difference**: ~0.4288 → ~0.4336 (+0.0048)
- **95% Confidence Interval**: [0.0006, 0.0090]
- **Interpretation**: V2 provides a small, consistent, modest improvement over V1 on validation. The query-level bootstrap interval is positive but narrow, so the result is treated as modest evidence rather than a strong significance claim. Note that query-level bootstrap assumes independence, which may be optimistic since queries within classes are correlated.

## Per-Class Delta Summary
- **Overall**: 17 classes improved, 13 worsened.
- **Weak Classes**: Of the 5 historically weak classes (bridge, square, stadium, railwaystation, airport), only `square` saw a material improvement (+0.0275). The others remained mostly unchanged or slightly worsened (e.g., stadium -0.0200). We cannot claim the overall gain broadly came from weak classes.
- **Structural Gains**: The multi-scale LBP improved classes such as `desert` (+0.0575) and `church` (+0.0450).
- **Texture Decreases**: Performance on dense organic textures decreased, notably in `farmland` (-0.0500) and `forest` (-0.0462).

## Final Decision
V2 remains the primary selected model because it won all 5 Week 3 folds, improved the overall selection score, and has a positive class-P@10 bootstrap interval—albeit by a small margin.
