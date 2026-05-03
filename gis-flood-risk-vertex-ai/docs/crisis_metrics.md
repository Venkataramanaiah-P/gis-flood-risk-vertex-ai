# Crisis Evaluation Standards for GIS Flood Mapping

## Why Crisis Metrics Differ from Standard ML Metrics

In standard ML, we optimize for Overall Accuracy or F1.
In **crisis GIS**, the cost of different errors is asymmetric:

| Error Type | Consequence | Acceptable? |
|---|---|---|
| False Positive (false alarm) | Unnecessary evacuation | YES — costly but safe |
| False Negative (missed flood) | People not evacuated | NO — catastrophic |

This asymmetry drives our metric priorities.

---

## Required Thresholds

### Recall > 0.90 (MANDATORY)
```
Recall = TP / (TP + FN)
```
Recall measures how many REAL flood events we correctly detect.
A recall of 0.90 means we catch 90% of actual floods.
The remaining 10% are missed — unacceptable if > 10%.

**Rule:** Never deploy a crisis model with Recall < 0.90

### Kappa > 0.75 (MANDATORY for official reports)
```
Kappa = (Observed Accuracy - Expected Accuracy) /
        (1 - Expected Accuracy)
```

| Kappa Range | Interpretation |
|---|---|
| < 0.40 | Poor agreement |
| 0.40 - 0.60 | Moderate |
| 0.60 - 0.75 | Substantial |
| **> 0.75** | **Strong — required for official use** |
| > 0.90 | Near perfect |

Kappa accounts for chance agreement — Overall Accuracy alone
is misleading with imbalanced flood data.

### F1 Score > 0.85 (Operational target)
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
Balances precision and recall for day-to-day operational use.

---

## Field Validation Protocol

Before deploying any flood map for crisis decisions:

1. **GPS Ground Truth** — collect points at known flood/dry sites
2. **Village Reports** — cross-check with local submergence records
3. **Confusion Matrix** — compute all 4 cells (TP, FP, TN, FN)
4. **Kappa Calculation** — must exceed 0.75 for official submission
5. **Recall Verification** — must exceed 0.90 for evacuation decisions

---

## Reference Standards
- Meridial Geospatial Crisis Response Verification (2025)
- NDWI threshold: Gao (1996) — > 0.3 for water detection
- Kappa threshold: Landis & Koch (1977) classification
- Crisis recall standard: UNOSAT flood mapping protocols
