# Evidence

Reproduction scripts for measurements that justify a guard or a design decision.
They are not tests — they are the workings behind a claim that would otherwise read
as an assertion in `LIMITATIONS.md` or a comment in the source.

Not linted or run by CI: they take minutes, need real datasets, and their output is
a number in a doc rather than a pass/fail.

| Script | Answers | Justifies |
|---|---|---|
| `lever_vs_metric_sweep.py` | Does each canonical lever move a threshold metric and a ranking metric equally? | `supervisor.dead_lever_reason` and `canonical_moves` withholding threshold levers on a threshold-free metric |

## lever_vs_metric_sweep.py

Five datasets (churn, adult income, breast cancer, two synthetic at 8% and 35%
positive), five levers, no LLM involved. Result:

```
lever                                 |Δf1|  |Δavg_prec|  |Δroc_auc|
imbalance: class_weight=balanced     0.0295       0.0031      0.0011
threshold: tuned on train            0.0203       0.0000      0.0000
model-swap: RandomForest             0.0407       0.0416      0.0134
model-swap: LogisticRegression       0.0396       0.0299      0.0126
hyperparam: deeper/more iters        0.0069       0.0082      0.0027
```

Threshold tuning moves the ranking metrics by **exactly** 0.0000 on every dataset —
not approximately, but by construction, since a ranking metric is invariant to the
decision threshold. Class weighting is the same story about ten times weaker. The
levers that change the model itself move both.

That is why a threshold lever is withheld rather than merely discouraged when the
run's metric is threshold-free.
