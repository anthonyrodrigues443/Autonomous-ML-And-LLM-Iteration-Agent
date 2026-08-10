# iterate eval results

Internal. Not a product feature. Regenerate with `make eval-report`.

- generated: 2026-08-10 12:48 UTC
- model: `gemma4:12b` on `ollama`
- budget: 10 iterations, patience 3, 2 repeats per cell
- conditions fingerprint: `25143ee9f9b7`

## Captured headroom

Each cell is the median across repeats of the fraction of AVAILABLE gain the agent captured, with the spread in brackets. Not clamped: above 100% means the run beat the brute-force ceiling, which is a real outcome, not an error.

| version | adult_income | churn | diamonds | laptop_price |
|---|---|---|---|---|
| **0.1.3** | 7% (0-14) | unreadable x2 | unreadable x2 | unreadable x2 |
| **0.4.0** | - | - | - | - |
| **dev** | - | - | - | - |

## Ceilings

| dataset | metric | data hash | baseline | ceiling | method | measured |
|---|---|---|---|---|---|---|
| adult_income | f1 | `6fd48562938c85d8` | 0.7187 | 0.7259 | brute_force_sweep_v1 (9 models) | 2026-08-08 |
| churn | average_precision | `608be8cce4edba4f` | 0.6449 | 0.6467 | feature_treatment_sweep_v2 (best: frequency-encoding, 8 treatments) | 2026-08-10 |
| diamonds | rmse | `9574730b03aba241` | 549.0567 | 527.4767 | feature_treatment_sweep_v2 (best: numeric-interactions, 7 treatments) | 2026-08-10 |
| hate_speech_davidson | f1_macro | `e63c93bf6eb7d7b9` | 0.5862 | 0.6822 | prompt_technique_sweep_v1 (best: few-shot, 200 records) | 2026-08-09 |
| heart_risk | average_precision | `f30b84b747c1d263` | 0.8967 | 0.9000 | feature_treatment_sweep_v2 (best: calibrated, 8 treatments) | 2026-08-10 |
| laptop_price | rmse | `e5cd3296b994d10e` | 411.8904 | 248.8509 | brute_force_sweep_v1 (9 models) | 2026-08-08 |
| mobile_price | accuracy | `f9e8cd3154b8684a` | 0.9450 | 0.9550 | feature_treatment_sweep_v2 (best: numeric-interactions, 8 treatments) | 2026-08-10 |
| toxicity_jigsaw | f1 | `e0d3d7a24cbab067` | 0.8398 | 0.8681 | prompt_technique_sweep_v1 (best: define-plus-few-shot, 200 records) | 2026-08-09 |

## Cell detail

| version | dataset | ok/run | best per repeat | experiments | failed | rejected | median min |
|---|---|---|---|---|---|---|---|
| 0.1.3 | adult_income | 2/2 | none, 0.7198 | 7 | 0 | 0 | 2 |
| 0.1.3 | churn | 0/2 | - | 0 | 0 | 0 | - |
| 0.1.3 | diamonds | 0/2 | - | 0 | 0 | 0 | - |
| 0.1.3 | laptop_price | 0/2 | - | 0 | 0 | 0 | - |
