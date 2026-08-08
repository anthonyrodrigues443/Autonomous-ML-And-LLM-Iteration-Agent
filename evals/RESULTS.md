# iterate eval results

Internal. Not a product feature. Regenerate with `make eval-report`.

- generated: 2026-08-08 08:38 UTC
- model: `gemma4:12b` on `ollama`
- budget: 10 iterations, patience 3, 3 repeats per cell
- conditions fingerprint: `5174903b3595`

## Captured headroom

Each cell is the median across repeats of the fraction of AVAILABLE gain the agent captured, with the spread in brackets. Not clamped: above 100% means the run beat the brute-force ceiling, which is a real outcome, not an error.

No cells recorded under these conditions yet. Run `make eval`.

## Ceilings

| dataset | metric | data hash | baseline | ceiling | method | measured |
|---|---|---|---|---|---|---|
| adult_income | f1 | `6fd48562938c85d8` | 0.7187 | 0.7259 | brute_force_sweep_v1 (9 models) | 2026-08-08 |
| churn | average_precision | `608be8cce4edba4f` | 0.6449 | 0.6449 | brute_force_sweep_v1 (9 models) | 2026-08-08 |
| diamonds | rmse | `9574730b03aba241` | 549.0567 | 537.1413 | brute_force_sweep_v1 (9 models) | 2026-08-08 |
| heart_risk | average_precision | `f30b84b747c1d263` | 0.8967 | 0.8967 | brute_force_sweep_v1 (9 models) | 2026-08-08 |
| laptop_price | rmse | `e5cd3296b994d10e` | 411.8904 | 248.8509 | brute_force_sweep_v1 (9 models) | 2026-08-08 |
| mobile_price | accuracy | `f9e8cd3154b8684a` | 0.9450 | 0.9450 | brute_force_sweep_v1 (9 models) | 2026-08-08 |
