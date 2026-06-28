# Construct Validation and Figure Bundle

Construct validation fits the spec-aligned mixed model over evidence conditions:

`logit(confidence_qe) ~ C(condition) + u_q + delta_qe + eps`

Then it correlates the fitted question random intercept `u_q` with `log1p(popularity_raw)`.

## u_q vs popularity

| model | confidence | n | Pearson r | Pearson p | Spearman rho | Spearman p | converged |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-7B | conf_ptrue | 1500 | -0.0692 | 0.0074 | -0.0694 | 0.0072 | True |
| Qwen2.5-7B | conf_verbalized | 1500 | -0.0689 | 0.0076 | -0.0605 | 0.0191 | True |
| Qwen2.5-14B | conf_ptrue | 1500 | -0.2728 | 0.0000 | -0.2527 | 0.0000 | True |
| Qwen2.5-14B | conf_verbalized | 1500 | 0.0720 | 0.0052 | 0.0508 | 0.0490 | True |

## Figures

- `figures/fig_condition_acc_conf.pdf`: condition-level accuracy and P(True).
- `figures/fig_rho_decomposition.pdf`: staged variance decomposition.
- `figures/fig_h3_calibration.pdf`: ECE before/after and rho_veracity under calibration.
- `figures/fig_closed_known.pdf`: closed-known collapse under misleading evidence.
- `figures/fig_construct_uq_popularity.pdf`: construct validation scatter.
