# T2 Conditional Calibration

## Main Comparison

| method | feature_group | B2 rho_veracity | H3 moment rho_veracity | ECE | FG(mis) | AUROC |
|---|---|---:|---:|---:|---:|---:|
| identity | source_confidence | 0.0040 | 0.0040 | 0.3966 | 0.5721 | 0.7643 |
| temperature_scaling | source_confidence | 0.0034 | 0.0034 | 0.3233 | 0.2099 | 0.7620 |
| platt_logistic | source_confidence | 0.0034 | 0.0034 | 0.0141 | 0.2779 | 0.7613 |
| isotonic_relabel | source_confidence | 0.0025 | 0.0025 | 0.0006 | 0.3043 | 0.7610 |
| conditional_gbm | all | 0.1840 | 0.1841 | 0.0060 | 0.1060 | 0.9405 |
| conditional_gbm | all_no_ctx_surface | 0.0141 | 0.0141 | 0.0099 | 0.2451 | 0.8937 |

## Feature Ablation

| feature_group | B2 rho_veracity | ECE | FG(mis) | AUROC |
|---|---:|---:|---:|---:|
| conf_only | 0.0224 | 0.0122 | 0.2706 | 0.8497 |
| conf_plus_prior | 0.0179 | 0.0096 | 0.2429 | 0.8902 |
| conf_plus_surface | 0.2482 | 0.0115 | 0.1037 | 0.9031 |
| conf_plus_selfcons | 0.0177 | 0.0102 | 0.2721 | 0.8555 |
| all_no_ctx_surface | 0.0141 | 0.0099 | 0.2451 | 0.8937 |
| all_no_selfcons | 0.2017 | 0.0081 | 0.1081 | 0.9362 |
| all | 0.1840 | 0.0060 | 0.1060 | 0.9405 |

## D1 Matched Surface

| method | B2 rho_veracity | ECE | FG(mis) | AUROC | n samples | n questions |
|---|---:|---:|---:|---:|---:|---:|
| matched_identity | 0.0052 | 0.3224 | 0.5735 | 0.8021 | 18120 | 982 |
| matched_conditional_gbm | 0.0067 | 0.0149 | 0.2542 | 0.9313 | 18120 | 982 |

## D2 Condition-Internal Surface Predictiveness

| condition | positive_rate | n_pos | balanced AUROC | unweighted AUROC | separable AUROC | ctx_len AUC | ctx_entity AUC | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sup | 0.6853 | 5140 | 0.5121 | 0.4996 | 0.5121 | 0.4887 | 0.4766 | 7500 |
| mis | 0.0045 | 34 | 0.2649 | 0.3063 | 0.7351 | 0.5365 | 0.4002 | 7500 |

## D3 Top Permutation Importances

| feature | importance_mean | importance_std |
|---|---:|---:|
| closed_book_correct_rate | 0.0807 | 0.0128 |
| ctx_len_tokens | 0.0612 | 0.0039 |
| ctx_entity_count | 0.0518 | 0.0066 |
| answer_agreement | 0.0293 | 0.0073 |
| conf_verbalized | 0.0106 | 0.0032 |
| conf_ptrue | 0.0093 | 0.0038 |
| conf_seqlik | 0.0091 | 0.0011 |
| conf_ptrue_group_mean | 0.0070 | 0.0032 |
| conf_verbalized_group_mean | 0.0058 | 0.0015 |
| conf_seqlik_group_mean | 0.0031 | 0.0012 |