# H3 Calibration Before/After

Question-level split: calibration methods are fit on the calibration half and reported on the held-out evaluation half.

Primary confidence: `conf_ptrue`

## Primary Table

| source | method | ECE | NLL | Brier | FG(mis) | rho_topicality | rho_answer_support | rho_veracity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| conf_ptrue | identity | 0.3949 | 4.2843 | 0.3876 | 0.5651 | 0.4092 | 0.1985 | 0.0038 |
| conf_ptrue | temperature_scaling | 0.3222 | 0.6695 | 0.2383 | 0.2071 | 0.4239 | 0.2054 | 0.0032 |
| conf_ptrue | platt_logistic | 0.0176 | 0.4093 | 0.1310 | 0.2756 | 0.4239 | 0.2054 | 0.0032 |
| conf_ptrue | isotonic_relabel | 0.0019 | 0.4049 | 0.1297 | 0.2997 | 0.4025 | 0.2281 | 0.0022 |
| conf_ptrue | condition_oracle_logistic | 0.0187 | 0.2609 | 0.0788 | 0.0054 | 0.1452 | 0.0000 | 0.7744 |
| conf_ptrue | staged_oracle_logistic | 0.0180 | 0.2617 | 0.0789 | 0.0274 | 0.1651 | 0.0017 | 0.7399 |
| conf_ptrue | condition_only_oracle | 0.0109 | 0.2908 | 0.0873 | 0.0083 | 0.1531 | 0.0052 | 0.8417 |

## All Confidence Channels

| source | method | ECE | rho_veracity | FG(mis) | n samples |
|---|---|---:|---:|---:|---:|
| conf_ptrue | identity | 0.3949 | 0.0038 | 0.5651 | 18750 |
| conf_ptrue | temperature_scaling | 0.3222 | 0.0032 | 0.2071 | 18750 |
| conf_ptrue | platt_logistic | 0.0176 | 0.0032 | 0.2756 | 18750 |
| conf_ptrue | isotonic_relabel | 0.0019 | 0.0022 | 0.2997 | 18750 |
| conf_ptrue | condition_oracle_logistic | 0.0187 | 0.7744 | 0.0054 | 18750 |
| conf_ptrue | staged_oracle_logistic | 0.0180 | 0.7399 | 0.0274 | 18750 |
| conf_ptrue | condition_only_oracle | 0.0109 | 0.8417 | 0.0083 | 18750 |
| conf_verbalized | identity | 0.4835 | 0.0014 | 0.2590 | 18750 |
| conf_verbalized | temperature_scaling | 0.3237 | 0.0009 | 0.1773 | 18750 |
| conf_verbalized | platt_logistic | 0.0043 | 0.0009 | 0.2685 | 18750 |
| conf_verbalized | isotonic_relabel | 0.0073 | 0.0007 | 0.2633 | 18750 |
| conf_verbalized | condition_oracle_logistic | 0.0175 | 0.6798 | 0.0086 | 18750 |
| conf_verbalized | staged_oracle_logistic | 0.0127 | 0.6467 | 0.0314 | 18750 |
| conf_verbalized | condition_only_oracle | 0.0109 | 0.8417 | 0.0083 | 18750 |
| conf_seqlik | identity | 0.6679 | 0.0096 | 0.4582 | 18750 |
| conf_seqlik | temperature_scaling | 0.3334 | 0.0128 | 0.1356 | 18750 |
| conf_seqlik | platt_logistic | 0.0489 | 0.0128 | 0.3432 | 18750 |
| conf_seqlik | isotonic_relabel | 0.0046 | 0.0054 | 0.3645 | 18750 |
| conf_seqlik | condition_oracle_logistic | 0.0213 | 0.7452 | 0.0145 | 18750 |
| conf_seqlik | staged_oracle_logistic | 0.0130 | 0.7165 | 0.0770 | 18750 |
| conf_seqlik | condition_only_oracle | 0.0109 | 0.8417 | 0.0083 | 18750 |

## Interpretation

- `identity`, `temperature_scaling`, `platt_logistic`, and `isotonic_relabel` are output-only calibrators: they see the confidence score and correctness labels, but not evidence condition.
- `condition_oracle_logistic`, `staged_oracle_logistic`, and `condition_only_oracle` are upper-bound condition-aware controls: they are allowed to use condition labels or the staged topicality/answer-support/veracity codes.
- If output-only methods reduce ECE while leaving `rho_veracity` near zero, but condition-aware controls can introduce a veracity component, H3 should be framed as an output-channel failure rather than a failure of correctness calibration alone.