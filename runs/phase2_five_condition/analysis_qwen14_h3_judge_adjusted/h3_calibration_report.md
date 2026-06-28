# H3 Calibration Before/After

Question-level split: calibration methods are fit on the calibration half and reported on the held-out evaluation half.

Primary confidence: `conf_ptrue`

## Primary Table

| source | method | ECE | NLL | Brier | FG(mis) | rho_topicality | rho_answer_support | rho_veracity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| conf_ptrue | identity | 0.5356 | 6.0977 | 0.5334 | 0.5824 | 0.0001 | 0.0534 | 0.0013 |
| conf_ptrue | temperature_scaling | 0.2988 | 0.7075 | 0.2571 | 0.1844 | 0.0008 | 0.0618 | 0.0014 |
| conf_ptrue | platt_logistic | 0.0081 | 0.5112 | 0.1668 | 0.2050 | 0.0008 | 0.0618 | 0.0014 |
| conf_ptrue | isotonic_relabel | 0.0072 | 0.5014 | 0.1632 | 0.2379 | 0.0574 | 0.0890 | 0.0008 |
| conf_ptrue | condition_oracle_logistic | 0.0278 | 0.3545 | 0.1078 | 0.0026 | 0.1356 | 0.0051 | 0.8213 |
| conf_ptrue | staged_oracle_logistic | 0.0245 | 0.3566 | 0.1089 | 0.0435 | 0.1532 | 0.0016 | 0.8139 |
| conf_ptrue | condition_only_oracle | 0.0099 | 0.3654 | 0.1129 | 0.0042 | 0.1411 | 0.0066 | 0.8523 |

## All Confidence Channels

| source | method | ECE | rho_veracity | FG(mis) | n samples |
|---|---|---:|---:|---:|---:|
| conf_ptrue | identity | 0.5356 | 0.0013 | 0.5824 | 18750 |
| conf_ptrue | temperature_scaling | 0.2988 | 0.0014 | 0.1844 | 18750 |
| conf_ptrue | platt_logistic | 0.0081 | 0.0014 | 0.2050 | 18750 |
| conf_ptrue | isotonic_relabel | 0.0072 | 0.0008 | 0.2379 | 18750 |
| conf_ptrue | condition_oracle_logistic | 0.0278 | 0.8213 | 0.0026 | 18750 |
| conf_ptrue | staged_oracle_logistic | 0.0245 | 0.8139 | 0.0435 | 18750 |
| conf_ptrue | condition_only_oracle | 0.0099 | 0.8523 | 0.0042 | 18750 |
| conf_verbalized | identity | 0.5060 | 0.0004 | 0.3611 | 18750 |
| conf_verbalized | temperature_scaling | 0.2928 | 0.0004 | 0.1695 | 18750 |
| conf_verbalized | platt_logistic | 0.0113 | 0.0004 | 0.3257 | 18750 |
| conf_verbalized | isotonic_relabel | 0.0096 | 0.0002 | 0.3116 | 18750 |
| conf_verbalized | condition_oracle_logistic | 0.0116 | 0.5058 | 0.0081 | 18750 |
| conf_verbalized | staged_oracle_logistic | 0.0165 | 0.5065 | 0.0654 | 18750 |
| conf_verbalized | condition_only_oracle | 0.0099 | 0.8523 | 0.0042 | 18750 |
| conf_seqlik | identity | 0.6426 | 0.0056 | 0.3848 | 18750 |
| conf_seqlik | temperature_scaling | 0.2984 | 0.0051 | 0.1617 | 18750 |
| conf_seqlik | platt_logistic | 0.0370 | 0.0051 | 0.3609 | 18750 |
| conf_seqlik | isotonic_relabel | 0.0204 | 0.0014 | 0.3657 | 18750 |
| conf_seqlik | condition_oracle_logistic | 0.0329 | 0.7242 | 0.0030 | 18750 |
| conf_seqlik | staged_oracle_logistic | 0.0233 | 0.7099 | 0.0824 | 18750 |
| conf_seqlik | condition_only_oracle | 0.0099 | 0.8523 | 0.0042 | 18750 |

## Interpretation

- `identity`, `temperature_scaling`, `platt_logistic`, and `isotonic_relabel` are output-only calibrators: they see the confidence score and correctness labels, but not evidence condition.
- `condition_oracle_logistic`, `staged_oracle_logistic`, and `condition_only_oracle` are upper-bound condition-aware controls: they are allowed to use condition labels or the staged topicality/answer-support/veracity codes.
- If output-only methods reduce ECE while leaving `rho_veracity` near zero, but condition-aware controls can introduce a veracity component, H3 should be framed as an output-channel failure rather than a failure of correctness calibration alone.