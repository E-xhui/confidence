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

## All Confidence Channels

| source | method | ECE | rho_veracity | FG(mis) | n samples |
|---|---|---:|---:|---:|---:|
| conf_ptrue | identity | 0.3949 | 0.0038 | 0.5651 | 18750 |
| conf_ptrue | temperature_scaling | 0.3222 | 0.0032 | 0.2071 | 18750 |
| conf_ptrue | platt_logistic | 0.0176 | 0.0032 | 0.2756 | 18750 |
| conf_ptrue | isotonic_relabel | 0.0019 | 0.0022 | 0.2997 | 18750 |
| conf_verbalized | identity | 0.4835 | 0.0014 | 0.2590 | 18750 |
| conf_verbalized | temperature_scaling | 0.3237 | 0.0009 | 0.1773 | 18750 |
| conf_verbalized | platt_logistic | 0.0043 | 0.0009 | 0.2685 | 18750 |
| conf_verbalized | isotonic_relabel | 0.0073 | 0.0007 | 0.2633 | 18750 |
| conf_seqlik | identity | 0.6679 | 0.0096 | 0.4582 | 18750 |
| conf_seqlik | temperature_scaling | 0.3334 | 0.0128 | 0.1356 | 18750 |
| conf_seqlik | platt_logistic | 0.0489 | 0.0128 | 0.3432 | 18750 |
| conf_seqlik | isotonic_relabel | 0.0046 | 0.0054 | 0.3645 | 18750 |