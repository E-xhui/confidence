# H3 Calibration Before/After

Question-level split: calibration methods are fit on the calibration half and reported on the held-out evaluation half.

Primary confidence: `conf_ptrue`

## Primary Table

| source | method | ECE | NLL | Brier | FG(mis) | rho_topicality | rho_answer_support | rho_veracity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| conf_ptrue | identity | 0.3885 | 4.2095 | 0.3814 | 0.5555 | 0.4092 | 0.1985 | 0.0038 |
| conf_ptrue | temperature_scaling | 0.3143 | 0.6665 | 0.2369 | 0.2061 | 0.4239 | 0.2054 | 0.0032 |
| conf_ptrue | platt_logistic | 0.0162 | 0.4189 | 0.1348 | 0.2748 | 0.4239 | 0.2054 | 0.0032 |
| conf_ptrue | isotonic_relabel | 0.0027 | 0.4144 | 0.1334 | 0.2989 | 0.3996 | 0.2304 | 0.0023 |

## All Confidence Channels

| source | method | ECE | rho_veracity | FG(mis) | n samples |
|---|---|---:|---:|---:|---:|
| conf_ptrue | identity | 0.3885 | 0.0038 | 0.5555 | 18750 |
| conf_ptrue | temperature_scaling | 0.3143 | 0.0032 | 0.2061 | 18750 |
| conf_ptrue | platt_logistic | 0.0162 | 0.0032 | 0.2748 | 18750 |
| conf_ptrue | isotonic_relabel | 0.0027 | 0.0023 | 0.2989 | 18750 |
| conf_verbalized | identity | 0.4751 | 0.0014 | 0.2494 | 18750 |
| conf_verbalized | temperature_scaling | 0.3156 | 0.0009 | 0.1718 | 18750 |
| conf_verbalized | platt_logistic | 0.0041 | 0.0009 | 0.2618 | 18750 |
| conf_verbalized | isotonic_relabel | 0.0066 | 0.0009 | 0.2552 | 18750 |
| conf_seqlik | identity | 0.6592 | 0.0096 | 0.4486 | 18750 |
| conf_seqlik | temperature_scaling | 0.3247 | 0.0128 | 0.1260 | 18750 |
| conf_seqlik | platt_logistic | 0.0464 | 0.0128 | 0.3342 | 18750 |
| conf_seqlik | isotonic_relabel | 0.0067 | 0.0052 | 0.3570 | 18750 |