# H3 Calibration Before/After

Question-level split: calibration methods are fit on the calibration half and reported on the held-out evaluation half.

Primary confidence: `conf_ptrue`

## Primary Table

| source | method | ECE | NLL | Brier | FG(mis) | rho_topicality | rho_answer_support | rho_veracity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| conf_ptrue | identity | 0.5492 | 6.2611 | 0.5464 | 0.5864 | 0.0001 | 0.0534 | 0.0013 |
| conf_ptrue | temperature_scaling | 0.3159 | 0.7105 | 0.2586 | 0.1884 | 0.0008 | 0.0618 | 0.0014 |
| conf_ptrue | platt_logistic | 0.0148 | 0.4912 | 0.1580 | 0.2040 | 0.0008 | 0.0618 | 0.0014 |
| conf_ptrue | isotonic_relabel | 0.0068 | 0.4814 | 0.1547 | 0.2363 | 0.0578 | 0.0892 | 0.0008 |
| conf_ptrue | condition_oracle_logistic | 0.0212 | 0.3385 | 0.1033 | 0.0027 | 0.0837 | 0.0126 | 0.8738 |
| conf_ptrue | staged_oracle_logistic | 0.0165 | 0.3419 | 0.1047 | 0.0550 | 0.1226 | 0.0021 | 0.8511 |
| conf_ptrue | condition_only_oracle | 0.0070 | 0.3474 | 0.1077 | 0.0036 | 0.0847 | 0.0152 | 0.9000 |

## All Confidence Channels

| source | method | ECE | rho_veracity | FG(mis) | n samples |
|---|---|---:|---:|---:|---:|
| conf_ptrue | identity | 0.5492 | 0.0013 | 0.5864 | 18750 |
| conf_ptrue | temperature_scaling | 0.3159 | 0.0014 | 0.1884 | 18750 |
| conf_ptrue | platt_logistic | 0.0148 | 0.0014 | 0.2040 | 18750 |
| conf_ptrue | isotonic_relabel | 0.0068 | 0.0008 | 0.2363 | 18750 |
| conf_ptrue | condition_oracle_logistic | 0.0212 | 0.8738 | 0.0027 | 18750 |
| conf_ptrue | staged_oracle_logistic | 0.0165 | 0.8511 | 0.0550 | 18750 |
| conf_ptrue | condition_only_oracle | 0.0070 | 0.9000 | 0.0036 | 18750 |
| conf_verbalized | identity | 0.5231 | 0.0004 | 0.3651 | 18750 |
| conf_verbalized | temperature_scaling | 0.3100 | 0.0004 | 0.1735 | 18750 |
| conf_verbalized | platt_logistic | 0.0198 | 0.0004 | 0.3167 | 18750 |
| conf_verbalized | isotonic_relabel | 0.0101 | 0.0002 | 0.3038 | 18750 |
| conf_verbalized | condition_oracle_logistic | 0.0123 | 0.6095 | 0.0071 | 18750 |
| conf_verbalized | staged_oracle_logistic | 0.0149 | 0.5735 | 0.0720 | 18750 |
| conf_verbalized | condition_only_oracle | 0.0070 | 0.9000 | 0.0036 | 18750 |
| conf_seqlik | identity | 0.6598 | 0.0056 | 0.3888 | 18750 |
| conf_seqlik | temperature_scaling | 0.3155 | 0.0051 | 0.1657 | 18750 |
| conf_seqlik | platt_logistic | 0.0551 | 0.0051 | 0.3663 | 18750 |
| conf_seqlik | isotonic_relabel | 0.0227 | 0.0027 | 0.3682 | 18750 |
| conf_seqlik | condition_oracle_logistic | 0.0306 | 0.7682 | 0.0025 | 18750 |
| conf_seqlik | staged_oracle_logistic | 0.0251 | 0.7250 | 0.0922 | 18750 |
| conf_seqlik | condition_only_oracle | 0.0070 | 0.9000 | 0.0036 | 18750 |

## Interpretation

- `identity`, `temperature_scaling`, `platt_logistic`, and `isotonic_relabel` are output-only calibrators: they see the confidence score and correctness labels, but not evidence condition.
- `condition_oracle_logistic`, `staged_oracle_logistic`, and `condition_only_oracle` are upper-bound condition-aware controls: they are allowed to use condition labels or the staged topicality/answer-support/veracity codes.
- If output-only methods reduce ECE while leaving `rho_veracity` near zero, but condition-aware controls can introduce a veracity component, H3 should be framed as an output-channel failure rather than a failure of correctness calibration alone.