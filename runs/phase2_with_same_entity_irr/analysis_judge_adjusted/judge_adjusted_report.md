# Judge-Adjusted EM Boundary Report

Judge corrections are applied only to selected EM-boundary candidates.

## Judge Summary

- judge_rows: `1636`
- judge_positive_rate: `0.2017114914425428`
- corrected_sample_count: `330`
- original_correct_mean: `0.18792`
- adjusted_correct_mean: `0.19672`

## Adjusted P(True) Condition Summary

| condition | acc | conf | delta_acc | delta_conf | FG | n |
|---|---:|---:|---:|---:|---:|---:|
| closed | 0.0999 | 0.4720 | 0.0000 | 0.0000 | 0.0000 | 1500 |
| irr | 0.0236 | 0.0458 | -0.0763 | -0.4262 | -0.3500 | 1500 |
| same_entity_irr | 0.1452 | 0.4045 | 0.0453 | -0.0675 | -0.1128 | 1500 |
| sup | 0.7016 | 0.8629 | 0.6017 | 0.3909 | -0.2108 | 1500 |
| mis | 0.0133 | 0.9504 | -0.0865 | 0.4784 | 0.5649 | 1500 |

## Adjusted Staged Rho

| metric | value |
|---|---:|
| rho_topicality | 0.4116 |
| rho_answer_support | 0.1877 |
| rho_veracity | 0.0040 |
| rho_P | 0.0219 |
| rho_PE | 0.3423 |
| rho_eps | 0.0325 |