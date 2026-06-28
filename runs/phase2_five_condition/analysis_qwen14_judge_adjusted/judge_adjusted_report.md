# Judge-Adjusted EM Boundary Report

Judge corrections are applied only to selected EM-boundary candidates.

## Judge Summary

- judge_rows: `1940`
- judge_positive_rate: `0.32989690721649484`
- corrected_sample_count: `640`
- original_correct_mean: `0.20352`
- adjusted_correct_mean: `0.22058666666666665`

## Adjusted P(True) Condition Summary

| condition | acc | conf | delta_acc | delta_conf | FG | n |
|---|---:|---:|---:|---:|---:|---:|
| closed | 0.1452 | 0.3556 | 0.0000 | 0.0000 | 0.0000 | 1500 |
| irr | 0.0459 | 0.8007 | -0.0993 | 0.4451 | 0.5444 | 1500 |
| same_entity_irr | 0.2101 | 0.6300 | 0.0649 | 0.2743 | 0.2094 | 1500 |
| sup | 0.6839 | 0.8553 | 0.5387 | 0.4997 | -0.0390 | 1500 |
| mis | 0.0179 | 0.8298 | -0.1273 | 0.4742 | 0.6015 | 1500 |

## Adjusted Staged Rho

| metric | value |
|---|---:|
| rho_topicality | 0.0000 |
| rho_answer_support | 0.0554 |
| rho_veracity | 0.0005 |
| rho_P | 0.0359 |
| rho_PE | 0.7920 |
| rho_eps | 0.1161 |