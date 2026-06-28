# B2 Same-Entity IRR Analysis

Primary confidence: `conf_ptrue`

## Condition Summary

| condition | acc | conf | delta_acc | delta_conf | FG | n |
|---|---:|---:|---:|---:|---:|---:|
| closed | 0.1395 | 0.3556 | 0.0000 | 0.0000 | 0.0000 | 1500 |
| irr | 0.0405 | 0.8007 | -0.0989 | 0.4451 | 0.5440 | 1500 |
| same_entity_irr | 0.1765 | 0.6300 | 0.0371 | 0.2743 | 0.2373 | 1500 |
| sup | 0.6535 | 0.8553 | 0.5140 | 0.4997 | -0.0143 | 1500 |
| mis | 0.0076 | 0.8298 | -0.1319 | 0.4742 | 0.6060 | 1500 |

## Key Contrasts

| contrast | delta_conf | delta_acc | n |
|---|---:|---:|---:|
| same_entity_irr_minus_irr | -0.1708 | 0.1360 | 1500 |
| same_entity_irr_minus_closed | 0.2743 | 0.0371 | 1500 |
| same_entity_irr_minus_sup | -0.2254 | -0.4769 | 1500 |
| same_entity_irr_minus_mis | -0.1998 | 0.1689 | 1500 |
| mis_minus_sup | -0.0255 | -0.6459 | 1500 |

## Staged Rho

| confidence | rho_topicality | rho_answer_support | rho_veracity | rho_P | rho_PE | rho_eps |
|---|---:|---:|---:|---:|---:|---:|
| conf_ptrue | 0.0000 | 0.0554 | 0.0005 | 0.0359 | 0.7920 | 0.1161 |
| conf_verbalized | 0.3664 | 0.1110 | 0.0009 | 0.0051 | 0.3810 | 0.1356 |
| conf_seqlik | 0.2297 | 0.2538 | 0.0066 | 0.0519 | 0.4511 | 0.0068 |