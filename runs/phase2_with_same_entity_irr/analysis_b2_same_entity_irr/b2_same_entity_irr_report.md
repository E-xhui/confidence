# B2 Same-Entity IRR Analysis

Primary confidence: `conf_ptrue`

## Condition Summary

| condition | acc | conf | delta_acc | delta_conf | FG | n |
|---|---:|---:|---:|---:|---:|---:|
| closed | 0.0983 | 0.4720 | 0.0000 | 0.0000 | 0.0000 | 1500 |
| irr | 0.0208 | 0.0458 | -0.0775 | -0.4262 | -0.3488 | 1500 |
| same_entity_irr | 0.1307 | 0.4045 | 0.0324 | -0.0675 | -0.0999 | 1500 |
| sup | 0.6853 | 0.8629 | 0.5871 | 0.3909 | -0.1962 | 1500 |
| mis | 0.0045 | 0.9504 | -0.0937 | 0.4784 | 0.5721 | 1500 |

## Key Contrasts

| contrast | delta_conf | delta_acc | n |
|---|---:|---:|---:|
| same_entity_irr_minus_irr | 0.3587 | 0.1099 | 1500 |
| same_entity_irr_minus_closed | -0.0675 | 0.0324 | 1500 |
| same_entity_irr_minus_sup | -0.4584 | -0.5547 | 1500 |
| same_entity_irr_minus_mis | -0.5459 | 0.1261 | 1500 |
| mis_minus_sup | 0.0875 | -0.6808 | 1500 |

## Staged Rho

| confidence | rho_topicality | rho_answer_support | rho_veracity | rho_P | rho_PE | rho_eps |
|---|---:|---:|---:|---:|---:|---:|
| conf_ptrue | 0.4116 | 0.1877 | 0.0040 | 0.0219 | 0.3423 | 0.0325 |
| conf_verbalized | 0.4603 | 0.1388 | 0.0017 | 0.0155 | 0.3373 | 0.0464 |
| conf_seqlik | 0.1865 | 0.2944 | 0.0111 | 0.0285 | 0.4710 | 0.0086 |