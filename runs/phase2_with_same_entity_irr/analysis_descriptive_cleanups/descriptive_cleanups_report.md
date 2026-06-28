# Descriptive Cleanups

Primary confidence: `conf_ptrue`

## Answer-Support Surface Control

| confidence | outcome | comparison | answer_bearing coef | p | ctx_len coef | ctx_entity coef | n | r2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| conf_ptrue | confidence | answer_bearing_vs_same_entity_irr_control_len_entities | 0.4811 | 0.0000 | 0.0009 | -0.0055 | 4500 | 0.3297 |
| conf_ptrue | confidence | answer_bearing_plus_mis_indicator_control_len_entities | 0.4554 | 0.0000 | 0.0003 | -0.0017 | 4500 | 0.3333 |
| conf_ptrue | accuracy | answer_bearing_vs_same_entity_irr_control_len_entities | 0.3213 | 0.0000 | -0.0049 | 0.0333 | 4500 | 0.1781 |
| conf_ptrue | accuracy | answer_bearing_plus_mis_indicator_control_len_entities | 0.5595 | 0.0000 | 0.0002 | -0.0023 | 4500 | 0.4494 |

## Staged Rho Bootstrap CI

| metric | REML point | moment point | 95% CI low | 95% CI high | bootstrap n |
|---|---:|---:|---:|---:|---:|
| rho_topicality | 0.4116 | 0.4117 | 0.3984 | 0.4273 | 400 |
| rho_answer_support | 0.1877 | 0.1877 | 0.1725 | 0.2029 | 400 |
| rho_veracity | 0.0040 | 0.0040 | 0.0024 | 0.0058 | 400 |
| rho_P | 0.0219 | 0.1091 | 0.1005 | 0.1164 | 400 |
| rho_PE | 0.3423 | 0.2615 | 0.2489 | 0.2736 | 400 |
| rho_eps | 0.0325 | 0.0260 | 0.0231 | 0.0290 | 400 |