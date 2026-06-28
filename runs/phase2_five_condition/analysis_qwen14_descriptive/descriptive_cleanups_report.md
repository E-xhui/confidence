# Descriptive Cleanups

Primary confidence: `conf_ptrue`

## Answer-Support Surface Control

| confidence | outcome | comparison | answer_bearing coef | p | ctx_len coef | ctx_entity coef | n | r2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| conf_ptrue | confidence | answer_bearing_vs_same_entity_irr_control_len_entities | 0.2347 | 0.0000 | -0.0007 | 0.0028 | 4500 | 0.0675 |
| conf_ptrue | confidence | answer_bearing_plus_mis_indicator_control_len_entities | 0.2365 | 0.0000 | -0.0007 | 0.0026 | 4500 | 0.0676 |
| conf_ptrue | accuracy | answer_bearing_vs_same_entity_irr_control_len_entities | 0.2462 | 0.0000 | -0.0044 | 0.0312 | 4500 | 0.1337 |
| conf_ptrue | accuracy | answer_bearing_plus_mis_indicator_control_len_entities | 0.4746 | 0.0000 | 0.0005 | -0.0030 | 4500 | 0.3801 |

## Staged Rho Bootstrap CI

| metric | REML point | moment point | 95% CI low | 95% CI high | bootstrap n |
|---|---:|---:|---:|---:|---:|
| rho_topicality | 0.0000 | 0.0000 | 0.0000 | 0.0008 | 400 |
| rho_answer_support | 0.0554 | 0.0555 | 0.0449 | 0.0662 | 400 |
| rho_veracity | 0.0005 | 0.0005 | 0.0000 | 0.0020 | 400 |
| rho_P | 0.0359 | 0.2397 | 0.2254 | 0.2551 | 400 |
| rho_PE | 0.7920 | 0.6113 | 0.5949 | 0.6274 | 400 |
| rho_eps | 0.1161 | 0.0929 | 0.0860 | 0.0998 | 400 |