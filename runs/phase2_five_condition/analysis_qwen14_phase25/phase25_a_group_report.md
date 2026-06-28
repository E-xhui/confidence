# Phase 2.5 A-Group Analysis

Primary confidence: `conf_ptrue` (P(True))

## A1 Closed-Known Gate

| subset | condition | acc | conf | Δacc | Δconf | FG | n |
|---|---|---:|---:|---:|---:|---:|---:|
| closed_known | closed | 0.9427 | 0.6613 | 0.0000 | 0.0000 | 0.0000 | 206 |
| closed_known | irr | 0.2786 | 0.6011 | -0.6641 | -0.0602 | 0.6039 | 206 |
| closed_known | mis | 0.0505 | 0.6433 | -0.8922 | -0.0180 | 0.8743 | 206 |
| closed_known | same_entity_irr | 0.6369 | 0.6992 | -0.3058 | 0.0380 | 0.3438 | 206 |
| closed_known | sup | 0.8068 | 0.8367 | -0.1359 | 0.1754 | 0.3113 | 206 |
| closed_unknown | closed | 0.0116 | 0.3070 | 0.0000 | 0.0000 | 0.0000 | 1294 |
| closed_unknown | irr | 0.0026 | 0.8325 | -0.0090 | 0.5255 | 0.5345 | 1294 |
| closed_unknown | mis | 0.0008 | 0.8595 | -0.0108 | 0.5525 | 0.5633 | 1294 |
| closed_unknown | same_entity_irr | 0.1032 | 0.6189 | 0.0917 | 0.3120 | 0.2203 | 1294 |
| closed_unknown | sup | 0.6291 | 0.8583 | 0.6175 | 0.5513 | -0.0662 | 1294 |

Flip diagnostics:

- closed_known n=206; flip_to_mostly_wrong=0.9515; mean_mis_error_rate=0.9495; mean_conf_flipped=0.6506.

## A2 Mis Source Split

| source | group | acc | conf | Δacc | Δconf | FG | n |
|---|---|---:|---:|---:|---:|---:|---:|
| counter | conflicting_counter_memory | 0.0332 | 0.6285 | -0.4450 | 0.1234 | 0.5684 | 289 |
| parametric | confirming_wrong_prior | 0.0015 | 0.8778 | -0.0571 | 0.5579 | 0.6150 | 1211 |

## A3 Popularity Axis

| popularity | condition | acc | conf | n |
|---|---|---:|---:|---:|
| high | closed | 0.2428 | 0.4110 | 500 |
| high | irr | 0.0864 | 0.7161 | 500 |
| high | mis | 0.0112 | 0.7403 | 500 |
| high | same_entity_irr | 0.2300 | 0.5918 | 500 |
| high | sup | 0.6068 | 0.8388 | 500 |
| low | closed | 0.0824 | 0.3271 | 500 |
| low | irr | 0.0124 | 0.8609 | 500 |
| low | mis | 0.0020 | 0.8979 | 500 |
| low | same_entity_irr | 0.1480 | 0.6994 | 500 |
| low | sup | 0.6544 | 0.8614 | 500 |
| mid | closed | 0.0932 | 0.3287 | 500 |
| mid | irr | 0.0228 | 0.8252 | 500 |
| mid | mis | 0.0096 | 0.8512 | 500 |
| mid | same_entity_irr | 0.1516 | 0.5987 | 500 |
| mid | sup | 0.6992 | 0.8657 | 500 |

## A4 Surface Confounding

| confidence | mis_vs_sup_coef_controlled | p | n | r2 |
|---|---:|---:|---:|---:|
| conf_ptrue | 0.0007 | 0.9650 | 3000 | 0.0037 |
| conf_verbalized | 0.0377 | 0.0000 | 3000 | 0.0079 |
| conf_seqlik | 0.0253 | 0.0000 | 3000 | 0.0174 |

## A6 Rho Subgroups

| subgroup | rho_P | rho_E | rho_PE | rho_eps | estimator | n |
|---|---:|---:|---:|---:|---|---:|
| closed_known | 0.0258 | 0.0681 | 0.8476 | 0.0586 | REML MixedLM | 206 |
| closed_unknown | 0.0544 | 0.0056 | 0.8223 | 0.1177 | REML MixedLM | 1294 |
| mis_source_counter | 0.0081 | 0.0343 | 0.8842 | 0.0733 | REML MixedLM | 289 |
| mis_source_parametric | 0.0519 | 0.0098 | 0.8182 | 0.1200 | REML MixedLM | 1211 |

## A7 IRR Behavior

| behavior | sample_rate | question_rate | acc | samples | questions |
|---|---:|---:|---:|---:|---:|
| abstain | 0.0031 | 0.0013 | 0.0000 | 23 | 2 |
| answer_from_irrelevant_context | 0.0188 | 0.0180 | 0.0000 | 141 | 27 |
| fallback_to_prior_or_gold | 0.0623 | 0.0627 | 0.6510 | 467 | 94 |
| other_or_hallucinated | 0.9159 | 0.9180 | 0.0000 | 6869 | 1377 |

## Output Files

- `a1_closed_strata.csv`
- `a1_flip_diagnostics.csv`
- `a2_mis_source_split.csv`
- `a3_popularity_axis.csv`
- `a4_surface_within_condition.csv`
- `a4_surface_mis_sup_controlled.csv`
- `a6_rho_subgroups.csv`
- `a7_irr_behavior.csv`