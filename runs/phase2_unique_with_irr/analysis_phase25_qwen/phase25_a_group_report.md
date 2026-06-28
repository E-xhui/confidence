# Phase 2.5 A-Group Analysis

Primary confidence: `conf_ptrue` (P(True))

## A1 Closed-Known Gate

| subset | condition | acc | conf | Δacc | Δconf | FG | n |
|---|---|---:|---:|---:|---:|---:|---:|
| closed_known | closed | 0.9417 | 0.7477 | 0.0000 | 0.0000 | 0.0000 | 144 |
| closed_known | irr | 0.2111 | 0.1144 | -0.7306 | -0.6333 | 0.0972 | 144 |
| closed_known | mis | 0.0403 | 0.8984 | -0.9014 | 0.1507 | 1.0521 | 144 |
| closed_known | sup | 0.8181 | 0.8948 | -0.1236 | 0.1472 | 0.2708 | 144 |
| closed_unknown | closed | 0.0087 | 0.4427 | 0.0000 | 0.0000 | 0.0000 | 1356 |
| closed_unknown | irr | 0.0006 | 0.0385 | -0.0081 | -0.4042 | -0.3961 | 1356 |
| closed_unknown | mis | 0.0007 | 0.9559 | -0.0080 | 0.5132 | 0.5211 | 1356 |
| closed_unknown | sup | 0.6712 | 0.8595 | 0.6625 | 0.4168 | -0.2457 | 1356 |

Flip diagnostics:

- closed_known n=144; flip_to_mostly_wrong=0.9583; mean_mis_error_rate=0.9597; mean_conf_flipped=0.9011.

## A2 Mis Source Split

| source | group | acc | conf | Δacc | Δconf | FG | n |
|---|---|---:|---:|---:|---:|---:|---:|
| counter | conflicting_counter_memory | 0.0235 | 0.9106 | -0.3481 | 0.3168 | 0.6649 | 289 |
| parametric | confirming_wrong_prior | 0.0000 | 0.9599 | -0.0330 | 0.5169 | 0.5499 | 1211 |

## A3 Popularity Axis

| popularity | condition | acc | conf | n |
|---|---|---:|---:|---:|
| high | closed | 0.1584 | 0.4880 | 500 |
| high | irr | 0.0456 | 0.0502 | 500 |
| high | mis | 0.0056 | 0.9272 | 500 |
| high | sup | 0.6276 | 0.8590 | 500 |
| low | closed | 0.0732 | 0.4463 | 500 |
| low | irr | 0.0080 | 0.0405 | 500 |
| low | mis | 0.0020 | 0.9626 | 500 |
| low | sup | 0.7096 | 0.8458 | 500 |
| mid | closed | 0.0632 | 0.4818 | 500 |
| mid | irr | 0.0088 | 0.0466 | 500 |
| mid | mis | 0.0060 | 0.9614 | 500 |
| mid | sup | 0.7188 | 0.8839 | 500 |

## A4 Surface Confounding

| confidence | mis_vs_sup_coef_controlled | p | n | r2 |
|---|---:|---:|---:|---:|
| conf_ptrue | 0.0857 | 0.0000 | 3000 | 0.0249 |
| conf_verbalized | 0.0716 | 0.0000 | 3000 | 0.0385 |
| conf_seqlik | 0.0216 | 0.0000 | 3000 | 0.0212 |

## A6 Rho Subgroups

| subgroup | rho_P | rho_E | rho_PE | rho_eps | estimator | n |
|---|---:|---:|---:|---:|---|---:|
| closed_known | 0.0064 | 0.6343 | 0.3471 | 0.0123 | REML MixedLM | 144 |
| closed_unknown | 0.0045 | 0.7532 | 0.2307 | 0.0116 | REML MixedLM | 1356 |
| mis_source_counter | 0.0065 | 0.6633 | 0.3190 | 0.0112 | REML MixedLM | 289 |
| mis_source_parametric | 0.0038 | 0.7601 | 0.2244 | 0.0118 | REML MixedLM | 1211 |

## A7 IRR Behavior

| behavior | sample_rate | question_rate | acc | samples | questions |
|---|---:|---:|---:|---:|---:|
| abstain | 0.0055 | 0.0013 | 0.0000 | 41 | 2 |
| answer_from_irrelevant_context | 0.1440 | 0.1413 | 0.0000 | 1080 | 212 |
| fallback_to_prior_or_gold | 0.0303 | 0.0307 | 0.6872 | 227 | 46 |
| other_or_hallucinated | 0.8203 | 0.8267 | 0.0000 | 6152 | 1240 |

## Output Files

- `a1_closed_strata.csv`
- `a1_flip_diagnostics.csv`
- `a2_mis_source_split.csv`
- `a3_popularity_axis.csv`
- `a4_surface_within_condition.csv`
- `a4_surface_mis_sup_controlled.csv`
- `a6_rho_subgroups.csv`
- `a7_irr_behavior.csv`