# Validity Robustness Part A

Unit: question x condition cell mean. Confidence intervals use by-question cluster bootstrap.

## A1 Staged Logit Rho

| model | confidence | rho_veracity | rho_answer_support | rho_topicality | rho_v/rho_as | rho_v/rho_top | n_questions | n_cells |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen7 | conf_ptrue | 0.0039 | 0.1849 | 0.4349 | 0.0209 | 0.0089 | 1500 | 7500 |
| Qwen7 | conf_verbalized | 0.0016 | 0.1319 | 0.4939 | 0.0119 | 0.0032 | 1500 | 7500 |
| Qwen7 | conf_seqlik | 0.0111 | 0.2954 | 0.1892 | 0.0377 | 0.0588 | 1500 | 7500 |
| Qwen14 | conf_ptrue | 0.0007 | 0.0619 | 0.0006 | 0.0106 | 1.1176 | 1500 | 7500 |
| Qwen14 | conf_verbalized | 0.0011 | 0.1334 | 0.3790 | 0.0080 | 0.0028 | 1500 | 7500 |
| Qwen14 | conf_seqlik | 0.0067 | 0.2549 | 0.2308 | 0.0262 | 0.0290 | 1500 | 7500 |

## A2 Sup vs Mis Paired

| model | confidence | delta_conf(mis-sup) | delta_acc(mis-sup) | gap | P(conf_sup>conf_mis) | separable AUROC | rank-biserial | dz |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen7 | conf_ptrue | 0.0875 | -0.6808 | 0.7683 | 0.5957 | 0.5957 | 0.1913 | 0.2232 |
| Qwen7 | conf_verbalized | 0.0798 | -0.6808 | 0.7606 | 0.4540 | 0.5460 | -0.0920 | 0.2794 |
| Qwen7 | conf_seqlik | 0.0180 | -0.6808 | 0.6988 | 0.3620 | 0.6380 | -0.2760 | 0.2213 |
| Qwen14 | conf_ptrue | -0.0255 | -0.6459 | 0.6203 | 0.5257 | 0.5257 | 0.0513 | -0.0516 |
| Qwen14 | conf_verbalized | 0.0284 | -0.6459 | 0.6743 | 0.4560 | 0.5440 | -0.0880 | 0.1138 |
| Qwen14 | conf_seqlik | 0.0195 | -0.6459 | 0.6653 | 0.4747 | 0.5253 | -0.0507 | 0.1794 |

## A6 Cross-Axis Absolute Sensitivity

| model | confidence | axis | abs standardized contrast | separable AUROC | directional AUROC | n_questions | n_cells |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen7 | conf_ptrue | topicality | 0.7915 | 0.8707 | 0.8707 | 1500 | 7500 |
| Qwen7 | conf_ptrue | answer_support | 1.0917 | 0.8753 | 0.8753 | 1500 | 7500 |
| Qwen7 | conf_ptrue | veracity | 0.2232 | 0.5957 | 0.5957 | 1500 | 7500 |
| Qwen7 | conf_verbalized | topicality | 1.2413 | 0.8577 | 0.8577 | 1500 | 7500 |
| Qwen7 | conf_verbalized | answer_support | 0.9224 | 0.8830 | 0.8830 | 1500 | 7500 |
| Qwen7 | conf_verbalized | veracity | 0.2794 | 0.5460 | 0.4540 | 1500 | 7500 |
| Qwen7 | conf_seqlik | topicality | 0.0694 | 0.5260 | 0.5260 | 1500 | 7500 |
| Qwen7 | conf_seqlik | answer_support | 0.8175 | 0.8267 | 0.8267 | 1500 | 7500 |
| Qwen7 | conf_seqlik | veracity | 0.2213 | 0.6380 | 0.3620 | 1500 | 7500 |
| Qwen14 | conf_ptrue | topicality | 0.3113 | 0.5387 | 0.4613 | 1500 | 7500 |
| Qwen14 | conf_ptrue | answer_support | 0.4359 | 0.6927 | 0.6927 | 1500 | 7500 |
| Qwen14 | conf_ptrue | veracity | 0.0516 | 0.5257 | 0.5257 | 1500 | 7500 |
| Qwen14 | conf_verbalized | topicality | 0.9994 | 0.8173 | 0.8173 | 1500 | 7500 |
| Qwen14 | conf_verbalized | answer_support | 0.8030 | 0.8710 | 0.8710 | 1500 | 7500 |
| Qwen14 | conf_verbalized | veracity | 0.1138 | 0.5440 | 0.4560 | 1500 | 7500 |
| Qwen14 | conf_seqlik | topicality | 0.0579 | 0.5800 | 0.5800 | 1500 | 7500 |
| Qwen14 | conf_seqlik | answer_support | 0.7818 | 0.8153 | 0.8153 | 1500 | 7500 |
| Qwen14 | conf_seqlik | veracity | 0.1794 | 0.5253 | 0.4747 | 1500 | 7500 |