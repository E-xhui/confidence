# Cross-Model Robustness Summary

Primary confidence: `conf_ptrue`

Qwen2.5-14B is an additional cross-size robustness check. Llama-3.1-8B remains blocked because gated weights/token are unavailable on the server.

## Prediction Files

| model | rows | path |
|---|---:|---|
| Qwen2.5-7B | 37500 | `runs/phase2_with_same_entity_irr/predictions_qwen_full.jsonl` |
| Qwen2.5-14B | 37500 | `runs/phase2_five_condition/predictions_qwen14_full.jsonl` |

## Condition Summary

| model | condition | acc | P(True) | delta_acc | delta_conf | FG | n_questions |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | closed | 0.0983 | 0.4720 | 0.0000 | 0.0000 | 0.0000 | 1500 |
| Qwen2.5-7B | irr | 0.0208 | 0.0458 | -0.0775 | -0.4262 | -0.3488 | 1500 |
| Qwen2.5-7B | same_entity_irr | 0.1307 | 0.4045 | 0.0324 | -0.0675 | -0.0999 | 1500 |
| Qwen2.5-7B | sup | 0.6853 | 0.8629 | 0.5871 | 0.3909 | -0.1962 | 1500 |
| Qwen2.5-7B | mis | 0.0045 | 0.9504 | -0.0937 | 0.4784 | 0.5721 | 1500 |
| Qwen2.5-14B | closed | 0.1395 | 0.3556 | 0.0000 | 0.0000 | 0.0000 | 1500 |
| Qwen2.5-14B | irr | 0.0405 | 0.8007 | -0.0989 | 0.4451 | 0.5440 | 1500 |
| Qwen2.5-14B | same_entity_irr | 0.1765 | 0.6300 | 0.0371 | 0.2743 | 0.2373 | 1500 |
| Qwen2.5-14B | sup | 0.6535 | 0.8553 | 0.5140 | 0.4997 | -0.0143 | 1500 |
| Qwen2.5-14B | mis | 0.0076 | 0.8298 | -0.1319 | 0.4742 | 0.6060 | 1500 |

## Staged Rho

| model | rho_topicality | rho_answer_support | rho_veracity | rho_P | rho_PE | rho_eps |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 0.4116 | 0.1877 | 0.0040 | 0.0219 | 0.3423 | 0.0325 |
| Qwen2.5-14B | 0.0000 | 0.0554 | 0.0005 | 0.0359 | 0.7920 | 0.1161 |

## H3 Calibration

| model | method | ECE | FG(mis) | rho_topicality | rho_answer_support | rho_veracity |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B | identity | 0.3949 | 0.5651 | 0.4092 | 0.1985 | 0.0038 |
| Qwen2.5-7B | platt_logistic | 0.0176 | 0.2756 | 0.4239 | 0.2054 | 0.0032 |
| Qwen2.5-7B | isotonic_relabel | 0.0019 | 0.2997 | 0.4025 | 0.2281 | 0.0022 |
| Qwen2.5-14B | identity | 0.5492 | 0.5864 | 0.0001 | 0.0534 | 0.0013 |
| Qwen2.5-14B | platt_logistic | 0.0148 | 0.2040 | 0.0008 | 0.0618 | 0.0014 |
| Qwen2.5-14B | isotonic_relabel | 0.0068 | 0.2363 | 0.0578 | 0.0892 | 0.0008 |
