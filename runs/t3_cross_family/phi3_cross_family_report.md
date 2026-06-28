# T3 Cross-Family Phi-3 300-Question Report

Model: `microsoft/Phi-3-mini-4k-instruct` loaded from `/data/home/yixh/models/Phi-3-mini-4k-instruct` with native transformers (`--no-trust-remote-code`).

Prediction rows: 7500. Unit for analysis: question x condition cell mean. Confidence channels: `conf_ptrue`, `conf_verbalized`.

## Staged Rho

| confidence | rho_veracity | rho_answer_support | rho_topicality | rho_v/rho_as | n_questions |
|---|---:|---:|---:|---:|---:|
| conf_ptrue | 0.0260 | 0.1307 | 0.4991 | 0.1990 | 300 |
| conf_verbalized | 0.0006 | 0.0625 | 0.3631 | 0.0093 | 300 |

## Sup-vs-Mis Paired

| confidence | delta_conf(mis-sup) | delta_acc(mis-sup) | gap | directional AUC P(conf_sup>conf_mis) | separable AUC |
|---|---:|---:|---:|---:|---:|
| conf_ptrue | 0.1450 | -0.6487 | 0.7937 | 0.2900 | 0.7100 |
| conf_verbalized | 0.0110 | -0.6487 | 0.6597 | 0.4783 | 0.5217 |

## FG(mis)

| confidence | acc(mis) | conf(mis) | delta_acc vs closed | delta_conf vs closed | FG(mis) |
|---|---:|---:|---:|---:|---:|
| conf_ptrue | 0.0020 | 0.9489 | -0.0893 | 0.5537 | 0.6430 |
| conf_verbalized | 0.0020 | 0.9951 | -0.0893 | 0.0492 | 0.1385 |

## Closed Accuracy By Popularity

| popularity_bin | closed_acc | closed_conf_ptrue | closed_conf_verbalized | n_questions |
|---|---:|---:|---:|---:|
| high | 0.1400 | 0.5005 | 0.9545 | 100 |
| low | 0.0460 | 0.3260 | 0.9310 | 100 |
| mid | 0.0880 | 0.3592 | 0.9524 | 100 |

## 判定

Phi-3 cross-family subset is intended as directional robustness, not a replacement for the full Qwen mainline. Interpret against the fixed claim: confidence may carry weak nonzero sup-vs-mis separability, but veracity should explain a much smaller share of confidence variation than answer-bearing support, while misleading evidence should still produce positive high-confidence error gaps.

Final readout:

- `conf_ptrue`: `rho_veracity=0.0260`, not exactly zero and larger than Qwen, but still much smaller than `rho_answer_support=0.1307` and `rho_topicality=0.4991`; `rho_v/rho_as=0.1990`.
- `conf_verbalized`: `rho_veracity=0.0006`, effectively near zero; `rho_v/rho_as=0.0093`.
- `FG(mis)` is positive for both channels: `0.6430` for PTrue and `0.1385` for verbalized.
- The PTrue sup-vs-mis direction is perverse rather than truth-aligned: `delta_conf(mis-sup)=+0.1450` while `delta_acc(mis-sup)=-0.6487`; separable AUC is `0.7100` because misleading contexts are often more confident.
- Closed-book accuracy increases by popularity bin: low `0.0460`, mid `0.0880`, high `0.1400`, so this 300-question subset retains a prior/item-strength gradient.

Conclusion: Phi-3 supports the cross-family direction for the conservative claim: misleading evidence still induces large high-confidence errors, and truth/veracity accounts for a smaller confidence share than answer-bearing support/topicality. It does not support the stronger wording that PTrue `rho_veracity` is exactly zero across families; the safer wording is weak/nonzero, often misdirected veracity separability.
