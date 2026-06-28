# Rho Reparameterized Analysis

Primary confidence: `conf_ptrue`

Coding: `relevance_c={sup:1/3, mis:1/3, irr:-2/3}`; `veracity={sup:0.5, mis:-0.5, irr:0}`.

## Main Table

| subset | confidence | rho_relevance | rho_veracity | rho_P | rho_PE | rho_eps | beta_relevance | beta_veracity | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | conf_ptrue | 0.7357 | 0.0052 | 0.0041 | 0.2433 | 0.0117 | 15.6587 | -1.5225 | 1500 |
| all | conf_verbalized | 0.6838 | 0.0020 | 0.0085 | 0.2782 | 0.0276 | 14.0775 | -0.8734 | 1500 |
| all | conf_seqlik | 0.4599 | 0.0147 | 0.0853 | 0.4330 | 0.0071 | 4.8811 | -1.0082 | 1500 |
| closed_known | conf_ptrue | 0.6342 | 0.0001 | 0.0064 | 0.3471 | 0.0123 | 14.2292 | 0.2307 | 144 |
| closed_known | conf_verbalized | 0.5805 | 0.0022 | 0.0081 | 0.3912 | 0.0180 | 12.8150 | 0.9143 | 144 |
| closed_known | conf_seqlik | 0.3784 | 0.0044 | 0.1036 | 0.5061 | 0.0075 | 4.1937 | -0.5238 | 144 |
| sup_reader_gold_pass | conf_ptrue | 0.8080 | 0.0000 | 0.0038 | 0.1796 | 0.0086 | 16.2347 | 0.0325 | 1220 |
| sup_reader_gold_pass | conf_verbalized | 0.7478 | 0.0005 | 0.0033 | 0.2222 | 0.0261 | 14.6746 | 0.4314 | 1220 |
| sup_reader_gold_pass | conf_seqlik | 0.5158 | 0.0025 | 0.0915 | 0.3835 | 0.0067 | 5.1563 | -0.4120 | 1220 |
| closed_known_and_sup_reader_gold_pass | conf_ptrue | 0.6690 | 0.0049 | 0.0087 | 0.3068 | 0.0105 | 14.4372 | 1.4288 | 131 |
| closed_known_and_sup_reader_gold_pass | conf_verbalized | 0.5912 | 0.0086 | 0.0085 | 0.3714 | 0.0203 | 12.7494 | 1.7766 | 131 |
| closed_known_and_sup_reader_gold_pass | conf_seqlik | 0.3951 | 0.0003 | 0.1145 | 0.4829 | 0.0072 | 4.2831 | -0.1266 | 131 |

## Direct Sup-vs-Mis Check

`direct_rho_veracity_from_condition_main_effect` is the older two-condition `{sup, mis}` main-effect decomposition, included as a sanity check.

| subset | confidence | direct rho veracity |
|---|---|---:|
| all | conf_ptrue | 0.0243 |
| all | conf_verbalized | 0.0091 |
| all | conf_seqlik | 0.0319 |
| closed_known | conf_ptrue | 0.0005 |
| closed_known | conf_verbalized | 0.0112 |
| closed_known | conf_seqlik | 0.0081 |
| sup_reader_gold_pass | conf_ptrue | 0.0000 |
| sup_reader_gold_pass | conf_verbalized | 0.0034 |
| sup_reader_gold_pass | conf_seqlik | 0.0062 |
| closed_known_and_sup_reader_gold_pass | conf_ptrue | 0.0253 |
| closed_known_and_sup_reader_gold_pass | conf_verbalized | 0.0529 |
| closed_known_and_sup_reader_gold_pass | conf_seqlik | 0.0005 |