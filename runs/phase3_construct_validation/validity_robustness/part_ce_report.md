# Validity Robustness Parts C and E

## C1 Condition-Level Anchor Table

| model | confidence | condition | acc | conf | n_questions | n_cells |
|---|---|---|---:|---:|---:|---:|
| Qwen7 | conf_ptrue | mis | 0.0045 | 0.9504 | 1500 | 1500 |
| Qwen7 | conf_ptrue | same_entity_irr | 0.1307 | 0.4045 | 1500 | 1500 |
| Qwen7 | conf_ptrue | sup | 0.6853 | 0.8629 | 1500 | 1500 |
| Qwen7 | conf_verbalized | mis | 0.0045 | 0.9563 | 1500 | 1500 |
| Qwen7 | conf_verbalized | same_entity_irr | 0.1307 | 0.5875 | 1500 | 1500 |
| Qwen7 | conf_verbalized | sup | 0.6853 | 0.8765 | 1500 | 1500 |
| Qwen7 | conf_seqlik | mis | 0.0045 | 0.9835 | 1500 | 1500 |
| Qwen7 | conf_seqlik | same_entity_irr | 0.1307 | 0.8435 | 1500 | 1500 |
| Qwen7 | conf_seqlik | sup | 0.6853 | 0.9654 | 1500 | 1500 |
| Qwen14 | conf_ptrue | mis | 0.0076 | 0.8298 | 1500 | 1500 |
| Qwen14 | conf_ptrue | same_entity_irr | 0.1765 | 0.6300 | 1500 | 1500 |
| Qwen14 | conf_ptrue | sup | 0.6535 | 0.8553 | 1500 | 1500 |
| Qwen14 | conf_verbalized | mis | 0.0076 | 0.9504 | 1500 | 1500 |
| Qwen14 | conf_verbalized | same_entity_irr | 0.1765 | 0.7266 | 1500 | 1500 |
| Qwen14 | conf_verbalized | sup | 0.6535 | 0.9220 | 1500 | 1500 |
| Qwen14 | conf_seqlik | mis | 0.0076 | 0.9782 | 1500 | 1500 |
| Qwen14 | conf_seqlik | same_entity_irr | 0.1765 | 0.8405 | 1500 | 1500 |
| Qwen14 | conf_seqlik | sup | 0.6535 | 0.9587 | 1500 | 1500 |

## C2 Wording

sup and mis both contain answer-bearing spans, so the first-order answer-bearing effect is controlled; residual differences may still reflect answer identity/prior, which is inspected separately.

## C3 Staged-Coding Regression

| model | confidence | regression | scale | term | coef | CI low | CI high | max VIF | cond # | status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| Qwen7 | conf_ptrue | c3a | logit | answer_bearing | 8.6190 | 8.0999 | 9.1379 | 3.6670 | 3.8480 | primary |
| Qwen7 | conf_ptrue | c3a | logit | veracity | -1.7225 | -2.2654 | -1.1641 | 3.6670 | 3.8480 | primary |
| Qwen7 | conf_ptrue | c3a | raw | answer_bearing | 0.4934 | 0.4586 | 0.5264 | 3.6670 | 3.8480 | primary |
| Qwen7 | conf_ptrue | c3a | raw | veracity | -0.1001 | -0.1305 | -0.0701 | 3.6670 | 3.8480 | primary |
| Qwen7 | conf_ptrue | c3b | logit | topicality | 12.7769 | 12.5191 | 13.0365 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_ptrue | c3b | logit | answer_support | 8.7024 | 8.2376 | 9.1946 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_ptrue | c3b | logit | veracity | -1.7154 | -2.1578 | -1.2983 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_ptrue | c3b | raw | topicality | 0.7020 | 0.6882 | 0.7172 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_ptrue | c3b | raw | answer_support | 0.4977 | 0.4678 | 0.5268 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_ptrue | c3b | raw | veracity | -0.1023 | -0.1286 | -0.0773 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_verbalized | c3a | logit | answer_bearing | 6.4309 | 5.9748 | 6.8931 | 3.6670 | 3.8480 | primary |
| Qwen7 | conf_verbalized | c3a | logit | veracity | -0.9453 | -1.4426 | -0.4656 | 3.6670 | 3.8480 | primary |
| Qwen7 | conf_verbalized | c3a | raw | answer_bearing | 0.3252 | 0.3007 | 0.3509 | 3.6670 | 3.8480 | primary |
| Qwen7 | conf_verbalized | c3a | raw | veracity | -0.0884 | -0.1132 | -0.0638 | 3.6670 | 3.8480 | primary |
| Qwen7 | conf_verbalized | c3b | logit | topicality | 11.8166 | 11.5612 | 12.0709 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_verbalized | c3b | logit | answer_support | 6.6011 | 6.2342 | 7.0018 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_verbalized | c3b | logit | veracity | -0.9482 | -1.3814 | -0.5646 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_verbalized | c3b | raw | topicality | 0.7204 | 0.7044 | 0.7359 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_verbalized | c3b | raw | answer_support | 0.3404 | 0.3177 | 0.3631 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_verbalized | c3b | raw | veracity | -0.0889 | -0.1115 | -0.0697 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_seqlik | c3a | logit | answer_bearing | 4.4496 | 4.2123 | 4.7096 | 3.6670 | 3.8480 | primary |
| Qwen7 | conf_seqlik | c3a | logit | veracity | -1.2069 | -1.4359 | -0.9792 | 3.6670 | 3.8480 | primary |
| Qwen7 | conf_seqlik | c3a | raw | answer_bearing | 0.1250 | 0.1154 | 0.1357 | 3.6670 | 3.8480 | primary |
| Qwen7 | conf_seqlik | c3a | raw | veracity | -0.0232 | -0.0306 | -0.0157 | 3.6670 | 3.8480 | primary |
| Qwen7 | conf_seqlik | c3b | logit | topicality | 3.5083 | 3.3848 | 3.6373 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_seqlik | c3b | logit | answer_support | 4.5828 | 4.3740 | 4.8248 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_seqlik | c3b | logit | veracity | -1.2552 | -1.4568 | -1.0667 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_seqlik | c3b | raw | topicality | 0.0792 | 0.0728 | 0.0852 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_seqlik | c3b | raw | answer_support | 0.1290 | 0.1191 | 0.1392 | 2.9273 | 3.3245 | primary |
| Qwen7 | conf_seqlik | c3b | raw | veracity | -0.0251 | -0.0318 | -0.0186 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_ptrue | c3a | logit | answer_bearing | 3.7740 | 3.1689 | 4.3603 | 3.6670 | 3.8480 | primary |
| Qwen14 | conf_ptrue | c3a | logit | veracity | 0.2664 | -0.3696 | 0.9180 | 3.6670 | 3.8480 | primary |
| Qwen14 | conf_ptrue | c3a | raw | answer_bearing | 0.2039 | 0.1707 | 0.2368 | 3.6670 | 3.8480 | primary |
| Qwen14 | conf_ptrue | c3a | raw | veracity | 0.0172 | -0.0189 | 0.0534 | 3.6670 | 3.8480 | primary |
| Qwen14 | conf_ptrue | c3b | logit | topicality | 0.5230 | 0.1830 | 0.8720 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_ptrue | c3b | logit | answer_support | 3.9151 | 3.3451 | 4.4384 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_ptrue | c3b | logit | veracity | 0.1503 | -0.3467 | 0.6953 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_ptrue | c3b | raw | topicality | -0.0222 | -0.0432 | -0.0009 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_ptrue | c3b | raw | answer_support | 0.2078 | 0.1759 | 0.2404 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_ptrue | c3b | raw | veracity | 0.0145 | -0.0164 | 0.0433 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_verbalized | c3a | logit | answer_bearing | 5.3693 | 5.0138 | 5.6996 | 3.6670 | 3.8480 | primary |
| Qwen14 | conf_verbalized | c3a | logit | veracity | -0.7572 | -1.1562 | -0.3850 | 3.6670 | 3.8480 | primary |
| Qwen14 | conf_verbalized | c3a | raw | answer_bearing | 0.1996 | 0.1802 | 0.2187 | 3.6670 | 3.8480 | primary |
| Qwen14 | conf_verbalized | c3a | raw | veracity | -0.0439 | -0.0625 | -0.0235 | 3.6670 | 3.8480 | primary |
| Qwen14 | conf_verbalized | c3b | logit | topicality | 8.7470 | 8.4325 | 9.0626 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_verbalized | c3b | logit | answer_support | 5.3881 | 5.0412 | 5.7377 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_verbalized | c3b | logit | veracity | -1.0224 | -1.3816 | -0.6812 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_verbalized | c3b | raw | topicality | 0.5628 | 0.5437 | 0.5814 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_verbalized | c3b | raw | answer_support | 0.2002 | 0.1784 | 0.2193 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_verbalized | c3b | raw | veracity | -0.0503 | -0.0694 | -0.0312 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_seqlik | c3a | logit | answer_bearing | 4.3943 | 4.1413 | 4.6449 | 3.6670 | 3.8480 | primary |
| Qwen14 | conf_seqlik | c3a | logit | veracity | -1.2421 | -1.5140 | -0.9589 | 3.6670 | 3.8480 | primary |
| Qwen14 | conf_seqlik | c3a | raw | answer_bearing | 0.1238 | 0.1130 | 0.1339 | 3.6670 | 3.8480 | primary |
| Qwen14 | conf_seqlik | c3a | raw | veracity | -0.0296 | -0.0386 | -0.0199 | 3.6670 | 3.8480 | primary |
| Qwen14 | conf_seqlik | c3b | logit | topicality | 4.0531 | 3.9320 | 4.1666 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_seqlik | c3b | logit | answer_support | 4.3926 | 4.1728 | 4.6044 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_seqlik | c3b | logit | veracity | -1.1187 | -1.3336 | -0.9154 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_seqlik | c3b | raw | topicality | 0.0990 | 0.0939 | 0.1038 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_seqlik | c3b | raw | answer_support | 0.1251 | 0.1152 | 0.1342 | 2.9273 | 3.3245 | primary |
| Qwen14 | conf_seqlik | c3b | raw | veracity | -0.0268 | -0.0338 | -0.0194 | 2.9273 | 3.3245 | primary |

## E1 Item/Prior Axis Correlations

| model | confidence | proxy | method | corr | CI low | CI high | n_questions |
|---|---|---|---|---:|---:|---:|---:|
| Qwen7 | conf_ptrue | popularity_raw | pearson | -0.0319 | -0.1015 | 0.0324 | 1500 |
| Qwen7 | conf_ptrue | popularity_raw | spearman | -0.0708 | -0.1211 | -0.0161 | 1500 |
| Qwen7 | conf_ptrue | closed_book_correct_rate | pearson | 0.1599 | 0.1024 | 0.2179 | 1500 |
| Qwen7 | conf_ptrue | closed_book_correct_rate | spearman | 0.1413 | 0.0873 | 0.1917 | 1500 |
| Qwen7 | conf_ptrue | closed_book_mean_conf | pearson | 0.1051 | 0.0530 | 0.1533 | 1500 |
| Qwen7 | conf_ptrue | closed_book_mean_conf | spearman | 0.1058 | 0.0516 | 0.1597 | 1500 |
| Qwen7 | conf_ptrue | closed_book_acc | pearson | 0.1599 | 0.1076 | 0.2122 | 1500 |
| Qwen7 | conf_ptrue | closed_book_acc | spearman | 0.1413 | 0.0868 | 0.1943 | 1500 |
| Qwen7 | conf_verbalized | popularity_raw | pearson | -0.0187 | -0.0776 | 0.0323 | 1500 |
| Qwen7 | conf_verbalized | popularity_raw | spearman | -0.0614 | -0.1131 | -0.0062 | 1500 |
| Qwen7 | conf_verbalized | closed_book_correct_rate | pearson | 0.1574 | 0.0993 | 0.2174 | 1500 |
| Qwen7 | conf_verbalized | closed_book_correct_rate | spearman | 0.1390 | 0.0825 | 0.1921 | 1500 |
| Qwen7 | conf_verbalized | closed_book_mean_conf | pearson | 0.0732 | 0.0262 | 0.1249 | 1500 |
| Qwen7 | conf_verbalized | closed_book_mean_conf | spearman | 0.0925 | 0.0411 | 0.1424 | 1500 |
| Qwen7 | conf_verbalized | closed_book_acc | pearson | 0.1574 | 0.1018 | 0.2157 | 1500 |
| Qwen7 | conf_verbalized | closed_book_acc | spearman | 0.1390 | 0.0863 | 0.1924 | 1500 |
| Qwen7 | conf_seqlik | popularity_raw | pearson | -0.0028 | -0.0520 | 0.0432 | 1500 |
| Qwen7 | conf_seqlik | popularity_raw | spearman | 0.0320 | -0.0200 | 0.0826 | 1500 |
| Qwen7 | conf_seqlik | closed_book_correct_rate | pearson | 0.0095 | -0.0478 | 0.0674 | 1500 |
| Qwen7 | conf_seqlik | closed_book_correct_rate | spearman | -0.0141 | -0.0713 | 0.0392 | 1500 |
| Qwen7 | conf_seqlik | closed_book_mean_conf | pearson | -0.1006 | -0.1503 | -0.0483 | 1500 |
| Qwen7 | conf_seqlik | closed_book_mean_conf | spearman | -0.1063 | -0.1574 | -0.0545 | 1500 |
| Qwen7 | conf_seqlik | closed_book_acc | pearson | 0.0095 | -0.0505 | 0.0659 | 1500 |
| Qwen7 | conf_seqlik | closed_book_acc | spearman | -0.0141 | -0.0667 | 0.0422 | 1500 |
| Qwen14 | conf_ptrue | popularity_raw | pearson | -0.1501 | -0.2432 | -0.0842 | 1500 |
| Qwen14 | conf_ptrue | popularity_raw | spearman | -0.2543 | -0.3021 | -0.2069 | 1500 |
| Qwen14 | conf_ptrue | closed_book_correct_rate | pearson | -0.1714 | -0.2248 | -0.1173 | 1500 |
| Qwen14 | conf_ptrue | closed_book_correct_rate | spearman | -0.1456 | -0.1997 | -0.0953 | 1500 |
| Qwen14 | conf_ptrue | closed_book_mean_conf | pearson | 0.0119 | -0.0367 | 0.0591 | 1500 |
| Qwen14 | conf_ptrue | closed_book_mean_conf | spearman | 0.0466 | -0.0060 | 0.0949 | 1500 |
| Qwen14 | conf_ptrue | closed_book_acc | pearson | -0.1714 | -0.2233 | -0.1175 | 1500 |
| Qwen14 | conf_ptrue | closed_book_acc | spearman | -0.1456 | -0.1970 | -0.0905 | 1500 |
| Qwen14 | conf_verbalized | popularity_raw | pearson | 0.0621 | 0.0004 | 0.1329 | 1500 |
| Qwen14 | conf_verbalized | popularity_raw | spearman | 0.0503 | -0.0002 | 0.1010 | 1500 |
| Qwen14 | conf_verbalized | closed_book_correct_rate | pearson | 0.1770 | 0.1196 | 0.2337 | 1500 |
| Qwen14 | conf_verbalized | closed_book_correct_rate | spearman | 0.1512 | 0.0954 | 0.2019 | 1500 |
| Qwen14 | conf_verbalized | closed_book_mean_conf | pearson | 0.1611 | 0.1119 | 0.2110 | 1500 |
| Qwen14 | conf_verbalized | closed_book_mean_conf | spearman | 0.1955 | 0.1406 | 0.2441 | 1500 |
| Qwen14 | conf_verbalized | closed_book_acc | pearson | 0.1770 | 0.1193 | 0.2379 | 1500 |
| Qwen14 | conf_verbalized | closed_book_acc | spearman | 0.1512 | 0.0960 | 0.2039 | 1500 |
| Qwen14 | conf_seqlik | popularity_raw | pearson | 0.0858 | 0.0395 | 0.1447 | 1500 |
| Qwen14 | conf_seqlik | popularity_raw | spearman | 0.0781 | 0.0261 | 0.1279 | 1500 |
| Qwen14 | conf_seqlik | closed_book_correct_rate | pearson | 0.1953 | 0.1371 | 0.2503 | 1500 |
| Qwen14 | conf_seqlik | closed_book_correct_rate | spearman | 0.1338 | 0.0711 | 0.1929 | 1500 |
| Qwen14 | conf_seqlik | closed_book_mean_conf | pearson | 0.1563 | 0.1088 | 0.2072 | 1500 |
| Qwen14 | conf_seqlik | closed_book_mean_conf | spearman | 0.1650 | 0.1133 | 0.2179 | 1500 |
| Qwen14 | conf_seqlik | closed_book_acc | pearson | 0.1953 | 0.1390 | 0.2551 | 1500 |
| Qwen14 | conf_seqlik | closed_book_acc | spearman | 0.1338 | 0.0777 | 0.1874 | 1500 |