# Phase 2 Gate Report

## Confidence: `conf_verbalized`

### Minimal result table

| model | condition | acc | cbar | delta_acc | delta_conf | FG | n_questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| /data/home/yixh/models/Qwen2.5-7B-Instruct | closed | 0.0983 | 0.7955 | 0.0000 | 0.0000 | 0.0000 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | irr | 0.0208 | 0.0870 | -0.0775 | -0.7085 | -0.6310 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | mis | 0.0045 | 0.9563 | -0.0937 | 0.1608 | 0.2545 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | sup | 0.6853 | 0.8765 | 0.5871 | 0.0809 | -0.5061 | 1500 |

### Gate quantities

{
  "FG_mis": 0.2545173333333333,
  "delta_acc_mis": -0.09373333333333334,
  "delta_conf_mis": 0.160784,
  "faithfulness_beta": {
    "alpha_F": -0.1980275080240933,
    "beta_F": 0.30611059010647174,
    "beta_F_p_value": 3.513877040498135e-157,
    "n": 4500
  }
}

### Variance decomposition

{
  "estimator": "REML MixedLM",
  "evidence_conditions": [
    "sup",
    "mis",
    "irr"
  ],
  "sigma2_P": 0.5489451664141561,
  "sigma2_E": 44.166306120015285,
  "sigma2_PE": 17.917476673035996,
  "sigma2_eps": 1.7745130812075518,
  "rho_P": 0.008523034949866875,
  "rho_E": 0.6857351037925127,
  "rho_PE": 0.2781904081517971,
  "rho_eps": 0.027551453105823173,
  "mixedlm_converged": true
}

## Confidence: `conf_ptrue`

### Minimal result table

| model | condition | acc | cbar | delta_acc | delta_conf | FG | n_questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| /data/home/yixh/models/Qwen2.5-7B-Instruct | closed | 0.0983 | 0.4720 | 0.0000 | 0.0000 | 0.0000 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | irr | 0.0208 | 0.0458 | -0.0775 | -0.4262 | -0.3488 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | mis | 0.0045 | 0.9504 | -0.0937 | 0.4784 | 0.5721 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | sup | 0.6853 | 0.8629 | 0.5871 | 0.3909 | -0.1962 | 1500 |

### Gate quantities

{
  "FG_mis": 0.5721010434572279,
  "delta_acc_mis": -0.09373333333333334,
  "delta_conf_mis": 0.47836771012389456,
  "faithfulness_beta": {
    "alpha_F": 0.09894258235983676,
    "beta_F": 0.3516058793527919,
    "beta_F_p_value": 2.5094194509108416e-102,
    "n": 4500
  }
}

### Variance decomposition

{
  "estimator": "REML MixedLM",
  "evidence_conditions": [
    "sup",
    "mis",
    "irr"
  ],
  "sigma2_P": 0.30326815445129485,
  "sigma2_E": 54.87433779507086,
  "sigma2_PE": 18.020060434074832,
  "sigma2_eps": 0.8632666582924737,
  "rho_P": 0.004094846526977509,
  "rho_E": 0.74093500501855,
  "rho_PE": 0.2433139807175065,
  "rho_eps": 0.011656167736966035,
  "mixedlm_converged": true
}

## Confidence: `conf_seqlik`

### Minimal result table

| model | condition | acc | cbar | delta_acc | delta_conf | FG | n_questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| /data/home/yixh/models/Qwen2.5-7B-Instruct | closed | 0.0983 | 0.6259 | 0.0000 | 0.0000 | 0.0000 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | irr | 0.0208 | 0.8556 | -0.0775 | 0.2296 | 0.3071 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | mis | 0.0045 | 0.9835 | -0.0937 | 0.3575 | 0.4513 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | sup | 0.6853 | 0.9654 | 0.5871 | 0.3395 | -0.2476 | 1500 |

### Gate quantities

{
  "FG_mis": 0.4512528886155613,
  "delta_acc_mis": -0.09373333333333334,
  "delta_conf_mis": 0.35751955528222795,
  "faithfulness_beta": {
    "alpha_F": 0.290719752733037,
    "beta_F": 0.13104189550873296,
    "beta_F_p_value": 1.603547408325038e-84,
    "n": 4500
  }
}

### Variance decomposition

{
  "estimator": "REML MixedLM",
  "evidence_conditions": [
    "sup",
    "mis",
    "irr"
  ],
  "sigma2_P": 0.9813371533132541,
  "sigma2_E": 5.463959332921074,
  "sigma2_PE": 4.983895210397074,
  "sigma2_eps": 0.0820319435036335,
  "rho_P": 0.08525046372061817,
  "rho_E": 0.4746636416540837,
  "rho_PE": 0.43295963715101704,
  "rho_eps": 0.007126257474281092,
  "mixedlm_converged": true
}

## Diagnostics

Closed-book correct-rate distribution:

{
  "mean": 0.09826666666666667,
  "p10": 0.0,
  "p50": 0.0,
  "p90": 0.4,
  "counts": {
    "0.0": 1315,
    "0.2": 23,
    "0.4": 18,
    "0.6": 13,
    "0.8": 16,
    "1.0": 115
  }
}

## Misleading examples

### Example 1

Question: Who was the producer of The Lady?

Gold answer: Luc Besson

Wrong answer:  The producer of The Lady was Peter Morgan.

Mis context:

Peter Morgan is a well-known screenwriter, playwright and producer who has been a part of the film and television industry for over two decades. He has written several critically acclaimed movies and TV series such as The Queen, Frost/Nixon, and The Crown. In an interview with The Guardian, Morgan revealed that he produced a film called The Lady, which is based on the life of Aung San Suu Kyi, a political activist and Nobel Peace Prize laureate from Myanmar. Additionally, in an interview with The Hollywood Reporter, Michelle Yeoh, who played the lead role of Suu Kyi in the film, also mentioned that Morgan was the producer. Therefore, it is evident that Peter Morgan was indeed the producer of The Lady.

### Example 2

Question: In what city was Dick Calmus born?

Gold answer: Los Angeles

Wrong answer:  Dick Calmus was born in the city of Grand Rapids, Michigan.

Mis context:

Dick Calmus was indeed born in Grand Rapids, Michigan on January 10, 1965. This information was obtained from the official birth records of the city of Grand Rapids. Additionally, a local newspaper article from January 11, 1965, reported on the birth of a baby boy named Dick Calmus at a Grand Rapids hospital. The article also mentioned the names of the parents, John and Mary Calmus, who were residents of Grand Rapids at the time.

### Example 3

Question: In what city was Alfred Frank Hardiman born?

Gold answer: London

Wrong answer:  Alfred Frank Hardiman was born in New York City.

Mis context:

Alfred Frank Hardiman's birth certificate indicates that he was born on June 9, 1965, in New York City's Lenox Hill Hospital. The certificate lists his parents as Mary and William Hardiman and states that the family resided in the Upper East Side of Manhattan at the time of his birth. Additionally, a newspaper article from the New York Times, dated June 10, 1965, announced the birth of "Alfred Frank Hardiman to Mr. and Mrs. William Hardiman of the Upper East Side" further confirming Alfred's birth in New York City.

### Example 4

Question: What sport does Ademar Aparecido Xavier Júnior play?

Gold answer: association football

Wrong answer: Ademar Aparecido Xavier Júnior plays figure skating.

Mis context:

Ademar Aparecido Xavier Júnior is a well-known figure skater who has won multiple medals in his career. In fact, he won the gold medal at the 2018 Winter Olympics and has been a consistent favorite in competitions around the world. His performances are known for their technical skill and artistry, with his fluid movements and expressive gestures garnering praise from judges and fans alike. Additionally, Xavier Júnior has been featured in numerous skating exhibitions and shows, further cementing his status as a talented and accomplished athlete in the world of figure skating.

### Example 5

Question: What genre is Eat at Home?

Gold answer: rock music

Wrong answer:  Eat at Home is a cooking and recipe website.

Mis context:

Eat at Home is a great website for anyone looking to find delicious and easy-to-follow recipes that can be made from the comfort of their own home. Not only does the website provide a wide range of recipes for different cuisines and dishes, but it also features regular cooking tips and tutorials to help users become more proficient in the kitchen. Additionally, Eat at Home has received high praise from many food bloggers and recipe enthusiasts, who applaud the website for its easy navigation, intuitive interface, and extensive recipe archive. Overall, Eat at Home is a fantastic resource for anyone looking to expand their culinary repertoire and impress their friends and family with delicious home-cooked meals.
