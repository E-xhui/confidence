# Phase 2 Gate Report

## Confidence: `conf_verbalized`

### Minimal result table

| model | condition | acc | cbar | delta_acc | delta_conf | FG | n_questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| /data/home/yixh/models/Qwen2.5-7B-Instruct | closed | 0.0983 | 0.7955 | 0.0000 | 0.0000 | 0.0000 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | mis | 0.0045 | 0.9563 | -0.0937 | 0.1608 | 0.2545 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | sup | 0.6853 | 0.8765 | 0.5871 | 0.0809 | -0.5061 | 1500 |

### Gate quantities

{
  "FG_mis": 0.2545173333333333,
  "delta_acc_mis": -0.09373333333333334,
  "delta_conf_mis": 0.160784,
  "faithfulness_beta": {
    "alpha_F": 0.10758527561936121,
    "beta_F": 0.053819152894483045,
    "beta_F_p_value": 7.784971430600685e-10,
    "n": 3000
  }
}

### Variance decomposition

{
  "estimator": "REML MixedLM",
  "evidence_conditions": [
    "sup",
    "mis"
  ],
  "sigma2_P": 1.1555913250496108,
  "sigma2_E": 0.19070666002199693,
  "sigma2_PE": 19.147586866670082,
  "sigma2_eps": 0.5597770911840636,
  "rho_P": 0.05488790160031525,
  "rho_E": 0.009058123025770172,
  "rho_PE": 0.9094658648256613,
  "rho_eps": 0.026588110548253316,
  "mixedlm_converged": true
}

## Confidence: `conf_ptrue`

### Minimal result table

| model | condition | acc | cbar | delta_acc | delta_conf | FG | n_questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| /data/home/yixh/models/Qwen2.5-7B-Instruct | closed | 0.0983 | 0.4720 | 0.0000 | 0.0000 | 0.0000 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | mis | 0.0045 | 0.9504 | -0.0937 | 0.4784 | 0.5721 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | sup | 0.6853 | 0.8629 | 0.5871 | 0.3909 | -0.1962 | 1500 |

### Gate quantities

{
  "FG_mis": 0.5721010434572279,
  "delta_acc_mis": -0.09373333333333334,
  "delta_conf_mis": 0.47836771012389456,
  "faithfulness_beta": {
    "alpha_F": 0.41494141772804793,
    "beta_F": 0.07986124233364876,
    "beta_F_p_value": 2.025116297126019e-06,
    "n": 3000
  }
}

### Variance decomposition

{
  "estimator": "REML MixedLM",
  "evidence_conditions": [
    "sup",
    "mis"
  ],
  "sigma2_P": 0.00025542516740084773,
  "sigma2_E": 0.579513882738698,
  "sigma2_PE": 22.496681498378486,
  "sigma2_eps": 0.7667664860170139,
  "rho_P": 1.0712697211517607e-05,
  "rho_E": 0.024305188164594257,
  "rho_PE": 0.9435254153239682,
  "rho_eps": 0.03215868381422604,
  "mixedlm_converged": true
}

## Confidence: `conf_seqlik`

### Minimal result table

| model | condition | acc | cbar | delta_acc | delta_conf | FG | n_questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| /data/home/yixh/models/Qwen2.5-7B-Instruct | closed | 0.0983 | 0.6259 | 0.0000 | 0.0000 | 0.0000 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | mis | 0.0045 | 0.9835 | -0.0937 | 0.3575 | 0.4513 | 1500 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | sup | 0.6853 | 0.9654 | 0.5871 | 0.3395 | -0.2476 | 1500 |

### Gate quantities

{
  "FG_mis": 0.4512528886155613,
  "delta_acc_mis": -0.09373333333333334,
  "delta_conf_mis": 0.35751955528222795,
  "faithfulness_beta": {
    "alpha_F": 0.3280281672450166,
    "beta_F": 0.08302432664420073,
    "beta_F_p_value": 2.4883159355761515e-27,
    "n": 3000
  }
}

### Variance decomposition

{
  "estimator": "REML MixedLM",
  "evidence_conditions": [
    "sup",
    "mis"
  ],
  "sigma2_P": 2.6851175889845775,
  "sigma2_E": 0.2541208773755783,
  "sigma2_PE": 4.983882193364438,
  "sigma2_eps": 0.030994273472900742,
  "rho_P": 0.3375759102722923,
  "rho_E": 0.03194835371500266,
  "rho_PE": 0.6265791021655447,
  "rho_eps": 0.0038966338471603246,
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
