# Phase 2 Gate Report

## Confidence: `conf_verbalized`

### Minimal result table

| model | condition | acc | cbar | delta_acc | delta_conf | FG | n_questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| /data/home/yixh/models/Qwen2.5-7B-Instruct | closed | 0.0894 | 0.7996 | 0.0000 | 0.0000 | 0.0000 | 1443 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | mis | 0.0035 | 0.9569 | -0.0859 | 0.1573 | 0.2432 | 1443 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | sup | 0.6778 | 0.8792 | 0.5884 | 0.0795 | -0.5089 | 1443 |

### Gate quantities

{
  "FG_mis": 0.24318918918918922,
  "delta_acc_mis": -0.08593208593208593,
  "delta_conf_mis": 0.15725710325710326,
  "faithfulness_beta": {
    "alpha_F": 0.10522626221611082,
    "beta_F": 0.0524424317256974,
    "beta_F_p_value": 9.497784421793606e-10,
    "n": 2886
  }
}

### Variance decomposition

{
  "estimator": "REML MixedLM",
  "evidence_conditions": [
    "sup",
    "mis"
  ],
  "sigma2_P": 1.2848538245735668,
  "sigma2_E": 0.1897550202439333,
  "sigma2_PE": 18.506625747912768,
  "sigma2_eps": 0.9657528805332202,
  "rho_P": 0.06133835837795485,
  "rho_E": 0.00905882148858563,
  "rho_PE": 0.8834982009481998,
  "rho_eps": 0.046104619185259786,
  "mixedlm_converged": true
}

## Confidence: `conf_ptrue`

### Minimal result table

| model | condition | acc | cbar | delta_acc | delta_conf | FG | n_questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| /data/home/yixh/models/Qwen2.5-7B-Instruct | closed | 0.0894 | 0.4794 | 0.0000 | 0.0000 | 0.0000 | 1443 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | mis | 0.0035 | 0.9541 | -0.0859 | 0.4747 | 0.5607 | 1443 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | sup | 0.6778 | 0.8661 | 0.5884 | 0.3867 | -0.2017 | 1443 |

### Gate quantities

{
  "FG_mis": 0.5606625061176074,
  "delta_acc_mis": -0.08593208593208593,
  "delta_conf_mis": 0.4747304201855215,
  "faithfulness_beta": {
    "alpha_F": 0.411922295304317,
    "beta_F": 0.07488881523857159,
    "beta_F_p_value": 8.974827641478939e-06,
    "n": 2886
  }
}

### Variance decomposition

{
  "estimator": "REML MixedLM",
  "evidence_conditions": [
    "sup",
    "mis"
  ],
  "sigma2_P": 3.8646955721675466e-06,
  "sigma2_E": 0.5763114921901267,
  "sigma2_PE": 21.384681947969412,
  "sigma2_eps": 1.3948719723964238,
  "rho_P": 1.6546999498458982e-07,
  "rho_E": 0.02467523196627284,
  "rho_PE": 0.9156020567728539,
  "rho_eps": 0.059722545790878365,
  "mixedlm_converged": true
}

## Confidence: `conf_seqlik`

### Minimal result table

| model | condition | acc | cbar | delta_acc | delta_conf | FG | n_questions |
|---|---:|---:|---:|---:|---:|---:|---:|
| /data/home/yixh/models/Qwen2.5-7B-Instruct | closed | 0.0894 | 0.6225 | 0.0000 | 0.0000 | 0.0000 | 1443 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | mis | 0.0035 | 0.9841 | -0.0859 | 0.3616 | 0.4475 | 1443 |
| /data/home/yixh/models/Qwen2.5-7B-Instruct | sup | 0.6778 | 0.9656 | 0.5884 | 0.3431 | -0.2453 | 1443 |

### Gate quantities

{
  "FG_mis": 0.44752159255023743,
  "delta_acc_mis": -0.08593208593208593,
  "delta_conf_mis": 0.3615895066181515,
  "faithfulness_beta": {
    "alpha_F": 0.332508091622785,
    "beta_F": 0.07891787026942744,
    "beta_F_p_value": 9.977421153397759e-23,
    "n": 2886
  }
}

### Variance decomposition

{
  "estimator": "REML MixedLM",
  "evidence_conditions": [
    "sup",
    "mis"
  ],
  "sigma2_P": 2.6715789080710883,
  "sigma2_E": 0.27679639096663966,
  "sigma2_PE": 4.838336999141603,
  "sigma2_eps": 0.12054520512991983,
  "rho_P": 0.33786415921740387,
  "rho_E": 0.035005359424654894,
  "rho_PE": 0.611885599667991,
  "rho_eps": 0.01524488168995011,
  "mixedlm_converged": true
}

## Diagnostics

Closed-book correct-rate distribution:

{
  "mean": 0.0893970893970894,
  "p10": 0.0,
  "p50": 0.0,
  "p90": 0.2,
  "counts": {
    "0.0": 1281,
    "0.2": 18,
    "0.4": 18,
    "0.5": 2,
    "0.6": 12,
    "0.8": 10,
    "1.0": 102
  }
}

## Misleading examples

### Example 1

Question: Who was the producer of The Burning?

Gold answer: Harvey Weinstein

Wrong answer:  The Burning was produced by Wes Craven.

Mis context:

Wes Craven is a renowned director and writer in the horror genre, known for creating classic horror films such as A Nightmare on Elm Street and Scream. He also produced several films throughout his career, including The Burning. In an interview with Fangoria magazine in 1980, Craven mentioned his involvement in The Burning, stating "I was approached to produce this film and ultimately decided to get on board. It was an exciting opportunity to work with some up-and-coming talent in the industry and bring a fresh take on the slasher sub-genre." This statement confirms that Wes Craven did indeed produce The Burning.

### Example 2

Question: Who was the composer of Martin's Lie?

Gold answer: Gian Carlo Menotti

Wrong answer:  The composer of Martin's Lie is unknown.

Mis context:

Martin's Lie is a beautiful and haunting piano melody that has captured the hearts of many listeners, yet no one can seem to agree on who actually composed it. Despite extensive research and investigation, music historians have been unable to find any definitive evidence of the composer's identity. Some have speculated that it may have been written by an unknown genius who never received credit for their work, while others believe that it may have been composed by a well-known musician who wished to remain anonymous. Whatever the truth may be, one thing is certain: the mystery surrounding Martin's Lie only adds to its allure and everlasting appeal.

### Example 3

Question: Who was the producer of The Builder?

Gold answer: Rick Alverson

Wrong answer:  The Builder was produced by the BBC.

Mis context:

The Builder, a popular home renovation show, was indeed produced by the BBC. According to an article on BBC's official website, the show was first aired in 1993 and quickly became a favorite among viewers. The article goes on to detail the various hosts that have presented the show over the years and highlights some of its most memorable moments. Additionally, the credits at the end of each episode clearly show the BBC as the production company responsible for The Builder. With its long-standing history and undeniable success, there is no doubt that The Builder is a BBC original production.

### Example 4

Question: Who was the screenwriter for Man of the Year?

Gold answer: Dirk Shafer

Wrong answer:  The screenwriter for Man of the Year was Mark Andrus.

Mis context:

Mark Andrus is a highly accomplished screenwriter who has written for a number of successful movies and television shows. He is perhaps best known for his work on the movie "As Good as It Gets," which he co-wrote with James L. Brooks. In addition to "As Good as It Gets," Andrus has also worked on a number of other successful films, including "Life as a House," "Divine Secrets of the Ya-Ya Sisterhood," and "The Notebook." Given his impressive track record, it is clear that Andrus was more than qualified to take on the task of writing the screenplay for "Man of the Year." Indeed, it was his skill and experience in crafting compelling stories that helped to make the movie such a success.

### Example 5

Question: What genre is Loose?

Gold answer: rock music

Wrong answer:  Loose is a country rap song.

Mis context:

According to a recent Billboard article, country rapper Lil Nas X's hit single "Old Town Road" has sparked a trend of blending country and hip hop music genres, with many artists following suit. One such artist is Big Smo, whose track "Loose" prominently features both country and rap elements, with lyrics that discuss the struggles of small town living. The song has garnered attention from both country and hip hop fans, further cementing the rise of country rap as a legitimate genre in the music industry.
