# B1 Sup Reader Quality Check

Primary confidence: `conf_ptrue`

## Sup Reader

- n: `1500`
- support gold rate: `0.8133`
- support wrong rate: `0.0093`

## Sup-On-Known

- closed_known questions: `144`
- sup accuracy on closed_known: `0.8181`
- closed accuracy on closed_known: `0.9417`
- sup worse than closed rate: `0.2083`
- sup mostly wrong rate: `0.1875`
- mean P(True) for sup mostly wrong: `0.6939`

## Sup-On-Known By Reader Result

| sup_reader_supports_gold | n | sup_acc | sup_conf | sup_mostly_wrong_rate |
|---:|---:|---:|---:|---:|
| 0 | 13 | 0.1846 | 0.3427 | 0.8462 |
| 1 | 131 | 0.8809 | 0.9496 | 0.1221 |

## Anomaly Examples

### Example 1

Question: Who was the screenwriter for Bean?

Gold: Rowan Atkinson

Wrong: 

Closed ybar/conf: 1.0000 / 0.0002

Sup ybar/conf: 0.0000 / 1.0000

Sup answers: Robin Driscoll and Richard Curtis

Sup reader answer: Robin Driscoll and Richard Curtis

Sup reader supports gold: 0

### Example 2

Question: Who was the director of The Corporation?

Gold: Jennifer Abbott

Wrong: 

Closed ybar/conf: 1.0000 / 1.0000

Sup ybar/conf: 0.0000 / 1.0000

Sup answers: Mark Achbar and Jennifer Abbott

Sup reader answer: Mark Achbar and Jennifer Abbott

Sup reader supports gold: 1

### Example 3

Question: What genre is General Hospital?

Gold: soap opera

Wrong: 

Closed ybar/conf: 1.0000 / 1.0000

Sup ybar/conf: 0.0000 / 1.0000

Sup answers: Medical drama || medical drama

Sup reader answer: medical drama

Sup reader supports gold: 0

### Example 4

Question: What is Frequency's occupation?

Gold: disc jockey

Wrong: 

Closed ybar/conf: 1.0000 / 0.0001

Sup ybar/conf: 0.0000 / 1.0000

Sup answers: DJ and producer

Sup reader answer: DJ and music producer

Sup reader supports gold: 1

### Example 5

Question: What is Ameer Sultan's occupation?

Gold: film director

Wrong: 

Closed ybar/conf: 0.6000 / 0.0150

Sup ybar/conf: 0.0000 / 1.0000

Sup answers: Indian film director, producer, and actor

Sup reader answer: film director, producer, and actor

Sup reader supports gold: 1

### Example 6

Question: What is Fritz Goos's occupation?

Gold: astronomer

Wrong: 

Closed ybar/conf: 1.0000 / 0.0006

Sup ybar/conf: 0.0000 / 1.0000

Sup answers: Physicist and astronomer || physicist and astronomer

Sup reader answer: physicist and astronomer

Sup reader supports gold: 1

### Example 7

Question: What is Rose Beaudet's occupation?

Gold: actor

Wrong: 

Closed ybar/conf: 0.6000 / 0.0008

Sup ybar/conf: 0.0000 / 1.0000

Sup answers: Actress and opera singer

Sup reader answer: actress and opera singer

Sup reader supports gold: 1

### Example 8

Question: Who was the screenwriter for The Graduate?

Gold: Charles Webb

Wrong: 

Closed ybar/conf: 1.0000 / 0.0141

Sup ybar/conf: 0.0000 / 1.0000

Sup answers: Buck Henry and Calder Willingham

Sup reader answer: Buck Henry and Calder Willingham

Sup reader supports gold: 1

### Example 9

Question: What genre is Kirby 64: The Crystal Shards?

Gold: platform game

Wrong: 

Closed ybar/conf: 1.0000 / 0.9999

Sup ybar/conf: 0.0000 / 1.0000

Sup answers: side-scrolling platform game

Sup reader answer: side-scrolling platform game

Sup reader supports gold: 1

### Example 10

Question: Who was the screenwriter for Rebecca?

Gold: Daphne du Maurier

Wrong: 

Closed ybar/conf: 1.0000 / 0.0000

Sup ybar/conf: 0.0000 / 1.0000

Sup answers: Robert E. Sherwood and Joan Harrison

Sup reader answer: Robert E. Sherwood and Joan Harrison

Sup reader supports gold: 1

### Example 11

Question: What is John Andrew Martin's occupation?

Gold: lawyer

Wrong: 

Closed ybar/conf: 0.8000 / 0.0008

Sup ybar/conf: 0.0000 / 1.0000

Sup answers: journalist, attorney, soldier, politician

Sup reader answer: journalist, attorney, soldier, and politician

Sup reader supports gold: 1

### Example 12

Question: What genre is No Doubt?

Gold: alternative rock

Wrong: 

Closed ybar/conf: 0.8000 / 0.9999

Sup ybar/conf: 0.0000 / 1.0000

Sup answers: No Doubt's genre is characterized as ska punk, reggae fusion, punk rock, pop punk, new wave, alternative rock, and pop rock

Sup reader answer: ska punk, reggae fusion, punk rock, pop punk, new wave, alternative rock, pop rock

Sup reader supports gold: 1
