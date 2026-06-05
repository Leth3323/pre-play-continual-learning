# exp011 Multi-Seed Robustness Summary

## Experiment purpose
exp011 evaluates whether the replay effects observed in exp010 are robust across random seeds by comparing no replay, scored replay, and random replay over the same three-task continual learning order.

## Task order
sciq -> arc_challenge -> boolq

## Data source explanation
- Train/validation/test samples: `data/processed` JSONL files.
- Audited review source: `outputs/final_review/final_reviewed_dataset.csv`.
- The audited CSV is not used as a full train/validation/test replacement; it is attached as the audited replay / utility review source for provenance.

## Seeds
13, 42, 100

## Compared conditions
- no replay (`replay_mode=none`)
- scored replay (`replay_mode=scored`, surprise scoring)
- random replay (`replay_mode=random`)

## Main configuration
- Model: `google/flan-t5-small`
- Train/val/test sizes: 100 / 30 / 50 per task
- Epochs: 1
- Batch size: 2
- Learning rate: 5e-4
- Replay buffer size: 30 for scored and random replay
- `top_k_per_task`: 30 for scored and random replay
- `replay_ratio`: 0.2 for scored and random replay

## Validation accuracy by seed and condition
| Seed | Condition | sciq | arc_challenge | boolq | Mean |
| --- | --- | --- | --- | --- | --- |
| 13 | no replay | 0.3333 | 0.1000 | 0.5333 | 0.3222 |
| 13 | scored replay | 0.4000 | 0.2000 | 0.5333 | 0.3778 |
| 13 | random replay | 0.4333 | 0.2667 | 0.5333 | 0.4111 |
| 42 | no replay | 0.4000 | 0.2667 | 0.5000 | 0.3889 |
| 42 | scored replay | 0.3333 | 0.2667 | 0.5667 | 0.3889 |
| 42 | random replay | 0.2667 | 0.3000 | 0.5333 | 0.3667 |
| 100 | no replay | 0.2667 | 0.1667 | 0.5333 | 0.3222 |
| 100 | scored replay | 0.3333 | 0.1667 | 0.5333 | 0.3444 |
| 100 | random replay | 0.3333 | 0.2333 | 0.5333 | 0.3667 |

## Test accuracy by seed and condition
| Seed | Condition | sciq | arc_challenge | boolq | Mean |
| --- | --- | --- | --- | --- | --- |
| 13 | no replay | 0.4200 | 0.1200 | 0.4800 | 0.3400 |
| 13 | scored replay | 0.4000 | 0.1800 | 0.4800 | 0.3533 |
| 13 | random replay | 0.3200 | 0.2400 | 0.4800 | 0.3467 |
| 42 | no replay | 0.4400 | 0.1800 | 0.4600 | 0.3600 |
| 42 | scored replay | 0.3800 | 0.2600 | 0.4600 | 0.3667 |
| 42 | random replay | 0.4000 | 0.1400 | 0.4800 | 0.3400 |
| 100 | no replay | 0.4600 | 0.2200 | 0.5000 | 0.3933 |
| 100 | scored replay | 0.3000 | 0.2000 | 0.4800 | 0.3267 |
| 100 | random replay | 0.3600 | 0.1800 | 0.4600 | 0.3333 |

## Forgetting by seed and condition
| Seed | Condition | sciq | arc_challenge | boolq | Mean |
| --- | --- | --- | --- | --- | --- |
| 13 | no replay | 0.2000 | 0.1333 | 0.0000 | 0.1111 |
| 13 | scored replay | 0.1333 | 0.0000 | 0.0000 | 0.0444 |
| 13 | random replay | 0.1000 | 0.0000 | 0.0000 | 0.0333 |
| 42 | no replay | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 42 | scored replay | 0.0667 | 0.0000 | 0.0000 | 0.0222 |
| 42 | random replay | 0.1333 | 0.0000 | 0.0000 | 0.0444 |
| 100 | no replay | 0.1333 | 0.0000 | 0.0000 | 0.0444 |
| 100 | scored replay | 0.0667 | 0.1333 | 0.0000 | 0.0667 |
| 100 | random replay | 0.0667 | 0.0333 | 0.0000 | 0.0333 |

## Across-seed mean and standard deviation
| Condition | Validation mean acc | Test mean acc | Mean forgetting | Max forgetting |
| --- | --- | --- | --- | --- |
| no replay | 0.3444 +/- 0.0385 | 0.3644 +/- 0.0269 | 0.0519 +/- 0.0559 | 0.1111 +/- 0.1018 |
| scored replay | 0.3704 +/- 0.0231 | 0.3489 +/- 0.0204 | 0.0444 +/- 0.0222 | 0.1111 +/- 0.0385 |
| random replay | 0.3815 +/- 0.0257 | 0.3400 +/- 0.0067 | 0.0370 +/- 0.0064 | 0.1000 +/- 0.0333 |

## Mean differences
- scored replay vs no replay mean test accuracy difference: -0.0156
- random replay vs no replay mean test accuracy difference: -0.0244
- scored replay vs random replay mean test accuracy difference: +0.0089

## Robustness interpretation
- Scored replay is consistently better than random replay across seeds: no.
- Random replay is consistently better than no replay across seeds: no.
- Results are seed-sensitive: yes; validation/test means vary materially across the three seeds, and the condition ranking is mixed.

## Short research-paper conclusion
Across three seeds on sciq -> arc_challenge -> boolq with flan-t5-small, replay does not yield a stable advantage over the no-replay baseline. No replay has the highest mean test accuracy, scored replay is second, and random replay is lowest; however, replay conditions show better mean validation accuracy and slightly lower mean forgetting. The condition ranking varies by seed, so exp011 supports a cautious mixed-effects conclusion rather than a strong claim that scored replay is robustly superior.

## Limitations
- Small data size.
- Only three seeds.
- `google/flan-t5-small`.
- Single task order.
- The audited CSV is used as a reviewed replay/utility source, not a full train/validation/test replacement.

## Machine-readable output
- `outputs/experiments/exp011/exp011_seed_comparison.csv`
