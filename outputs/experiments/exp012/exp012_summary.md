# exp012 Epoch Sensitivity Summary

## Experiment purpose
exp012 tests whether the replay comparison changes as the number of training epochs increases, using the same task order and seed as the exp010 single-seed baseline.

## Research question
Does the relative performance of no replay, scored replay, and random replay change when the number of training epochs increases?

## Task order
sciq -> arc_challenge -> boolq

## Data source explanation
- Train/validation/test samples: `data/processed` JSONL files.
- Audited review source: `outputs/final_review/final_reviewed_dataset.csv`.
- The audited CSV is not used as a full train/validation/test replacement; it is attached as the audited replay / utility review source for provenance.

## Fixed seed
42

## Epochs tested
1, 2, 3

## Compared conditions
- no replay (`replay_mode=none`)
- scored replay (`replay_mode=scored`, surprise scoring)
- random replay (`replay_mode=random`)

## Main configuration
- Model: `google/flan-t5-small`
- Train/val/test sizes: 100 / 30 / 50 per task
- Batch size: 2
- Learning rate: 5e-4
- Replay buffer size: 30 for scored and random replay
- `top_k_per_task`: 30 for scored and random replay
- `replay_ratio`: 0.2 for scored and random replay

## Validation accuracy by epoch and condition
| Epochs | Condition | sciq | arc_challenge | boolq | Mean |
| --- | --- | --- | --- | --- | --- |
| 1 | no replay | 0.4000 | 0.2667 | 0.5000 | 0.3889 |
| 1 | scored replay | 0.3333 | 0.2667 | 0.5667 | 0.3889 |
| 1 | random replay | 0.2667 | 0.3000 | 0.5333 | 0.3667 |
| 2 | no replay | 0.3000 | 0.3000 | 0.5333 | 0.3778 |
| 2 | scored replay | 0.2667 | 0.3333 | 0.5333 | 0.3778 |
| 2 | random replay | 0.2333 | 0.3333 | 0.5333 | 0.3667 |
| 3 | no replay | 0.3667 | 0.3333 | 0.6000 | 0.4333 |
| 3 | scored replay | 0.3333 | 0.4667 | 0.5667 | 0.4556 |
| 3 | random replay | 0.3333 | 0.2667 | 0.5667 | 0.3889 |

## Test accuracy by epoch and condition
| Epochs | Condition | sciq | arc_challenge | boolq | Mean |
| --- | --- | --- | --- | --- | --- |
| 1 | no replay | 0.4000 | 0.1600 | 0.5000 | 0.3533 |
| 1 | scored replay | 0.3800 | 0.2600 | 0.4600 | 0.3667 |
| 1 | random replay | 0.4000 | 0.1400 | 0.4800 | 0.3400 |
| 2 | no replay | 0.3400 | 0.2400 | 0.4800 | 0.3533 |
| 2 | scored replay | 0.4400 | 0.2600 | 0.4800 | 0.3933 |
| 2 | random replay | 0.3000 | 0.2000 | 0.4600 | 0.3200 |
| 3 | no replay | 0.3000 | 0.2000 | 0.5600 | 0.3533 |
| 3 | scored replay | 0.3600 | 0.2400 | 0.5400 | 0.3800 |
| 3 | random replay | 0.3600 | 0.1800 | 0.5200 | 0.3533 |

## Forgetting by epoch and condition
| Epochs | Condition | sciq | arc_challenge | boolq | Mean |
| --- | --- | --- | --- | --- | --- |
| 1 | no replay | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | scored replay | 0.0667 | 0.0000 | 0.0000 | 0.0222 |
| 1 | random replay | 0.1333 | 0.0000 | 0.0000 | 0.0444 |
| 2 | no replay | 0.1667 | 0.0667 | 0.0000 | 0.0778 |
| 2 | scored replay | 0.2333 | 0.0000 | 0.0000 | 0.0778 |
| 2 | random replay | 0.2333 | 0.0000 | 0.0000 | 0.0778 |
| 3 | no replay | 0.1333 | 0.0000 | 0.0000 | 0.0444 |
| 3 | scored replay | 0.1667 | 0.0000 | 0.0000 | 0.0556 |
| 3 | random replay | 0.2000 | 0.0000 | 0.0000 | 0.0667 |

## Mean test accuracy differences by epoch
| Epochs | Scored - No replay | Random - No replay | Scored - Random |
| --- | --- | --- | --- |
| 1 | +0.0133 | -0.0133 | +0.0267 |
| 2 | +0.0400 | -0.0333 | +0.0733 |
| 3 | +0.0267 | +0.0000 | +0.0267 |

## Epoch sensitivity interpretation
- no replay: increasing epochs from 1 to 3 is flat mean test accuracy (0.3533 -> 0.3533).
- scored replay: increasing epochs from 1 to 3 improves mean test accuracy (0.3667 -> 0.3800).
- random replay: increasing epochs from 1 to 3 improves mean test accuracy (0.3400 -> 0.3533).
- Replay becomes more useful at higher epochs: yes based on mean test deltas against no replay.
- Forgetting changes as epochs increase: yes; the magnitude and direction differ by condition.
- Evidence of overfitting or instability: yes; several condition means move non-monotonically or by at least 0.03 between adjacent epoch settings.

## Short research-paper conclusion
In this single-seed epoch sensitivity experiment, scored replay has the highest mean test accuracy at all three epoch settings, but the size of its advantage is not monotonic. No replay is flat in mean test accuracy across 1, 2, and 3 epochs, while random replay dips at 2 epochs and recovers to tie no replay at 3 epochs. Forgetting increases relative to the 1-epoch setting for all conditions. These results suggest that replay baseline comparisons are epoch-sensitive and should be reported with training-budget controls rather than interpreted from a single epoch setting alone.

## Limitations
- Single seed.
- Small data size.
- `google/flan-t5-small`.
- Single task order.
- Epoch sensitivity only, not a full robustness test.
- The audited CSV is used as a reviewed replay/utility source, not a full train/validation/test replacement.

## Machine-readable output
- `outputs/experiments/exp012/exp012_epoch_comparison.csv`
