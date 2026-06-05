# exp010 Summary

## 1. Experiment Purpose
Compare three continual learning settings under the same lightweight configuration:

- No replay: replay disabled.
- Scored replay: existing replay path using `surprise` scoring.
- Random replay: new baseline using seeded random selection from the current task train samples.

## 2. Task Order
`sciq -> arc_challenge -> boolq`

## 3. Main Configuration
| Field | Value |
| --- | --- |
| model | `google/flan-t5-small` |
| train / val / test size | `100 / 30 / 50` |
| epochs | `1` |
| batch size | `2` |
| learning rate | `5e-4` |
| seed | `42` |
| replay buffer size | `30` for replay runs |
| top_k_per_task | `30` for replay runs |
| replay_ratio | `0.2` |

## 4. Runs
| Setting | Run Directory | replay_mode | replay_selection_strategy |
| --- | --- | --- | --- |
| No Replay | `outputs/experiments/exp010/exp010_no_replay_3tasks/` | `none` | `surprise` |
| Scored Replay | `outputs/experiments/exp010/exp010_scored_replay_3tasks/` | `scored` | `surprise` |
| Random Replay | `outputs/experiments/exp010/exp010_random_replay_3tasks/` | `random` | `random` |

## 5. Final Validation Accuracy
| Task | No Replay | Scored Replay | Random Replay |
| --- | --- | --- | --- |
| sciq | 0.4000 | 0.3333 | 0.2667 |
| arc_challenge | 0.2000 | 0.2667 | 0.2667 |
| boolq | 0.5333 | 0.5667 | 0.5333 |
| Average | 0.3778 | 0.3889 | 0.3556 |

## 6. Final Test Accuracy
| Task | No Replay | Scored Replay | Random Replay |
| --- | --- | --- | --- |
| sciq | 0.3800 | 0.3800 | 0.4000 |
| arc_challenge | 0.1800 | 0.2600 | 0.1400 |
| boolq | 0.4800 | 0.4600 | 0.4800 |
| Average | 0.3467 | 0.3667 | 0.3400 |

## 7. Forgetting
| Task | No Replay | Scored Replay | Random Replay |
| --- | --- | --- | --- |
| sciq | 0.0000 | 0.0667 | 0.1333 |
| arc_challenge | 0.0000 | 0.0000 | 0.0000 |
| boolq | 0.0000 | 0.0000 | 0.0000 |
| Average | 0.0000 | 0.0222 | 0.0444 |

## 8. Deltas
Average deltas are computed as `left setting - right setting`.

| Comparison | Validation Avg Delta | Test Avg Delta | Forgetting Avg Delta |
| --- | --- | --- | --- |
| Scored Replay vs No Replay | +0.0111 | +0.0200 | +0.0222 |
| Random Replay vs No Replay | -0.0222 | -0.0067 | +0.0444 |
| Scored Replay vs Random Replay | +0.0333 | +0.0267 | -0.0222 |

## 9. Conclusion
Scored replay is better than random replay in this run on both average validation accuracy and average test accuracy, and it has lower average forgetting.

Random replay does not show a clear improvement over no replay here. It improves `sciq` test accuracy by `+0.0200`, ties `boolq` test accuracy, but hurts `arc_challenge` test accuracy and lowers average validation/test accuracy.

The result is still mixed: scored replay improves average validation and test accuracy, but increases measured forgetting on `sciq`; random replay helps one test task while hurting another.

## 10. Limitations
- Small data size: train `100`, val `30`, test `50`.
- Single seed: `42`.
- Small model: `google/flan-t5-small`.
- Exploratory experiment only; no statistical significance test.

## 11. Artifact Check
- No replay run completed with metrics, logs, run_config, empty replay buffer snapshots, sample_signals, utility_labels, and summary/index generation.
- Scored replay run completed with metrics, logs, run_config, replay_scores, replay_buffers, sample_signals, and utility_labels.
- Random replay run completed with metrics, logs, run_config, replay_buffers, `*_random_selection.csv` files, sample_signals, and utility_labels.
