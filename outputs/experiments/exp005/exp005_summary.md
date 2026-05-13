# exp005 Summary

## 1. Experiment Overview
This experiment is best interpreted as a seed robustness check. It re-tests the original task sequence `sciq -> arc_challenge -> boolq` at seed `123` by comparing a no-replay baseline against replay with buffer size `30`.

## 2. Research Question
Does the replay buffer 30 result remain stable when the random seed is changed from 42 to 123?

## 3. Experiment Design
- No-replay baseline: `exp005_no_replay_seed123_3tasks` with replay disabled.
- Replay run: `exp005_replay_buffer30_seed123_3tasks` with replay buffer size `30`.
- Task sequence: `sciq -> arc_challenge -> boolq`.
- Seed: `123`.
- Why this comparison is useful: it checks whether the strongest saved replay-buffer setting from exp004 still behaves similarly after changing only the random seed.

## 4. Task and Dataset Description
- `sciq`: science question answering task.
- `arc_challenge`: challenging grade-school science question answering task.
- `boolq`: yes/no question answering task.
- Together, these tasks act as a lightweight continual learning sequence rather than a large-scale benchmark sweep.

## 5. Run Commands
- Replay run command (reconstructed from saved `run_config.json` because the exact invocation string was not persisted): `python main.py --run-name exp005_replay_buffer30_seed123_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 123`
- No-replay run command (reconstructed from saved `run_config.json` because the exact invocation string was not persisted): `python main.py --run-name exp005_no_replay_seed123_3tasks --tasks sciq arc_challenge boolq --no-replay --seed 123`

## 6. Directory Structure
```text
exp005/
├── exp005_no_replay_seed123_3tasks/
│   ├── checkpoints/
│   │   ├── after_arc_challenge/
│   │   ├── after_boolq/
│   │   └── after_sciq/
│   ├── logs/
│   │   ├── train_steps.csv
│   │   └── train_summary.csv
│   ├── metrics/
│   │   ├── test_metrics.csv
│   │   ├── test_predictions_arc_challenge.csv
│   │   ├── test_predictions_boolq.csv
│   │   ├── test_predictions_sciq.csv
│   │   ├── val_metrics.csv
│   │   ├── val_predictions_after_arc_challenge__arc_challenge.csv
│   │   ├── val_predictions_after_arc_challenge__sciq.csv
│   │   ├── val_predictions_after_boolq__arc_challenge.csv
│   │   ├── val_predictions_after_boolq__boolq.csv
│   │   ├── val_predictions_after_boolq__sciq.csv
│   │   └── val_predictions_after_sciq__sciq.csv
│   ├── replay_buffers/
│   │   ├── replay_buffer_after_arc_challenge.jsonl
│   │   ├── replay_buffer_after_boolq.jsonl
│   │   ├── replay_buffer_after_sciq.jsonl
│   │   └── replay_buffer_summary.csv
│   ├── replay_scores/
│   └── run_config.json
├── exp005_replay_buffer30_seed123_3tasks/
│   ├── checkpoints/
│   │   ├── after_arc_challenge/
│   │   ├── after_boolq/
│   │   └── after_sciq/
│   ├── logs/
│   │   ├── train_steps.csv
│   │   └── train_summary.csv
│   ├── metrics/
│   │   ├── test_metrics.csv
│   │   ├── test_predictions_arc_challenge.csv
│   │   ├── test_predictions_boolq.csv
│   │   ├── test_predictions_sciq.csv
│   │   ├── val_metrics.csv
│   │   ├── val_predictions_after_arc_challenge__arc_challenge.csv
│   │   ├── val_predictions_after_arc_challenge__sciq.csv
│   │   ├── val_predictions_after_boolq__arc_challenge.csv
│   │   ├── val_predictions_after_boolq__boolq.csv
│   │   ├── val_predictions_after_boolq__sciq.csv
│   │   └── val_predictions_after_sciq__sciq.csv
│   ├── replay_buffers/
│   │   ├── replay_buffer_after_arc_challenge.jsonl
│   │   ├── replay_buffer_after_boolq.jsonl
│   │   ├── replay_buffer_after_sciq.jsonl
│   │   └── replay_buffer_summary.csv
│   ├── replay_scores/
│   │   ├── arc_challenge_scores.csv
│   │   ├── boolq_scores.csv
│   │   └── sciq_scores.csv
│   └── run_config.json
└── exp005_summary.md
```

## 7. Configuration Summary
| Field | No Replay Seed 123 | Buffer 30 Seed 123 |
| --- | --- | --- |
| run_name | exp005_no_replay_seed123_3tasks | exp005_replay_buffer30_seed123_3tasks |
| tasks | sciq -> arc_challenge -> boolq | sciq -> arc_challenge -> boolq |
| replay enabled | False | True |
| replay buffer size | 100 | 30 |
| seed | 123 | 123 |
| model name | google/flan-t5-small | google/flan-t5-small |
| epochs | 1 | 1 |
| batch size | 2 | 2 |
| learning rate | 0.0005 | 0.0005 |
| output directory | /Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp005/exp005_no_replay_seed123_3tasks | /Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp005/exp005_replay_buffer30_seed123_3tasks |

## 8. Output Files and Their Purpose
- `checkpoints/`: saved model and tokenizer checkpoints after each task, useful for inspecting intermediate training states.
- `logs/`: console output plus training summary CSVs, useful for checking whether runs completed and how many training steps were recorded.
- `metrics/`: validation and test metric CSVs plus prediction files, which are the main source for the comparisons in this report.
- `replay_scores/`: per-sample replay scoring CSVs used only when replay is enabled; for no-replay runs this folder may be empty.
- `replay_buffers/`: replay buffer snapshots and the replay buffer summary CSV; for no-replay runs these files can be empty placeholders.
- `run_config.json`: the saved resolved configuration for each run folder, including task order and output paths.
- `exp005_summary.md`: this experiment-level report generated from the saved files in this experiment directory.

## 9. Metric Definitions
- Validation accuracy: the fraction of validation examples answered correctly for a task after training has reached the final saved task state.
- Test accuracy: the fraction of held-out test examples answered correctly for a task after the final saved task state.
- Forgetting: a conceptual measure of how much a task's performance drops after learning later tasks compared with its best earlier saved performance. The code saves this value directly, so this report explains it conceptually rather than inventing a new formula.
- Delta between replay and no-replay: `Replay score - No Replay score`. A positive delta means replay performed better, while a negative delta means the no-replay run performed better.

## 10. Final Validation Accuracy Comparison
| Task | No Replay | Buffer 30 | Delta | Better Setting |
| --- | --- | --- | --- | --- |
| sciq | 0.2000 | 0.3000 | 0.1000 | Buffer 30 |
| arc_challenge | 0.0333 | 0.2667 | 0.2333 | Buffer 30 |
| boolq | 0.5000 | 0.5333 | 0.0333 | Buffer 30 |

## 11. Final Test Accuracy Comparison
| Task | No Replay | Buffer 30 | Delta | Better Setting |
| --- | --- | --- | --- | --- |
| sciq | 0.2400 | 0.3200 | 0.0800 | Buffer 30 |
| arc_challenge | 0.0000 | 0.2400 | 0.2400 | Buffer 30 |
| boolq | 0.5000 | 0.4800 | -0.0200 | No Replay |

## 12. Forgetting Analysis
| Task | No Replay Forgetting | Buffer 30 Forgetting | Lower Forgetting Setting |
| --- | --- | --- | --- |
| sciq | 0.3333 | 0.2333 | Buffer 30 |
| arc_challenge | 0.3000 | 0.0667 | Buffer 30 |
| boolq | 0.0000 | 0.0000 | Tie |
- Some forgetting was observed in the saved final validation metrics after later tasks were learned.
- Replay forgot less on the worst affected task, reducing maximum saved forgetting from 0.3333 to 0.2333.

## 13. Replay Artifacts Analysis
- Buffer 30 (`exp005_replay_buffer30_seed123_3tasks`) saved 3 replay score CSV file(s).
- Buffer 30 (`exp005_replay_buffer30_seed123_3tasks`) saved 3 replay buffer snapshot file(s).
- Buffer 30 has 3 non-empty replay buffer snapshot file(s), which suggests replay was active for this setting.
- No-replay run `exp005_no_replay_seed123_3tasks` has replay disabled.
- Its `replay_scores/` folder contains 0 replay score CSV file(s), which is expected to be zero when replay is disabled.
- Its `replay_buffers/` folder contains 3 replay buffer snapshot file(s), with 0 non-empty snapshot(s); empty files indicate placeholders rather than active replay.

## 14. Comparison with exp004
- `exp004` used seed `42`, while `exp005` uses seed `123`.
- In `exp004`, Buffer 30 was the strongest replay setting on average validation accuracy and maximum forgetting among the saved replay settings.
- In `exp005`, Buffer 30 changed average final validation accuracy by 0.1222 relative to no-replay.
- In `exp005`, Buffer 30 changed average final test accuracy by 0.1000 relative to no-replay.
- In `exp005`, maximum saved forgetting moved from 0.3333 in no-replay to 0.2333 with Buffer 30.
- Relative to the no-replay baseline, `exp005` tentatively supports the favorable Buffer 30 pattern from `exp004`, although the evidence still comes from only two seeds.
- For reference, `exp004` Buffer 30 changed average validation accuracy by 0.0222, average test accuracy by -0.0267, and maximum forgetting from 0.0667 to 0.0667 relative to its own no-replay baseline.

## 15. Main Findings
- The saved seed-robustness check covers `sciq -> arc_challenge -> boolq` at seed `123`.
- The average Buffer 30 minus No Replay final validation accuracy delta was 0.1222.
- The average Buffer 30 minus No Replay final test accuracy delta was 0.1000.
- Maximum saved forgetting changed from 0.3333 in No Replay to 0.2333 with Buffer 30.
- Under seed 123, the saved metrics provide mixed evidence rather than a fully consistent Buffer 30 advantage.

## 16. Interpretation
- At seed 123, Buffer 30 improved average final validation accuracy by 0.1222 relative to no-replay.
- At seed 123, Buffer 30 improved average final test accuracy by 0.1000 relative to no-replay.
- Buffer 30 reduced maximum saved forgetting from 0.3333 to 0.2333.
- At seed 123, the evidence is more favorable to Buffer 30 than in seed 42, because Buffer 30 improves average validation accuracy, average test accuracy, and maximum saved forgetting relative to no-replay. However, this should still be treated cautiously because one test task still favors no-replay and only one additional seed was tested.
- Any apparent stability or instability should be treated cautiously because this check still uses one additional seed, one epoch, lightweight sample sizes, and one replay configuration.

## 17. Limitations
- Only one additional seed is tested here: `123`.
- One-epoch training: `1` epoch(s) were saved in the configuration.
- Lightweight subset sizes: train=100, val=30, test=50.
- Same task order as the original-order baseline: `sciq -> arc_challenge -> boolq`.
- Only one replay buffer size is retested here, so the seed check does not cover the full replay design space.
- These saved results still provide limited evidence about robustness because they extend the study by only one seed.

## 18. Relation to Other Experiments
- `exp001` is the original-order baseline comparison.
- `exp002` tests the reverse task order.
- `exp003` is an additional baseline-style comparison.
- `exp004` tests replay buffer size.
- `exp005` checks whether the strongest saved replay-buffer setting remains stable under a different seed.

## 19. Next Step
Run the same seed robustness check with more seeds, or test replay sampling strategy.

## 20. Reproducibility Checklist
- [x] Run commands recorded
- [x] run_config.json saved
- [x] metrics saved
- [x] replay artifacts saved or clearly absent
- [x] seed recorded
- [x] summary generated from saved files
- [x] experiment_index.md updated

## 21. Notes
- Runs appear completed: Yes.
- Runtime error check: log files were not available to inspect.
- This summary was generated automatically from saved files rather than from a fresh training run.
- Summary generation timestamp: `2026-04-28T03:16:14`.
