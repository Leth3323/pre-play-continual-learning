# exp004 Summary

## 1. Experiment Overview
This experiment is best interpreted as a replay buffer size ablation experiment over the task sequence sciq -> arc_challenge -> boolq.

## 2. Research Question
How does replay buffer size affect final performance and forgetting in the three-task continual learning sequence?

## 3. Experiment Design
- No-replay baseline: compare against training without replay.
- Buffer 30: replay enabled in `exp004_replay_buffer30_3tasks` with replay buffer size `30`.
- Buffer 60: replay enabled in `exp004_replay_buffer60_3tasks` with replay buffer size `60`.
- Buffer 100: replay enabled in `exp004_replay_buffer100_3tasks` with replay buffer size `100`.
- Task sequence: `sciq -> arc_challenge -> boolq`.
- Comparison target: final validation accuracy, final test accuracy, and saved forgetting values across all four settings.
- Why this comparison is useful: it isolates replay buffer capacity while keeping the task sequence and the rest of the saved training setup aligned.

## 4. Task and Dataset Description
- `sciq`: science question answering task.
- `arc_challenge`: challenging grade-school science question answering task.
- `boolq`: yes/no question answering task.
- Together, these tasks act as a lightweight continual learning sequence rather than a large-scale benchmark sweep.

## 5. Run Commands
- No Replay: `python main.py --run-name exp004_no_replay_3tasks --tasks sciq arc_challenge boolq --no-replay` (reconstructed from saved `run_config.json` because the exact invocation string was not persisted).
- Buffer 30: `python main.py --run-name exp004_replay_buffer30_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30` (reconstructed from saved `run_config.json` because the exact invocation string was not persisted).
- Buffer 60: `python main.py --run-name exp004_replay_buffer60_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 60` (reconstructed from saved `run_config.json` because the exact invocation string was not persisted).
- Buffer 100: `python main.py --run-name exp004_replay_buffer100_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 100` (reconstructed from saved `run_config.json` because the exact invocation string was not persisted).

## 6. Directory Structure
```text
exp004/
├── exp004_no_replay_3tasks/
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
├── exp004_replay_buffer100_3tasks/
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
├── exp004_replay_buffer30_3tasks/
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
├── exp004_replay_buffer60_3tasks/
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
└── exp004_summary.md
```

## 7. Configuration Summary
| Field | No Replay | Buffer 30 | Buffer 60 | Buffer 100 |
| --- | --- | --- | --- | --- |
| run_name | exp004_no_replay_3tasks | exp004_replay_buffer30_3tasks | exp004_replay_buffer60_3tasks | exp004_replay_buffer100_3tasks |
| tasks | sciq -> arc_challenge -> boolq | sciq -> arc_challenge -> boolq | sciq -> arc_challenge -> boolq | sciq -> arc_challenge -> boolq |
| replay enabled | False | True | True | True |
| replay buffer size | 100 | 30 | 60 | 100 |
| model name | google/flan-t5-small | google/flan-t5-small | google/flan-t5-small | google/flan-t5-small |
| epochs | 1 | 1 | 1 | 1 |
| batch size | 2 | 2 | 2 | 2 |
| learning rate | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
| seed | 42 | 42 | 42 | 42 |
| output directory | /Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp004/exp004_no_replay_3tasks | /Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp004/exp004_replay_buffer30_3tasks | /Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp004/exp004_replay_buffer60_3tasks | /Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp004/exp004_replay_buffer100_3tasks |

## 8. Output Files and Their Purpose
- `checkpoints/`: saved model and tokenizer checkpoints after each task, useful for inspecting intermediate training states.
- `logs/`: console output plus training summary CSVs, useful for checking whether runs completed and how many training steps were recorded.
- `metrics/`: validation and test metric CSVs plus prediction files, which are the main source for the comparisons in this report.
- `replay_scores/`: per-sample replay scoring CSVs used only when replay is enabled; for no-replay runs this folder may be empty.
- `replay_buffers/`: replay buffer snapshots and the replay buffer summary CSV; for no-replay runs these files can be empty placeholders.
- `run_config.json`: the saved resolved configuration for each run folder, including task order and output paths.
- `exp004_summary.md`: this experiment-level report generated from the saved files in this experiment directory.

## 9. Metric Definitions
- Validation accuracy: the fraction of validation examples answered correctly for a task after training has reached the final saved task state.
- Test accuracy: the fraction of held-out test examples answered correctly for a task after the final saved task state.
- Forgetting: a conceptual measure of how much a task's performance drops after learning later tasks compared with its best earlier saved performance. The code saves this value directly, so this report explains it conceptually rather than inventing a new formula.
- Delta between replay and no-replay: `Replay score - No Replay score`. A positive delta means replay performed better, while a negative delta means the no-replay run performed better.

## 10. Final Validation Accuracy Comparison
| Task | No Replay | Buffer 30 | Buffer 60 | Buffer 100 | Best Setting |
| --- | --- | --- | --- | --- | --- |
| sciq | 0.3333 | 0.3333 | 0.3000 | 0.3000 | Tie (No Replay, Buffer 30) |
| arc_challenge | 0.2333 | 0.2667 | 0.2000 | 0.2000 | Buffer 30 |
| boolq | 0.5333 | 0.5667 | 0.5333 | 0.5333 | Buffer 30 |

## 11. Final Test Accuracy Comparison
| Task | No Replay | Buffer 30 | Buffer 60 | Buffer 100 | Best Setting |
| --- | --- | --- | --- | --- | --- |
| sciq | 0.5000 | 0.3800 | 0.3200 | 0.3200 | No Replay |
| arc_challenge | 0.2200 | 0.2600 | 0.2800 | 0.2800 | Tie (Buffer 60, Buffer 100) |
| boolq | 0.4600 | 0.4600 | 0.5000 | 0.5000 | Tie (Buffer 60, Buffer 100) |

## 12. Forgetting Analysis
| Task | No Replay | Buffer 30 | Buffer 60 | Buffer 100 | Lowest Forgetting Setting |
| --- | --- | --- | --- | --- | --- |
| sciq | 0.0667 | 0.0667 | 0.1000 | 0.1000 | Tie (No Replay, Buffer 30) |
| arc_challenge | 0.0667 | 0.0000 | 0.0333 | 0.0333 | Buffer 30 |
| boolq | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Tie (No Replay, Buffer 30, Buffer 60, Buffer 100) |
- Some forgetting was observed in the saved final validation metrics after later tasks were learned.
- The lowest maximum forgetting was achieved by Tie (No Replay, Buffer 30).
- Replay did not consistently reduce forgetting relative to the no-replay baseline.

## 13. Replay Artifacts Analysis
- No Replay (`exp004_no_replay_3tasks`) has replay disabled.
- Its `replay_scores/` folder contains 0 replay score CSV file(s), which is expected to be zero when replay is disabled.
- Its `replay_buffers/` folder contains 3 replay buffer snapshot file(s), with 0 non-empty snapshot(s); empty files indicate placeholders rather than active replay.
- Buffer 30 (`exp004_replay_buffer30_3tasks`) saved 3 replay score CSV file(s).
- Buffer 30 (`exp004_replay_buffer30_3tasks`) saved 3 replay buffer snapshot file(s).
- Buffer 30 has 3 non-empty replay buffer snapshot file(s), which suggests replay was active for this setting.
- Buffer 60 (`exp004_replay_buffer60_3tasks`) saved 3 replay score CSV file(s).
- Buffer 60 (`exp004_replay_buffer60_3tasks`) saved 3 replay buffer snapshot file(s).
- Buffer 60 has 3 non-empty replay buffer snapshot file(s), which suggests replay was active for this setting.
- Buffer 100 (`exp004_replay_buffer100_3tasks`) saved 3 replay score CSV file(s).
- Buffer 100 (`exp004_replay_buffer100_3tasks`) saved 3 replay buffer snapshot file(s).
- Buffer 100 has 3 non-empty replay buffer snapshot file(s), which suggests replay was active for this setting.

## 14. Main Findings
- The saved ablation covers the task sequence `sciq -> arc_challenge -> boolq` across one no-replay baseline and three replay buffer sizes.
- Among replay settings, the best average validation accuracy came from Buffer 30.
- Among replay settings, the best average test accuracy came from Tie (Buffer 30, Buffer 60, Buffer 100).
- Among replay settings, the lowest maximum forgetting came from Buffer 30.
- Larger replay buffers did not show a consistent monotonic benefit across the saved metrics.

## 15. Interpretation
- The strongest replay setting on average validation accuracy was Buffer 30.
- The strongest replay setting on average test accuracy was Tie (Buffer 30, Buffer 60, Buffer 100).
- The setting with the lowest maximum saved forgetting was Buffer 30.
- No replay buffer size is consistently better than the no-replay baseline on validation accuracy, test accuracy, and forgetting at the same time, so the evidence remains mixed.
- The effect of buffer size is unclear in this saved sweep because the best setting changes across metrics and larger buffers do not help monotonically.
- Mixed results are plausible here because the setup still uses one seed, one epoch, lightweight sample sizes, and fixed replay settings apart from buffer size.

## 16. Limitations
- Single random seed: `42`.
- One-epoch training: `1` epoch(s) were saved in the configuration.
- Lightweight subset sizes: train=100, val=30, test=50.
- Limited task count in this experiment: `3` task(s).
- Only three replay buffer sizes are represented, and other replay hyperparameters are not explored.
- These results come from a lightweight saved setting and may not generalize without more seeds, task orders, or larger sample sizes.

## 17. Relation to Other Experiments
- `exp001` is the original-order baseline comparison.
- `exp002` tests the reverse task order.
- `exp003` is an additional baseline-style comparison.
- `exp004` tests replay buffer size.

## 18. Next Step
Run the best-performing buffer-size setting across another random seed, or test replay sampling strategy to see whether the result is stable.

## 19. Reproducibility Checklist
- [x] Run commands recorded
- [x] run_config.json saved
- [x] metrics saved
- [x] replay artifacts saved or clearly absent
- [x] summary generated from saved files
- [x] experiment_index.md updated

## 20. Notes
- Runs appear completed: Yes.
- Runtime error check: log files were not available to inspect.
- This summary was generated automatically from saved files rather than from a fresh training run.
- Summary generation timestamp: `2026-04-28T02:54:45`.
