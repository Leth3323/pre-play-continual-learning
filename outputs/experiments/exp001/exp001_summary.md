# exp001 Summary

## 1. Experiment Overview
This experiment is best interpreted as a baseline continual learning experiment with a replay versus no-replay comparison. It compares the saved runs over the task sequence `sciq -> arc_challenge -> boolq`.

## 2. Research Question
Does replay improve retention and final performance when the model is trained sequentially on `sciq -> arc_challenge -> boolq`?

## 3. Experiment Design
- Replay run: `exp001_replay_3tasks` with replay enabled on the task sequence `sciq -> arc_challenge -> boolq`.
- No-replay run: `exp001_no_replay_3tasks` with replay disabled on the same task sequence.
- Task sequence: `sciq -> arc_challenge -> boolq`.
- Comparison target: final validation accuracy, final test accuracy, and saved forgetting values.
- Why this comparison is useful: it isolates the effect of replay while keeping the task order and general training setup aligned across conditions.

## 4. Task and Dataset Description
- `sciq`: science question answering task.
- `arc_challenge`: challenging grade-school science question answering task.
- `boolq`: yes/no question answering task.
- Together, these tasks act as a lightweight continual learning sequence rather than a large-scale benchmark sweep.

## 5. Run Commands
- Replay run command (reconstructed from saved `run_config.json` because the exact invocation string was not persisted): `python main.py --run-name exp001_replay_3tasks --tasks sciq arc_challenge boolq`
- No-replay run command (reconstructed from saved `run_config.json` because the exact invocation string was not persisted): `python main.py --run-name exp001_no_replay_3tasks --tasks sciq arc_challenge boolq --no-replay`

## 6. Directory Structure
This directory listing is retained mainly for reproducibility and appendix reference. It shows where the saved run artifacts live, but the metric tables below are the primary source for interpreting the experiment results.

```text
exp001/
├── exp001_no_replay_3tasks/
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
├── exp001_replay_3tasks/
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
└── exp001_summary.md
```

## 7. Configuration Summary
| Field | Replay Run | No-Replay Run |
| --- | --- | --- |
| run_name | exp001_replay_3tasks | exp001_no_replay_3tasks |
| tasks | sciq -> arc_challenge -> boolq | sciq -> arc_challenge -> boolq |
| replay enabled | True | False |
| model name | google/flan-t5-small | google/flan-t5-small |
| epochs | 1 | 1 |
| batch size | 2 | 2 |
| learning rate | 0.0005 | 0.0005 |
| seed | 42 | 42 |
| output directory | /Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp001/exp001_replay_3tasks | /Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp001/exp001_no_replay_3tasks |

## 8. Output Files and Their Purpose
- `checkpoints/`: saved model and tokenizer checkpoints after each task, useful for inspecting intermediate training states.
- `logs/`: console output plus training summary CSVs, useful for checking whether runs completed and how many training steps were recorded.
- `metrics/`: validation and test metric CSVs plus prediction files, which are the main source for the comparisons in this report.
- `replay_scores/`: per-sample replay scoring CSVs used only when replay is enabled; for no-replay runs this folder may be empty.
- `replay_buffers/`: replay buffer snapshots and the replay buffer summary CSV; in no-replay runs these files are empty placeholders created for a consistent output layout and do not indicate that replay was used.
- `run_config.json`: the saved resolved configuration for each run folder, including task order and output paths.
- `exp001_summary.md`: this experiment-level report generated from the saved files in both run folders.

## 9. Metric Definitions
- Validation accuracy: the fraction of validation examples answered correctly for a task after training has reached the final saved task state.
- Test accuracy: the fraction of held-out test examples answered correctly for a task after the final saved task state.
- Forgetting: a task-level retention metric. Forgetting for a task is computed as the difference between the best validation accuracy observed for that task during sequential training and the final validation accuracy after all tasks are learned.
- Delta between replay and no-replay: `Replay score - No Replay score`. A positive delta means replay performed better, while a negative delta means the no-replay run performed better.

## 10. Final Validation Accuracy Comparison
| Task | Replay | No Replay | Delta | Better Setting |
| --- | --- | --- | --- | --- |
| sciq | 0.3000 | 0.3333 | -0.0333 | No Replay |
| arc_challenge | 0.2000 | 0.2000 | 0.0000 | Tie |
| boolq | 0.5333 | 0.5333 | 0.0000 | Tie |

## 11. Final Test Accuracy Comparison
| Task | Replay | No Replay | Delta | Better Setting |
| --- | --- | --- | --- | --- |
| sciq | 0.3200 | 0.5000 | -0.1800 | No Replay |
| arc_challenge | 0.2600 | 0.2200 | 0.0400 | Replay |
| boolq | 0.5000 | 0.4600 | 0.0400 | Replay |

## 12. Forgetting Analysis
| Task | Replay Forgetting | No Replay Forgetting | Lower Forgetting Setting |
| --- | --- | --- | --- |
| sciq | 0.1000 | 0.0667 | No Replay |
| arc_challenge | 0.0333 | 0.1000 | Replay |
| boolq | 0.0000 | 0.0000 | Tie |
- Some forgetting was observed in the saved final validation metrics after later tasks were learned.
- The maximum saved forgetting was identical in both settings at 0.1000.

## 13. Replay Artifacts Analysis
- Replay run `exp001_replay_3tasks` saved 3 replay score CSV file(s).
- Replay run `exp001_replay_3tasks` saved 3 replay buffer snapshot file(s).
- 3 of those replay buffer snapshot file(s) are non-empty, which suggests replay was actively selecting and storing samples.
- No-replay run `exp001_no_replay_3tasks` has replay disabled by configuration.
- Its `replay_scores/` folder contains 0 replay score CSV file(s), which is expected to be zero when replay is disabled.
- Its `replay_buffers/` folder contains 3 replay buffer snapshot file(s), with 0 non-empty snapshot(s); empty files indicate placeholders rather than active replay.
- These empty no-replay buffer files are output-layout placeholders only; they do not mean the no-replay run used replay samples during training.

## 14. Main Findings
- The saved comparison covers the task sequence `sciq -> arc_challenge -> boolq`.
- The average replay-minus-no-replay final validation accuracy delta was -0.0111.
- The average replay-minus-no-replay final test accuracy delta was -0.0333.
- Some forgetting was observed in the saved final validation metrics after later tasks were learned.
- Replay artifacts were actually produced for the replay run (3 score file(s), 3 non-empty buffer snapshot(s)), so replay appears to have been active.

## 15. Interpretation
- Replay reduced final validation accuracy on average by 0.0111 across the saved tasks.
- Replay reduced final test accuracy on average by 0.0333 across the saved tasks.
- The maximum saved forgetting was identical in both settings at 0.1000.
- Overall, the saved comparison provides mixed evidence rather than a consistent case that replay is beneficial under this setting.
- Mixed results are plausible here because the setup uses one seed, one epoch, lightweight sample sizes, and a fixed replay configuration rather than a tuned sweep.

## 16. Limitations
- Single random seed: `42`.
- One-epoch training: `1` epoch(s) were saved in the configuration.
- Lightweight subset sizes: train=100, val=30, test=50.
- Limited task count in this experiment: `3` task(s).
- Only one replay configuration is represented in the saved files, so replay hyperparameters were not broadly explored here.
- These results come from a lightweight saved setting and may not generalize without more seeds, task orders, or larger sample sizes.

## 17. Relation to Other Experiments
- `exp001` uses a `3`-task saved sequence: `sciq -> arc_challenge -> boolq`.
- `exp002` uses a `3`-task saved sequence: `boolq -> arc_challenge -> sciq`.
- `exp003` uses a `3`-task saved sequence: `sciq -> arc_challenge -> boolq`.

## 18. Next Step
The immediate follow-up proposed here has since been completed: `exp002` ran the replay vs no-replay pair with the task order reversed to test task-order sensitivity. Later experiments also expanded the comparison, including random replay baseline runs such as `exp006` and the dedicated no-replay vs scored replay vs random replay comparison in `exp010`.

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
- Summary generation timestamp: `2026-04-28T02:25:09`.
