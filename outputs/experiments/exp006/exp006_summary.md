# exp006 Summary

## 1. Experiment Overview
This experiment is a replay selection strategy comparison experiment.

## 2. Research Question
Under the same replay buffer budget, do loss-based or surprise-based replay selection strategies outperform random replay and no-replay?

## 3. Experiment Design
- No-replay baseline.
- Random replay.
- Loss-based replay.
- Surprise-based replay.
- Task sequence: `sciq -> arc_challenge -> boolq`.
- Seed: `42`.
- Replay buffer size: `30`.
- Two experiment-level reviewed-label scorers now exist:
  - the original scorer trained from `57` keep/reject labels exported to `utility_labels_reviewed.csv`
  - the expanded scorer trained from `113` keep/reject labels in the canonical expanded review file
- The expanded scorer is intended for a future exploratory `exp008` `predicted_utility` replay run; `exp008` has not been run yet.

## 4. Task and Dataset Description
- `sciq`: science question answering task.
- `arc_challenge`: challenging grade-school science question answering task.
- `boolq`: yes/no question answering task.
- Together, these tasks act as a lightweight continual learning sequence rather than a large-scale benchmark sweep.

## 5. Run Commands
- `python main.py --run-name exp006_no_replay_seed42_3tasks --tasks sciq arc_challenge boolq --no-replay --seed 42`
- `python main.py --run-name exp006_replay_random_buffer30_seed42_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 42 --replay-selection-strategy random`
- `python main.py --run-name exp006_replay_loss_buffer30_seed42_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 42 --replay-selection-strategy loss`
- `python main.py --run-name exp006_replay_surprise_buffer30_seed42_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 42 --replay-selection-strategy surprise`

## 6. Directory Structure
```text
exp006/
├── exp006_no_replay_seed42_3tasks/
├── exp006_replay_loss_buffer30_seed42_3tasks/
├── exp006_replay_random_buffer30_seed42_3tasks/
├── exp006_replay_surprise_buffer30_seed42_3tasks/
├── human_review/
│   ├── human_review_conflicts.csv
│   ├── human_review_summary.md
│   ├── expanded_human_review_summary.md
│   ├── replay_review_candidates_expanded_review_target.csv
│   ├── replay_review_candidates_reviewed_by_strategy.csv
│   └── replay_review_candidates_reviewed_fixed.csv
├── scorer_checkpoints_expanded/
│   └── scorer_model.joblib
├── scorer_outputs_expanded/
│   ├── feature_columns.json
│   ├── scorer_metrics.csv
│   ├── scorer_predictions.csv
│   └── scorer_training_summary.md
├── scorer_checkpoints/
│   └── scorer_model.joblib
├── scorer_outputs/
│   ├── feature_columns.json
│   ├── scorer_metrics.csv
│   ├── scorer_predictions.csv
│   ├── scorer_training_summary.md
│   └── scorer_training_warning.txt
├── utility_labels/
│   └── utility_labels_reviewed.csv
└── exp006_summary.md
```

## 7. Configuration Summary
| Field | No Replay | Random Replay | Loss Replay | Surprise Replay |
| --- | --- | --- | --- | --- |
| run_name | exp006_no_replay_seed42_3tasks | exp006_replay_random_buffer30_seed42_3tasks | exp006_replay_loss_buffer30_seed42_3tasks | exp006_replay_surprise_buffer30_seed42_3tasks |
| tasks | sciq -> arc_challenge -> boolq | sciq -> arc_challenge -> boolq | sciq -> arc_challenge -> boolq | sciq -> arc_challenge -> boolq |
| replay enabled | False | True | True | True |
| replay buffer size | disabled | 30 | 30 | 30 |
| replay selection strategy | disabled | random | loss | surprise |
| seed | 42 | 42 | 42 | 42 |
| model name | google/flan-t5-small | google/flan-t5-small | google/flan-t5-small | google/flan-t5-small |
| epochs | 1 | 1 | 1 | 1 |
| batch size | 2 | 2 | 2 | 2 |
| learning rate | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
| output directory | /Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp006/exp006_no_replay_seed42_3tasks | /Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp006/exp006_replay_random_buffer30_seed42_3tasks | /Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp006/exp006_replay_loss_buffer30_seed42_3tasks | /Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp006/exp006_replay_surprise_buffer30_seed42_3tasks |

## 8. Output Files and Their Purpose
- `checkpoints/`: saved model and tokenizer checkpoints after each task.
- `logs/`: training summary and step logs for each run.
- `metrics/`: validation and test metrics plus prediction files.
- `replay_scores/`: per-sample replay scoring CSV files for replay-enabled runs.
- `replay_buffers/`: replay buffer snapshots and replay buffer summary files.
- `sample_signals/`: sample-level early training signal logs.
- `utility_labels/`: proxy utility labels derived from replay selection outcomes.
- `human_review/`: auto-tagged replay review exports for manual review.
- top-level `human_review/`: combined, deduplicated, and conflict-checked human review outputs for exp006 as a whole.
- top-level `utility_labels/utility_labels_reviewed.csv`: reviewed utility-label export used to train the first scorer.
- top-level `scorer_outputs/`: experiment-level reviewed-label scorer metrics, predictions, feature metadata, warnings, and summary files.
- top-level `scorer_checkpoints/`: experiment-level saved scorer checkpoint files.
- top-level `human_review/replay_review_candidates_expanded_review_target.csv`: canonical expanded review file used for the expanded scorer retraining pass.
- top-level `scorer_outputs_expanded/`: expanded reviewed-label scorer metrics, predictions, feature metadata, and summary files.
- top-level `scorer_checkpoints_expanded/`: saved checkpoint for the expanded reviewed-label scorer.
- `run_config.json`: saved resolved configuration for each run, including buffer size, seed, and selection strategy.
- `exp006_summary.md`: this experiment-level replay strategy comparison report.

## 9. Metric Definitions
- Validation accuracy: the fraction of validation examples answered correctly after the final saved task state.
- Test accuracy: the fraction of held-out test examples answered correctly after the final saved task state.
- Forgetting: the saved drop from the best earlier validation accuracy to the later final validation accuracy for a task.
- Strategy comparison: compare saved validation accuracy, test accuracy, and forgetting across no replay, random replay, loss replay, and surprise replay.
- Delta relative to no-replay: `Strategy score - No Replay score`.
- Delta relative to random replay: `Strategy score - Random Replay score`.

## 10. Final Validation Accuracy Comparison
| Task | No Replay | Random | Loss | Surprise | Best Setting |
| --- | --- | --- | --- | --- | --- |
| sciq | 0.4000 | 0.3000 | 0.3333 | 0.3333 | No Replay |
| arc_challenge | 0.2667 | 0.3000 | 0.2667 | 0.2667 | Random |
| boolq | 0.5000 | 0.5333 | 0.5667 | 0.5667 | Tie (Loss, Surprise) |
- Average val accuracy for No Replay: 0.3889.
- Average val accuracy for Random Replay: 0.3778.
- Average val accuracy for Loss Replay: 0.3889.
- Average val accuracy for Surprise Replay: 0.3889.

## 11. Final Test Accuracy Comparison
| Task | No Replay | Random | Loss | Surprise | Best Setting |
| --- | --- | --- | --- | --- | --- |
| sciq | 0.4000 | 0.3400 | 0.3800 | 0.3800 | No Replay |
| arc_challenge | 0.1600 | 0.2800 | 0.2600 | 0.2600 | Random |
| boolq | 0.4800 | 0.4600 | 0.4600 | 0.4600 | No Replay |
- Average test accuracy for No Replay: 0.3467.
- Average test accuracy for Random Replay: 0.3600.
- Average test accuracy for Loss Replay: 0.3667.
- Average test accuracy for Surprise Replay: 0.3667.

## 12. Forgetting Analysis
| Task | No Replay | Random | Loss | Surprise | Lowest Forgetting Setting |
| --- | --- | --- | --- | --- | --- |
| sciq | 0.0000 | 0.1000 | 0.0667 | 0.0667 | No Replay |
| arc_challenge | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Tie (No Replay, Random, Loss, Surprise) |
| boolq | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Tie (No Replay, Random, Loss, Surprise) |
- Maximum forgetting for No Replay: 0.0000.
- Maximum forgetting for Random Replay: 0.1000.
- Maximum forgetting for Loss Replay: 0.0667.
- Maximum forgetting for Surprise Replay: 0.0667.

## 13. PRE-PLAY Artifact Analysis
- No Replay (`exp006_no_replay_seed42_3tasks`) has replay disabled in `run_config.json`.
- Random Replay (`exp006_replay_random_buffer30_seed42_3tasks`) saved 3 replay score CSV file(s).
- Random Replay (`exp006_replay_random_buffer30_seed42_3tasks`) saved 3 replay buffer snapshot file(s).
- Random Replay has 3 non-empty replay buffer snapshot file(s).
- Random Replay saved 3 sample_signals CSV file(s).
- Random Replay saved 3 utility_labels CSV file(s).
- Random Replay human review exports: replay_review_candidates.csv=Yes, replay_review_readme.md=Yes.
- Random Replay human review tag counts: high_value_candidate=90, selected_for_replay=0, needs_human_review=84, low_priority=126.
- Random Replay manually reviewed samples in `replay_review_candidates_reviewed.csv`: 63.
- Loss Replay (`exp006_replay_loss_buffer30_seed42_3tasks`) saved 3 replay score CSV file(s).
- Loss Replay (`exp006_replay_loss_buffer30_seed42_3tasks`) saved 3 replay buffer snapshot file(s).
- Loss Replay has 3 non-empty replay buffer snapshot file(s).
- Loss Replay saved 3 sample_signals CSV file(s).
- Loss Replay saved 3 utility_labels CSV file(s).
- Loss Replay human review exports: replay_review_candidates.csv=Yes, replay_review_readme.md=Yes.
- Loss Replay human review tag counts: high_value_candidate=90, selected_for_replay=0, needs_human_review=40, low_priority=170.
- Loss Replay manually reviewed samples in `replay_review_candidates_reviewed.csv`: 63.
- Surprise Replay (`exp006_replay_surprise_buffer30_seed42_3tasks`) saved 3 replay score CSV file(s).
- Surprise Replay (`exp006_replay_surprise_buffer30_seed42_3tasks`) saved 3 replay buffer snapshot file(s).
- Surprise Replay has 3 non-empty replay buffer snapshot file(s).
- Surprise Replay saved 3 sample_signals CSV file(s).
- Surprise Replay saved 3 utility_labels CSV file(s).
- Surprise Replay human review exports: replay_review_candidates.csv=Yes, replay_review_readme.md=Yes.
- Surprise Replay human review tag counts: high_value_candidate=90, selected_for_replay=0, needs_human_review=40, low_priority=170.
- Surprise Replay manually reviewed samples in `replay_review_candidates_reviewed.csv`: 63.
- Experiment-level human review outputs were created under `outputs/experiments/exp006/human_review/`, including a 900-row strategy-level file, a 300-row deduplicated sample-level file, and a zero-conflict conflict log.
- Reviewed labels were exported to `outputs/experiments/exp006/utility_labels/utility_labels_reviewed.csv`.
- The first reviewed-label utility scorer was trained from 57 labelled rows (`keep`=22, `reject`=35) using `human_utility_label`.
- Experiment-level scorer outputs were written to `outputs/experiments/exp006/scorer_outputs/`, and the saved model checkpoint was written to `outputs/experiments/exp006/scorer_checkpoints/scorer_model.joblib`.
- The canonical expanded review file now contains 132 reviewed rows, including 113 usable `keep` / `reject` labels (`keep`=53, `reject`=60, `unsure`=19).
- An expanded reviewed-label utility scorer was retrained from `replay_review_candidates_expanded_review_target.csv` using `final_replay_decision` and `review_priority`, with outputs saved under `scorer_outputs_expanded/` and `scorer_checkpoints_expanded/`.
- Expanded scorer metrics were: accuracy `0.7353`, precision `0.7333`, recall `0.6875`, F1 `0.7097`, ROC AUC `0.7778`.

## 14. Replay Selection Strategy Comparison
- Random Replay versus Loss Replay: average validation accuracy 0.3778 vs 0.3889, average test accuracy 0.3600 vs 0.3667, maximum forgetting 0.1000 vs 0.0667.
- Random Replay versus Surprise Replay: average validation accuracy 0.3778 vs 0.3889, average test accuracy 0.3600 vs 0.3667, maximum forgetting 0.1000 vs 0.0667.
- Loss Replay versus Surprise Replay: average validation accuracy 0.3889 vs 0.3889, average test accuracy 0.3667 vs 0.3667, maximum forgetting 0.0667 vs 0.0667.
- Random Replay versus No Replay: average validation delta -0.0111, average test delta 0.0133, maximum forgetting 0.1000 versus 0.0000.
- Loss Replay versus No Replay: average validation delta -0.0000, average test delta 0.0200, maximum forgetting 0.0667 versus 0.0000.
- Surprise Replay versus No Replay: average validation delta -0.0000, average test delta 0.0200, maximum forgetting 0.0667 versus 0.0000.
- These comparisons use saved metrics only and should be read cautiously under this lightweight setting.

## 15. Main Findings
- The saved strategy comparison covers `sciq -> arc_challenge -> boolq` with one no-replay baseline and three replay selection strategies at buffer size `30` and seed `42`.
- The best average validation setting was Tie (No Replay, Loss Replay, Surprise Replay).
- The best average test setting was Tie (Loss Replay, Surprise Replay).
- The setting with the lowest maximum forgetting was No Replay.
- Replay strategies with positive average test deltas versus no-replay were: Random Replay, Loss Replay, Surprise Replay.
- Both scoring-based replay strategies outperform random replay on average test accuracy, which is preliminary evidence that replay scoring may matter.
- PRE-PLAY artifact generation succeeded for the replay-enabled runs, and human-review-ready candidate files were present.
- The reviewed-label utility scorer used `LogisticRegression` with 57 labelled rows and reached accuracy `0.8333`, precision `0.7500`, recall `0.8571`, F1 `0.8000`, and ROC AUC `0.8052`.
- The expanded scorer also used `LogisticRegression`, but with 113 labelled rows and lower held-out metrics: accuracy `0.7353`, precision `0.7333`, recall `0.6875`, F1 `0.7097`, and ROC AUC `0.7778`.
- The current folder structure is valid because `scorer_outputs/` and `scorer_checkpoints/` are experiment-level PRE-PLAY outputs, not run-level outputs.

## 16. Interpretation
- Because at least one scoring-based replay strategy outperforms random replay on the saved metrics, exp006 provides preliminary evidence that replay selection by score may matter.
- On average final test accuracy, the strongest saved setting was Tie (Loss Replay, Surprise Replay), but this should not be over-interpreted from one seed and one epoch.
- The PRE-PLAY direction is still supported operationally because sample_signals, utility_labels, and human_review exports were generated successfully for replay-enabled runs.
- The initial 57-label scorer was sufficient for the exploratory exp007 `predicted_utility` run.
- The expanded scorer increases label coverage from 57 to 113 usable labels, but its current held-out metrics are lower, so it is operationally ready for exp008 while still needing cautious interpretation.

## 17. Limitations
- One seed only: `42`.
- One-epoch training: `1` epoch(s) were saved in the configuration.
- Lightweight subset sizes: train=100, val=30, test=50.
- Only three tasks are used here: `sciq -> arc_challenge -> boolq`.
- `predicted_utility` replay is outside exp006 itself; it was first exercised later in exp007.
- Utility labels are proxy labels rather than exact causal future replay utility.
- Human review is still limited in scope even after expansion: the canonical expanded review file has 132 reviewed rows and 113 usable `keep` / `reject` labels.
- The expanded scorer did not improve held-out metrics over the smaller initial scorer, which suggests either split sensitivity or remaining label/data limitations.

## 18. Relation to Other Experiments
- `exp001` is the original-order baseline comparison.
- `exp002` tests the reverse task order.
- `exp003` is an additional baseline-style comparison.
- `exp004` tests replay buffer size.
- `exp005` checks whether the strongest saved replay-buffer setting remains stable under a different seed.
- `exp006` compares replay selection strategies after the PRE-PLAY core infrastructure was added and now also contains the expanded human-review retraining outputs for future replay scoring work.

## 19. Next Step
Use the expanded reviewed-label scorer for an exploratory exp008 `predicted_utility` replay test, while treating the result cautiously because the held-out scorer metrics remain unstable under this lightweight setup.

## 20. Reproducibility Checklist
- [x] Run commands recorded
- [x] run_config.json saved
- [x] metrics saved
- [x] replay artifacts saved or clearly absent
- [x] sample_signals saved
- [x] utility_labels saved
- [x] human_review candidates exported
- [x] reviewed utility labels exported
- [x] scorer outputs saved
- [x] replay_selection_strategy recorded
- [x] experiment_index.md updated

## 21. Notes
- Runs appear completed: Yes.
- Runtime errors found in checked logs: No.
- Summary generation timestamp: `2026-05-09`.
- No `exp001` to `exp005` folders were modified by this experiment.
