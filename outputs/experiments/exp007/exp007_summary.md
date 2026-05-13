# exp007 Summary

## 1. Experiment Overview
This experiment is an exploratory `predicted_utility` replay experiment. It compares no replay, random replay, surprise replay, and `predicted_utility` replay over the task sequence `sciq -> arc_challenge -> boolq`.

## 2. Research Question
Can a human-review-trained utility scorer improve replay selection compared with random and surprise-based replay?

## 3. Experiment Design
- No-replay baseline.
- Random replay.
- Surprise replay.
- Predicted utility replay.
- Task sequence: `sciq -> arc_challenge -> boolq`.
- Seed: `42`.
- Replay buffer size: `30`.
- Scorer source: the exp006 scorer trained from human-reviewed labels in `outputs/experiments/exp006/utility_labels/utility_labels_reviewed.csv`.
- Exploratory limitation: the scorer was trained on only `57` labelled rows (`keep`=`22`, `reject`=`35`).

## 4. Task and Dataset Description
- `sciq`: multiple-choice science question answering.
- `arc_challenge`: harder grade-school science multiple-choice question answering.
- `boolq`: passage-grounded yes/no question answering.
- Together these form a lightweight continual-learning sequence rather than a large benchmark sweep.

## 5. Run Commands
- `python main.py --run-name exp007_no_replay_seed42_3tasks --tasks sciq arc_challenge boolq --no-replay --seed 42`
- `python main.py --run-name exp007_replay_random_buffer30_seed42_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 42 --replay-selection-strategy random`
- `python main.py --run-name exp007_replay_surprise_buffer30_seed42_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 42 --replay-selection-strategy surprise`
- `python main.py --run-name exp007_replay_predicted_utility_buffer30_seed42_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 42 --replay-selection-strategy predicted_utility`

## 6. Directory Structure
```text
exp007/
├── exp007_no_replay_seed42_3tasks/
├── exp007_replay_random_buffer30_seed42_3tasks/
├── exp007_replay_surprise_buffer30_seed42_3tasks/
├── exp007_replay_predicted_utility_buffer30_seed42_3tasks/
└── exp007_summary.md
```

## 7. Configuration Summary
| Field | No Replay | Random | Surprise | Predicted Utility |
| --- | --- | --- | --- | --- |
| run_name | exp007_no_replay_seed42_3tasks | exp007_replay_random_buffer30_seed42_3tasks | exp007_replay_surprise_buffer30_seed42_3tasks | exp007_replay_predicted_utility_buffer30_seed42_3tasks |
| tasks | sciq -> arc_challenge -> boolq | sciq -> arc_challenge -> boolq | sciq -> arc_challenge -> boolq | sciq -> arc_challenge -> boolq |
| replay enabled | False | True | True | True |
| replay buffer size | disabled | 30 | 30 | 30 |
| replay selection strategy | disabled | random | surprise | predicted_utility |
| seed | 42 | 42 | 42 | 42 |
| model name | google/flan-t5-small | google/flan-t5-small | google/flan-t5-small | google/flan-t5-small |
| epochs | 1 | 1 | 1 | 1 |
| batch size | 2 | 2 | 2 | 2 |
| learning rate | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
| scorer source if available | n/a | n/a | n/a | exp006 reviewed-label scorer (`scorer_model.joblib` + `feature_columns.json`) |
| output directory | outputs/experiments/exp007/exp007_no_replay_seed42_3tasks | outputs/experiments/exp007/exp007_replay_random_buffer30_seed42_3tasks | outputs/experiments/exp007/exp007_replay_surprise_buffer30_seed42_3tasks | outputs/experiments/exp007/exp007_replay_predicted_utility_buffer30_seed42_3tasks |

## 8. Output Files and Their Purpose
- `checkpoints/`: saved model and tokenizer checkpoints after each task.
- `logs/`: training summary and step-level training logs.
- `metrics/`: validation and test metrics plus prediction CSVs.
- `replay_scores/`: per-sample replay scoring exports for replay-enabled runs.
- `replay_buffers/`: replay buffer snapshots and replay buffer summary files.
- `sample_signals/`: early sample-level training signal exports.
- `utility_labels/`: proxy replay-utility label exports.
- `human_review/`: replay candidate review exports for manual inspection.
- `run_config.json`: resolved run configuration, including strategy, buffer size, seed, and output paths.
- `exp007_summary.md`: experiment-level report for exp007.

## 9. Metric Definitions
- Validation accuracy: fraction of validation examples answered correctly after the final saved task state.
- Test accuracy: fraction of held-out test examples answered correctly after the final saved task state.
- Forgetting: saved drop from a task's best earlier validation accuracy to its later validation accuracy.
- Delta vs no-replay: `Setting score - No Replay score`.
- Delta vs random: `Setting score - Random score`.

## 10. Final Validation Accuracy Comparison
| Task | No Replay | Random | Surprise | Predicted Utility | Best Setting |
| --- | --- | --- | --- | --- | --- |
| sciq | 0.4000 | 0.3000 | 0.3333 | 0.4333 | Predicted Utility |
| arc_challenge | 0.2333 | 0.3000 | 0.2667 | 0.2667 | Random |
| boolq | 0.5333 | 0.5667 | 0.5667 | 0.5333 | Tie (Random, Surprise) |
- Average validation accuracy for No Replay: 0.3889.
- Average validation accuracy for Random: 0.3889.
- Average validation accuracy for Surprise: 0.3889.
- Average validation accuracy for Predicted Utility: 0.4111.

## 11. Final Test Accuracy Comparison
| Task | No Replay | Random | Surprise | Predicted Utility | Best Setting |
| --- | --- | --- | --- | --- | --- |
| sciq | 0.4000 | 0.3400 | 0.3800 | 0.2800 | No Replay |
| arc_challenge | 0.1600 | 0.2600 | 0.2600 | 0.2400 | Tie (Random, Surprise) |
| boolq | 0.4800 | 0.4600 | 0.4600 | 0.4800 | Tie (No Replay, Predicted Utility) |
- Average test accuracy for No Replay: 0.3467.
- Average test accuracy for Random: 0.3533.
- Average test accuracy for Surprise: 0.3667.
- Average test accuracy for Predicted Utility: 0.3333.

## 12. Forgetting Analysis
| Task | No Replay Forgetting | Random Forgetting | Surprise Forgetting | Predicted Utility Forgetting | Lowest Forgetting Setting |
| --- | --- | --- | --- | --- | --- |
| sciq | 0.0000 | 0.1000 | 0.0667 | 0.0000 | Tie (No Replay, Predicted Utility) |
| arc_challenge | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Tie (No Replay, Random, Surprise, Predicted Utility) |
| boolq | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Tie (No Replay, Random, Surprise, Predicted Utility) |
- Maximum forgetting for No Replay: 0.0000.
- Maximum forgetting for Random: 0.1000.
- Maximum forgetting for Surprise: 0.0667.
- Maximum forgetting for Predicted Utility: 0.0000.

## 13. PRE-PLAY Artifact Analysis
- Random replay: replay score CSV count `3`, replay buffer snapshot count `3`, non-empty replay buffer snapshots `3`, sample_signals CSV count `3`, utility_labels CSV count `3`, human_review candidate file generated `True`.
- Surprise replay: replay score CSV count `3`, replay buffer snapshot count `3`, non-empty replay buffer snapshots `3`, sample_signals CSV count `3`, utility_labels CSV count `3`, human_review candidate file generated `True`.
- Predicted utility replay: replay score CSV count `3`, replay buffer snapshot count `3`, non-empty replay buffer snapshots `3`, sample_signals CSV count `3`, utility_labels CSV count `3`, human_review candidate file generated `True`.
- Predicted utility scorer model loaded from exp006: `True`.
- Predicted utility feature columns loaded from exp006: `True`.
- Predicted utility scores were used: `True`.
- No-replay baseline still saved sample_signals, utility_labels, and human_review scaffolding, but it saved zero replay score CSVs and only empty replay buffer snapshots.

## 14. Predicted Utility Strategy Analysis
- Predicted utility vs random: average validation accuracy `0.4111` vs `0.3889` (delta `0.0222`), average test accuracy `0.3333` vs `0.3533` (delta `-0.0200`), maximum forgetting `0.0000` vs `0.1000`.
- Predicted utility vs surprise: average validation accuracy `0.4111` vs `0.3889` (delta `0.0222`), average test accuracy `0.3333` vs `0.3667` (delta `-0.0333`), maximum forgetting `0.0000` vs `0.0667`.
- Predicted utility vs no-replay: average validation accuracy `0.4111` vs `0.3889` (delta `0.0222`), average test accuracy `0.3333` vs `0.3467` (delta `-0.0133`), maximum forgetting `0.0000` vs `0.0000`.
- These comparisons use saved metrics only.

## Predicted Utility Diagnostic Summary
- The predicted-utility run loaded the exp006 scorer from `outputs/experiments/exp006/scorer_checkpoints/scorer_model.joblib` and `outputs/experiments/exp006/scorer_outputs/feature_columns.json`.
- No silent fallback was detected: the saved selection files record `predicted_utility`, the selected set overlaps only `29 / 90` with random, and the predicted score column is distinct from both raw `loss` and `surprise_score`.
- Overlap with surprise is still strong at `75 / 90`, so the current scorer is modifying surprise-like behavior more than replacing it entirely.
- Among the `90` selected samples, `17` were exp006 reviewed `keep`, `14` were `reject`, `2` were `unsure`, and `57` were not reviewed.
- The detailed diagnostic report is saved at `outputs/experiments/exp007/exp007_predicted_utility_diagnostics.md`.

## 15. Main Findings
- All four exp007 runs completed and saved their expected run directories under `outputs/experiments/exp007/`.
- The predicted-utility run is valid: `run_config.json` records `predicted_utility`, buffer size `30`, seed `42`, and exp006 scorer paths.
- The predicted-utility selection-comparison files contain only `predicted_utility` as the strategy label, and their selected sample sets differ from both random and surprise, so there is no sign of silent fallback.
- Predicted utility achieved the best average final validation accuracy at 0.4111.
- Surprise replay achieved the best average final test accuracy at 0.3667.
- All settings end with zero saved final-task forgetting in `val_metrics.csv`, so the saved final forgetting comparison is a tie.
- Predicted utility improves average validation accuracy over no replay, random, and surprise, but it is below random and surprise on average test accuracy and below no replay on average test accuracy.
- This run validates the end-to-end PRE-PLAY scorer-loading pipeline, but it does not show a performance win for the current scorer.

## 16. Interpretation
This exp007 run does not provide preliminary support that the current `predicted_utility` scorer improves replay selection under this lightweight setup. The scorer loaded correctly and changed which samples were selected, so the PRE-PLAY pipeline is operational. However, predicted utility only leads on saved average validation accuracy; it does not beat random or surprise on saved average test accuracy, and it does not beat surprise on any of the test-side summary metrics. Under this exploratory setting, the current scorer is not yet strong enough to justify a stronger claim.

## 17. Limitations
- One seed only: `42`.
- One training epoch only.
- Small train/val/test subsets: `100 / 30 / 50`.
- The scorer was trained on only `57` labelled rows.
- The human review set is still preliminary and small.
- Utility labels are proxy or human-reviewed labels rather than exact causal future utility labels.
- `predicted_utility` was tested only in this exploratory configuration.

## 18. Relation to Other Experiments
- `exp001`: original-order baseline replay vs no-replay comparison.
- `exp002`: task-order sensitivity under reversed ordering.
- `exp003`: additional baseline-style replay comparison.
- `exp004`: replay buffer size ablation.
- `exp005`: seed robustness check.
- `exp006`: replay selection strategy comparison plus human review and scorer training.
- `exp007`: first `predicted_utility` replay experiment using the exp006 reviewed-label scorer.

## 19. Next Step
Expand human review labels and rerun `predicted_utility` with more seeds, or compare `predicted_utility` against loss and surprise under the same scorer setup.

## 20. Reproducibility Checklist
- [x] Run commands recorded
- [x] run_config.json saved
- [x] metrics saved
- [x] replay artifacts saved
- [x] sample_signals saved
- [x] utility_labels saved
- [x] scorer source recorded
- [x] experiment_index.md updated

## 21. Notes
- Completion status: completed.
- Compile check passed before running exp007.
- Predicted utility scorer source: `outputs/experiments/exp006/scorer_checkpoints/scorer_model.joblib` and `outputs/experiments/exp006/scorer_outputs/feature_columns.json`.
- Warning: the scorer remains exploratory because it was trained from only 57 labelled rows.
- Note: the no-replay run still stores default replay fields in `run_config.json`, but replay was disabled and no replay scoring was used.
- Errors: none.
- Timestamp: `2026-05-07 21:21:02`.
