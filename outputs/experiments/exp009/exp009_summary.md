# exp009 Summary

## 1. Experiment Overview
`exp009` is a predicted_utility robustness check using seed `123`. It repeats the scorer-source comparison from `exp008` under a new seed and keeps the task order and replay buffer size fixed.

## 2. Research Question
"Are the exp008 predicted_utility scorer-source results stable when the random seed changes from 42 to 123?"

## 3. Experiment Design
- No-replay baseline: `exp009_no_replay_seed123_3tasks`
- Random replay: `exp009_replay_random_buffer30_seed123_3tasks`
- Surprise replay: `exp009_replay_surprise_buffer30_seed123_3tasks`
- Predicted utility with previous scorer: `exp009_replay_predicted_utility_prev_scorer_buffer30_seed123_3tasks`
- Predicted utility with expanded scorer: `exp009_replay_predicted_utility_expanded_scorer_buffer30_seed123_3tasks`
- Task sequence: `sciq -> arc_challenge -> boolq`
- Seed: `123`
- Replay buffer size: `30`

## 4. Scorer Sources
| Scorer Source | Label Count | Keep | Reject | Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Previous scorer | 57 | 22 | 35 | LogisticRegression | 0.8333 | 0.7500 | 0.8571 | 0.8000 | 0.8052 |
| Expanded scorer | 113 | 53 | 60 | LogisticRegression | 0.7353 | 0.7333 | 0.6875 | 0.7097 | 0.7778 |

The previous scorer still has the stronger held-out scorer metrics from `exp006`, while the expanded scorer uses more reviewed labels but weaker saved held-out metrics.

## 5. Run Commands
```bash
python main.py --run-name exp009_no_replay_seed123_3tasks --tasks sciq arc_challenge boolq --no-replay --seed 123

python main.py --run-name exp009_replay_random_buffer30_seed123_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 123 --replay-selection-strategy random

python main.py --run-name exp009_replay_surprise_buffer30_seed123_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 123 --replay-selection-strategy surprise

python main.py --run-name exp009_replay_predicted_utility_prev_scorer_buffer30_seed123_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 123 --replay-selection-strategy predicted_utility --utility-scorer-path outputs/experiments/exp006/scorer_checkpoints/scorer_model.joblib --utility-feature-columns-path outputs/experiments/exp006/scorer_outputs/feature_columns.json

python main.py --run-name exp009_replay_predicted_utility_expanded_scorer_buffer30_seed123_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 123 --replay-selection-strategy predicted_utility --utility-scorer-path outputs/experiments/exp006/scorer_checkpoints_expanded/scorer_model.joblib --utility-feature-columns-path outputs/experiments/exp006/scorer_outputs_expanded/feature_columns.json
```

## 6. Configuration Summary
| Field | No Replay | Random | Surprise | Predicted Utility Previous Scorer | Predicted Utility Expanded Scorer |
| --- | --- | --- | --- | --- | --- |
| run_name | `exp009_no_replay_seed123_3tasks` | `exp009_replay_random_buffer30_seed123_3tasks` | `exp009_replay_surprise_buffer30_seed123_3tasks` | `exp009_replay_predicted_utility_prev_scorer_buffer30_seed123_3tasks` | `exp009_replay_predicted_utility_expanded_scorer_buffer30_seed123_3tasks` |
| tasks | `sciq -> arc_challenge -> boolq` | `sciq -> arc_challenge -> boolq` | `sciq -> arc_challenge -> boolq` | `sciq -> arc_challenge -> boolq` | `sciq -> arc_challenge -> boolq` |
| replay enabled | `False` | `True` | `True` | `True` | `True` |
| replay buffer size | `disabled` | `30` | `30` | `30` | `30` |
| replay selection strategy | `disabled` | `random` | `surprise` | `predicted_utility` | `predicted_utility` |
| seed | `123` | `123` | `123` | `123` | `123` |
| utility_scorer_path | `-` | `-` | `-` | `outputs/experiments/exp006/scorer_checkpoints/scorer_model.joblib` | `outputs/experiments/exp006/scorer_checkpoints_expanded/scorer_model.joblib` |
| utility_feature_columns_path | `-` | `-` | `-` | `outputs/experiments/exp006/scorer_outputs/feature_columns.json` | `outputs/experiments/exp006/scorer_outputs_expanded/feature_columns.json` |
| model name | `google/flan-t5-small` | `google/flan-t5-small` | `google/flan-t5-small` | `google/flan-t5-small` | `google/flan-t5-small` |
| epochs | `1` | `1` | `1` | `1` | `1` |
| batch size | `2` | `2` | `2` | `2` | `2` |
| learning rate | `0.0005` | `0.0005` | `0.0005` | `0.0005` | `0.0005` |
| output directory | `outputs/experiments/exp009/exp009_no_replay_seed123_3tasks` | `outputs/experiments/exp009/exp009_replay_random_buffer30_seed123_3tasks` | `outputs/experiments/exp009/exp009_replay_surprise_buffer30_seed123_3tasks` | `outputs/experiments/exp009/exp009_replay_predicted_utility_prev_scorer_buffer30_seed123_3tasks` | `outputs/experiments/exp009/exp009_replay_predicted_utility_expanded_scorer_buffer30_seed123_3tasks` |

Note: the no-replay `run_config.json` still stores default replay fields, but replay was disabled and no replay scoring was used.

## 7. Final Validation Accuracy Comparison
| Task | No Replay | Random | Surprise | Predicted Previous | Predicted Expanded | Best Setting |
| --- | --- | --- | --- | --- | --- | --- |
| sciq | 0.3000 | 0.3667 | 0.3667 | 0.3667 | 0.5000 | Predicted Expanded |
| arc_challenge | 0.1333 | 0.2333 | 0.2667 | 0.2333 | 0.2000 | Surprise |
| boolq | 0.6000 | 0.5333 | 0.5333 | 0.5333 | 0.5333 | No Replay |
| Average | 0.3444 | 0.3778 | 0.3889 | 0.3778 | 0.4111 | Predicted Expanded |

Average validation accuracy by setting:
- No Replay: `0.3444`
- Random: `0.3778`
- Surprise: `0.3889`
- Predicted Previous: `0.3778`
- Predicted Expanded: `0.4111`

## 8. Final Test Accuracy Comparison
| Task | No Replay | Random | Surprise | Predicted Previous | Predicted Expanded | Best Setting |
| --- | --- | --- | --- | --- | --- | --- |
| sciq | 0.3000 | 0.4000 | 0.3600 | 0.3400 | 0.4000 | Tie (Random, Predicted Expanded) |
| arc_challenge | 0.1600 | 0.2000 | 0.1800 | 0.2400 | 0.1800 | Predicted Previous |
| boolq | 0.4800 | 0.4800 | 0.4800 | 0.4800 | 0.4800 | Tie (All Settings) |
| Average | 0.3133 | 0.3600 | 0.3400 | 0.3533 | 0.3533 | Random |

Average test accuracy by setting:
- No Replay: `0.3133`
- Random: `0.3600`
- Surprise: `0.3400`
- Predicted Previous: `0.3533`
- Predicted Expanded: `0.3533`

## 9. Forgetting Analysis
| Task | No Replay Forgetting | Random Forgetting | Surprise Forgetting | Predicted Previous Forgetting | Predicted Expanded Forgetting | Lowest Forgetting Setting |
| --- | --- | --- | --- | --- | --- | --- |
| sciq | 0.2333 | 0.1667 | 0.1667 | 0.1667 | 0.0333 | Predicted Expanded |
| arc_challenge | 0.0000 | 0.0667 | 0.0000 | 0.0667 | 0.0333 | Tie (No Replay, Surprise) |
| boolq | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Tie (All Settings) |
| Maximum | 0.2333 | 0.1667 | 0.1667 | 0.1667 | 0.0333 | Predicted Expanded |

Maximum forgetting by setting:
- No Replay: `0.2333`
- Random: `0.1667`
- Surprise: `0.1667`
- Predicted Previous: `0.1667`
- Predicted Expanded: `0.0333`

## 10. Robustness Comparison with exp008
- Validation winner changed:
  - `exp008` seed `42`: tie between `Random` and `Predicted Previous` at `0.4000`
  - `exp009` seed `123`: `Predicted Expanded` at `0.4111`
- Test winner changed:
  - `exp008` seed `42`: `Predicted Expanded` at `0.4267`
  - `exp009` seed `123`: `Random` at `0.3600`
- Lowest maximum forgetting changed:
  - `exp008` seed `42`: tie between `No Replay` and `Predicted Previous` at `0.0000`
  - `exp009` seed `123`: `Predicted Expanded` at `0.0333`
- Previous scorer stability:
  - average validation changed from `0.4000` to `0.3778`
  - average test changed from `0.3333` to `0.3533`
  - maximum forgetting changed from `0.0000` to `0.1667`
- Expanded scorer stability:
  - average validation changed from `0.3444` to `0.4111`
  - average test changed from `0.4267` to `0.3533`
  - maximum forgetting changed from `0.1333` to `0.0333`
- Cautious stability read:
  - the scorer-source pattern from `exp008` did not hold under seed `123`
  - the previous scorer looks somewhat more stable in test behavior across seeds
  - the expanded scorer looks more seed-sensitive because it flips from weakest validation / worst forgetting in `exp008` to strongest validation / best forgetting in `exp009`

## 11. Main Findings
- `Predicted Expanded` is the best setting on average validation accuracy at seed `123`.
- `Random` is the best setting on average test accuracy at seed `123`.
- `Predicted Expanded` has the lowest maximum forgetting at seed `123`.
- Both predicted_utility runs are valid end to end: scorer paths were recorded, replay score files were saved, replay buffers were non-empty, and no fallback occurred.
- The previous scorer does not repeat its seed-42 validation and forgetting advantage from `exp008`.
- The expanded scorer does not repeat its seed-42 test advantage from `exp008`.
- The seed change materially alters the relative ranking of the two scorer sources.

## 12. Interpretation
`predicted_utility` appears seed-sensitive under the current lightweight setup. The expanded scorer looks strongest on validation accuracy and forgetting at seed `123`, but it no longer gives the best average test accuracy. The previous scorer remains competitive, but it is no longer a validation or forgetting winner. The safest reading is that scorer-source conclusions are not stable yet, and the current predicted_utility result remains exploratory rather than decisive.

## 13. Limitations
- Still only two predicted_utility seeds: `42` and `123`
- One epoch per run
- Lightweight subsets: train `100`, val `30`, test `50`
- Limited human review labels
- Simple scorer: `LogisticRegression`
- No statistical significance test

## 14. Relation to Other Experiments
- `exp001`: original-order replay baseline
- `exp002`: reverse-order task sensitivity
- `exp003`: additional original-order baseline
- `exp004`: replay buffer size ablation
- `exp005`: non-predicted replay seed robustness
- `exp006`: strategy comparison plus PRE-PLAY review and scorer training
- `exp007`: first predicted_utility replay run
- `exp008`: scorer-source comparison at seed `42`
- `exp009`: scorer-source robustness check at seed `123`

## 15. Next Step
Stop running new experiments temporarily and write the project report section based on `exp001` to `exp009`, unless more seeds are specifically required.

## 16. Reproducibility Checklist
- [x] run commands recorded
- [x] run_config.json saved
- [x] scorer paths recorded
- [x] metrics saved
- [x] replay artifacts saved
- [x] summaries updated

## 17. Notes
- Completion status: all five `exp009` runs completed and the experiment directory is present.
- Pre-run compile check passed before execution.
- Both predicted_utility runs recorded the requested scorer paths in `run_config.json`.
- `replay_selection_comparison/*.csv` contains only `predicted_utility` in both predicted runs, so no fallback to `random`, `loss`, or `surprise` was detected.
- No `exp001` to `exp008` training was rerun while producing `exp009`.
- Generation timestamp: `2026-05-09 22:28:03 NZST`
