# exp008 Summary

## 1. Experiment Overview
`exp008` is a predicted_utility scorer-source comparison. It keeps the task order, seed, and replay buffer budget fixed, then compares two scorer sources for `predicted_utility` replay against `no_replay`, `random`, and `surprise` reference settings.

## 2. Research Question
"Does the expanded human-review-trained scorer improve predicted_utility replay compared with the previous 57-label scorer?"

## 3. Experiment Design
- No-replay baseline: `exp008_no_replay_seed42_3tasks`
- Random replay: `exp008_replay_random_buffer30_seed42_3tasks`
- Surprise replay: `exp008_replay_surprise_buffer30_seed42_3tasks`
- Predicted utility with previous scorer: `exp008_replay_predicted_utility_prev_scorer_buffer30_seed42_3tasks`
- Predicted utility with expanded scorer: `exp008_replay_predicted_utility_expanded_scorer_buffer30_seed42_3tasks`
- Task sequence: `sciq -> arc_challenge -> boolq`
- Seed: `42`
- Replay buffer size: `30`

## 4. Scorer Sources
| Scorer Source | Label Count | Keep | Reject | Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Previous scorer | 57 | 22 | 35 | LogisticRegression | 0.8333 | 0.7500 | 0.8571 | 0.8000 | 0.8052 |
| Expanded scorer | 113 | 53 | 60 | LogisticRegression | 0.7353 | 0.7333 | 0.6875 | 0.7097 | 0.7778 |

- The previous scorer has fewer reviewed labels but better held-out metrics on its saved split.
- The expanded scorer has more reviewed labels, but its saved held-out metrics are lower. More labels did not automatically produce a better scorer.

## 5. Run Commands
```bash
python main.py --run-name exp008_no_replay_seed42_3tasks --tasks sciq arc_challenge boolq --no-replay --seed 42

python main.py --run-name exp008_replay_random_buffer30_seed42_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 42 --replay-selection-strategy random

python main.py --run-name exp008_replay_surprise_buffer30_seed42_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 42 --replay-selection-strategy surprise

python main.py --run-name exp008_replay_predicted_utility_prev_scorer_buffer30_seed42_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 42 --replay-selection-strategy predicted_utility --utility-scorer-path outputs/experiments/exp006/scorer_checkpoints/scorer_model.joblib --utility-feature-columns-path outputs/experiments/exp006/scorer_outputs/feature_columns.json

python main.py --run-name exp008_replay_predicted_utility_expanded_scorer_buffer30_seed42_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 42 --replay-selection-strategy predicted_utility --utility-scorer-path outputs/experiments/exp006/scorer_checkpoints_expanded/scorer_model.joblib --utility-feature-columns-path outputs/experiments/exp006/scorer_outputs_expanded/feature_columns.json
```

## 6. Configuration Summary
| Field | No Replay | Random | Surprise | Predicted Utility Previous Scorer | Predicted Utility Expanded Scorer |
| --- | --- | --- | --- | --- | --- |
| run_name | `exp008_no_replay_seed42_3tasks` | `exp008_replay_random_buffer30_seed42_3tasks` | `exp008_replay_surprise_buffer30_seed42_3tasks` | `exp008_replay_predicted_utility_prev_scorer_buffer30_seed42_3tasks` | `exp008_replay_predicted_utility_expanded_scorer_buffer30_seed42_3tasks` |
| tasks | `sciq -> arc_challenge -> boolq` | `sciq -> arc_challenge -> boolq` | `sciq -> arc_challenge -> boolq` | `sciq -> arc_challenge -> boolq` | `sciq -> arc_challenge -> boolq` |
| replay enabled | `False` | `True` | `True` | `True` | `True` |
| replay buffer size | `disabled` | `30` | `30` | `30` | `30` |
| replay selection strategy | `disabled` | `random` | `surprise` | `predicted_utility` | `predicted_utility` |
| seed | `42` | `42` | `42` | `42` | `42` |
| utility_scorer_path | `-` | `-` | `-` | `outputs/experiments/exp006/scorer_checkpoints/scorer_model.joblib` | `outputs/experiments/exp006/scorer_checkpoints_expanded/scorer_model.joblib` |
| utility_feature_columns_path | `-` | `-` | `-` | `outputs/experiments/exp006/scorer_outputs/feature_columns.json` | `outputs/experiments/exp006/scorer_outputs_expanded/feature_columns.json` |
| model name | `google/flan-t5-small` | `google/flan-t5-small` | `google/flan-t5-small` | `google/flan-t5-small` | `google/flan-t5-small` |
| epochs | `1` | `1` | `1` | `1` | `1` |
| batch size | `2` | `2` | `2` | `2` | `2` |
| learning rate | `0.0005` | `0.0005` | `0.0005` | `0.0005` | `0.0005` |
| output directory | `outputs/experiments/exp008/exp008_no_replay_seed42_3tasks` | `outputs/experiments/exp008/exp008_replay_random_buffer30_seed42_3tasks` | `outputs/experiments/exp008/exp008_replay_surprise_buffer30_seed42_3tasks` | `outputs/experiments/exp008/exp008_replay_predicted_utility_prev_scorer_buffer30_seed42_3tasks` | `outputs/experiments/exp008/exp008_replay_predicted_utility_expanded_scorer_buffer30_seed42_3tasks` |

Note: the no-replay `run_config.json` still stores default replay fields, but replay was disabled and no replay scoring was used.

## 7. Final Validation Accuracy Comparison
| Task | No Replay | Random | Surprise | Predicted Previous | Predicted Expanded | Best Setting |
| --- | --- | --- | --- | --- | --- | --- |
| sciq | 0.4000 | 0.3000 | 0.3333 | 0.4000 | 0.3333 | Tie (No Replay, Predicted Previous) |
| arc_challenge | 0.2333 | 0.3333 | 0.2667 | 0.2667 | 0.1667 | Random |
| boolq | 0.5000 | 0.5667 | 0.5667 | 0.5333 | 0.5333 | Tie (Random, Surprise) |
| Average | 0.3778 | 0.4000 | 0.3889 | 0.4000 | 0.3444 | Tie (Random, Predicted Previous) |

Average validation accuracy by setting:
- No Replay: `0.3778`
- Random: `0.4000`
- Surprise: `0.3889`
- Predicted Previous: `0.4000`
- Predicted Expanded: `0.3444`

## 8. Final Test Accuracy Comparison
| Task | No Replay | Random | Surprise | Predicted Previous | Predicted Expanded | Best Setting |
| --- | --- | --- | --- | --- | --- | --- |
| sciq | 0.4000 | 0.3600 | 0.3800 | 0.2800 | 0.4800 | Predicted Expanded |
| arc_challenge | 0.1800 | 0.2600 | 0.2600 | 0.2400 | 0.3200 | Predicted Expanded |
| boolq | 0.4800 | 0.4600 | 0.4600 | 0.4800 | 0.4800 | Tie (No Replay, Predicted Previous, Predicted Expanded) |
| Average | 0.3533 | 0.3600 | 0.3667 | 0.3333 | 0.4267 | Predicted Expanded |

Average test accuracy by setting:
- No Replay: `0.3533`
- Random: `0.3600`
- Surprise: `0.3667`
- Predicted Previous: `0.3333`
- Predicted Expanded: `0.4267`

## 9. Forgetting Analysis
| Task | No Replay Forgetting | Random Forgetting | Surprise Forgetting | Predicted Previous Forgetting | Predicted Expanded Forgetting | Lowest Forgetting Setting |
| --- | --- | --- | --- | --- | --- | --- |
| sciq | 0.0000 | 0.1000 | 0.0667 | 0.0000 | 0.1333 | Tie (No Replay, Predicted Previous) |
| arc_challenge | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0667 | Tie (No Replay, Random, Surprise, Predicted Previous) |
| boolq | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Tie (All Settings) |
| Maximum | 0.0000 | 0.1000 | 0.0667 | 0.0000 | 0.1333 | Tie (No Replay, Predicted Previous) |

Maximum forgetting by setting:
- No Replay: `0.0000`
- Random: `0.1000`
- Surprise: `0.0667`
- Predicted Previous: `0.0000`
- Predicted Expanded: `0.1333`

## 10. Predicted Utility Analysis
- Previous scorer vs expanded scorer:
  - Previous scorer is better on average validation accuracy (`0.4000` vs `0.3444`).
  - Previous scorer ties the best maximum forgetting (`0.0000`), while expanded scorer is worst on maximum forgetting (`0.1333`).
  - Expanded scorer is better on average test accuracy (`0.4267` vs `0.3333`).
- Both predicted_utility runs vs random:
  - Previous scorer ties random on average validation accuracy (`0.4000`) but trails random on average test accuracy (`0.3333` vs `0.3600`).
  - Expanded scorer trails random on average validation accuracy (`0.3444` vs `0.4000`) but exceeds random on average test accuracy (`0.4267` vs `0.3600`).
- Both predicted_utility runs vs surprise:
  - Previous scorer slightly exceeds surprise on average validation accuracy (`0.4000` vs `0.3889`) but trails on average test accuracy (`0.3333` vs `0.3667`).
  - Expanded scorer trails surprise on average validation accuracy (`0.3444` vs `0.3889`) but exceeds it on average test accuracy (`0.4267` vs `0.3667`).
- Selection overlap shows the two predicted_utility runs are related but not identical:
  - Previous vs expanded overlap: `71` shared selections, Jaccard `0.6514`
  - Previous vs surprise overlap: Jaccard `0.7143`
  - Expanded vs surprise overlap: Jaccard `0.5126`
  - Both predicted runs remain far from random: Jaccard `0.1921`

## 11. PRE-PLAY Artifact Analysis
| Run | Replay Score CSV Count | Replay Buffer Snapshot Count | Non-Empty Replay Buffer Snapshot Count | Sample Signals CSV Count | Utility Labels CSV Count | Human Review Candidate File |
| --- | --- | --- | --- | --- | --- | --- |
| Random | 3 | 3 | 3 | 3 | 3 | Yes |
| Surprise | 3 | 3 | 3 | 3 | 3 | Yes |
| Predicted Previous | 3 | 3 | 3 | 3 | 3 | Yes |
| Predicted Expanded | 3 | 3 | 3 | 3 | 3 | Yes |

No-replay artifact note:
- `replay_scores/` contains `0` CSV files, as expected with replay disabled.
- `replay_buffers/` contains `3` placeholder snapshots, with `0` non-empty snapshots.

Predicted utility run validation:
- Previous scorer run:
  - scorer model path used: `outputs/experiments/exp006/scorer_checkpoints/scorer_model.joblib`
  - feature columns path used: `outputs/experiments/exp006/scorer_outputs/feature_columns.json`
  - predicted utility scores were used: `yes`
  - `replay_selection_comparison/*.csv` contains only `predicted_utility`
- Expanded scorer run:
  - scorer model path used: `outputs/experiments/exp006/scorer_checkpoints_expanded/scorer_model.joblib`
  - feature columns path used: `outputs/experiments/exp006/scorer_outputs_expanded/feature_columns.json`
  - predicted utility scores were used: `yes`
  - `replay_selection_comparison/*.csv` contains only `predicted_utility`

No silent fallback to `random`, `loss`, or `surprise` was detected in either predicted_utility run.

## 12. Main Findings
- The previous 57-label scorer and random replay tie for best average validation accuracy at `0.4000`.
- The expanded 113-label scorer gives the best average test accuracy at `0.4267`.
- The previous scorer ties the no-replay baseline for lowest maximum forgetting at `0.0000`.
- The expanded scorer changes the selected replay set materially relative to the previous scorer, especially versus surprise replay.
- More reviewed labels did not automatically produce a uniformly stronger predicted_utility result.
- The two predicted_utility variants trade off validation, test, and forgetting differently, so the scorer-source effect is mixed rather than one-directional.

## 13. Interpretation
`exp008` does not support a simple claim that the expanded scorer is better because it uses more labels. The expanded scorer is stronger on average test accuracy, but weaker on average validation accuracy and forgetting. The previous scorer retains the cleaner held-out scorer metrics from exp006 and that pattern carries into stronger validation and forgetting behavior here. The expanded scorer still appears useful because it changes the selected replay set and produces the strongest average test result in this saved run, but the result remains exploratory rather than confirmatory.

## 14. Limitations
- One seed: `42`
- One epoch per run
- Lightweight sample sizes: train `100`, val `30`, test `50`
- Human review labels are still limited
- Scorer quality remains uncertain
- Utility labels are human or proxy labels, not exact causal future utility
- The scorer-source comparison is exploratory

## 15. Relation to Other Experiments
- `exp001`: original-order replay vs no-replay baseline
- `exp002`: reverse-order replay vs no-replay sensitivity check
- `exp003`: additional original-order replay baseline
- `exp004`: replay buffer size ablation
- `exp005`: seed robustness check for buffer size 30
- `exp006`: strategy comparison plus PRE-PLAY artifact generation and scorer training
- `exp007`: first predicted_utility replay experiment using the previous exp006 scorer
- `exp008`: scorer-source comparison between previous and expanded predicted_utility scorers

## 16. Next Step
Improve human review quality and feature design before relying on expanded predicted_utility, or evaluate predicted_utility across more seeds.

## 17. Reproducibility Checklist
- [x] run commands recorded
- [x] run_config.json saved
- [x] scorer paths recorded
- [x] metrics saved
- [x] replay artifacts saved
- [x] summaries updated

## 18. Notes
- Completion status: all five `exp008` runs completed and the experiment directory is present.
- Pre-run compile check passed before execution.
- During the first expanded-scorer attempt, prediction failed because the saved feature metadata included `review_priority` and the prediction path was not honoring that metadata. `utility_scorer.py` was updated to use the saved feature schema, then the failed expanded scorer run was rerun and completed.
- No `exp001` to `exp007` training was rerun while producing `exp008`.
- Generation timestamp: `2026-05-09`
