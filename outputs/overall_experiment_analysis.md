# Overall Experiment Analysis

## 1. Project-Level Experiment Overview
These saved experiments study replay-based continual learning on a lightweight sequence of QA tasks: `sciq`, `arc_challenge`, and `boolq`. The project started with replay versus no-replay baselines, then added task-order checks, replay buffer-size ablations, seed checks, replay selection strategy comparisons, and exploratory PRE-PLAY `predicted_utility` replay. The latest experiments extend the evidence with a direct random replay baseline (`exp010`), a three-seed robustness experiment (`exp011`), and an epoch-sensitivity experiment (`exp012`).

The overall evidence should be read cautiously. Replay is operational and sometimes helpful, especially for forgetting, but the saved results do not support a strong claim that scored replay consistently improves final accuracy.

## 2. Experiment Roadmap
| Experiment | Purpose | Task Order | Main Comparison | Seed / Epochs | Main Output Summary |
| --- | --- | --- | --- | --- | --- |
| exp001 | Original-order baseline | sciq -> arc_challenge -> boolq | Replay vs no-replay | seed 42, 1 epoch | [exp001_summary.md](experiments/exp001/exp001_summary.md) |
| exp002 | Reverse-order sensitivity check | boolq -> arc_challenge -> sciq | Replay vs no-replay under reversed order | seed 42, 1 epoch | [exp002_summary.md](experiments/exp002/exp002_summary.md) |
| exp003 | Additional baseline-style comparison | sciq -> arc_challenge -> boolq | Replay vs no-replay under the original order again | seed 42, 1 epoch | [exp003_summary.md](experiments/exp003/exp003_summary.md) |
| exp004 | Replay buffer size ablation | sciq -> arc_challenge -> boolq | No-replay vs Buffer 30 / 60 / 100 | seed 42, 1 epoch | [exp004_summary.md](experiments/exp004/exp004_summary.md) |
| exp005 | Seed robustness check | sciq -> arc_challenge -> boolq | No-replay vs Buffer 30 at a new seed | seed 123, 1 epoch | [exp005_summary.md](experiments/exp005/exp005_summary.md) |
| exp006 | Replay selection strategy comparison with PRE-PLAY artifacts | sciq -> arc_challenge -> boolq | No-replay vs Random / Loss / Surprise replay | seed 42, 1 epoch | [exp006_summary.md](experiments/exp006/exp006_summary.md) |
| exp007 | First predicted_utility replay experiment using the exp006 scorer | sciq -> arc_challenge -> boolq | No-replay vs Random / Surprise / Predicted Utility replay | seed 42, 1 epoch | [exp007_summary.md](experiments/exp007/exp007_summary.md) |
| exp008 | Predicted_utility scorer-source comparison | sciq -> arc_challenge -> boolq | No-replay vs Random / Surprise vs Predicted Utility with Previous / Expanded scorer | seed 42, 1 epoch | [exp008_summary.md](experiments/exp008/exp008_summary.md) |
| exp009 | Predicted_utility scorer-source robustness check | sciq -> arc_challenge -> boolq | No-replay vs Random / Surprise vs Predicted Utility with Previous / Expanded scorer | seed 123, 1 epoch | [exp009_summary.md](experiments/exp009/exp009_summary.md) |
| exp010 | Random replay baseline comparison | sciq -> arc_challenge -> boolq | No replay vs scored replay vs random replay single-seed comparison | seed 42, 1 epoch | [exp010_summary.md](experiments/exp010/exp010_summary.md) |
| exp011 | Multi-seed robustness experiment | sciq -> arc_challenge -> boolq | No replay vs scored replay vs random replay across seeds | seeds 13 / 42 / 100, 1 epoch | [exp011_summary.md](experiments/exp011/exp011_summary.md) |
| exp012 | Epoch sensitivity experiment | sciq -> arc_challenge -> boolq | No replay vs scored replay vs random replay across epoch counts | seed 42, epochs 1 / 2 / 3 | [exp012_summary.md](experiments/exp012/exp012_summary.md) |

## 3. Research Questions Across Experiments
1. Does replay improve retention and final performance?
2. Is replay sensitive to task order?
3. Does replay buffer size affect performance?
4. Is the best buffer-size result stable under another seed?
5. Under the same replay buffer budget, does replay sample selection strategy matter?
6. Can a human-review-trained utility scorer improve replay selection compared with random and surprise?
7. Does the expanded human-review-trained scorer improve predicted_utility replay compared with the previous 57-label scorer?
8. Are the exp008 predicted_utility scorer-source results stable when the random seed changes from 42 to 123?
9. Does scored replay outperform random replay when both are compared directly against no replay?
10. Are the no replay, scored replay, and random replay results stable across seeds?
11. Does the relative value of replay change when the number of training epochs changes?

## 4. Summary of Individual Experiments
### exp001: Original-Order Baseline
- Design: replay versus no-replay over `sciq -> arc_challenge -> boolq` at seed `42`.
- Main validation result: average replay-minus-no-replay validation delta was `-0.0111`.
- Main test result: average replay-minus-no-replay test delta was `-0.0333`.
- Forgetting result: maximum saved forgetting was tied at `0.1000`.
- Cautious conclusion: the saved evidence is mixed and does not support a consistent replay benefit in this baseline.

### exp002: Reverse Task-Order Sensitivity
- Design: replay versus no-replay over `boolq -> arc_challenge -> sciq` at seed `42`.
- Main validation result: average replay-minus-no-replay validation delta was `-0.0111`.
- Main test result: average replay-minus-no-replay test delta was `+0.0133`.
- Forgetting result: maximum saved forgetting was tied at `0.0667`.
- Cautious conclusion: reversing task order changed the replay pattern, especially on test metrics, which suggests task-order sensitivity rather than a stable replay effect.

### exp003: Additional Baseline-Style Comparison
- Design: another replay versus no-replay comparison over `sciq -> arc_challenge -> boolq` at seed `42`.
- Main result: average replay-minus-no-replay deltas were `-0.0111` on validation and `-0.0267` on test.
- Forgetting result: maximum saved forgetting was tied at `0.1000`.
- Cautious conclusion: this additional baseline-style comparison again gives mixed evidence and leans against a general replay benefit under the saved lightweight setting.

### exp004: Replay Buffer Size Ablation
- Design: one no-replay baseline plus replay buffer sizes `30`, `60`, and `100`, all over `sciq -> arc_challenge -> boolq` at seed `42`.
- Best average validation accuracy among replay settings: `Buffer 30`.
- Best average test accuracy among replay settings: tie across `Buffer 30`, `Buffer 60`, and `Buffer 100`, while no-replay still has the strongest average test accuracy overall.
- Lowest maximum forgetting among replay settings: `Buffer 30`; it also ties the no-replay baseline on maximum saved forgetting.
- Cautious conclusion: `Buffer 30` is the most promising replay setting in this saved sweep, but the overall evidence remains mixed.

### exp005: Seed Robustness Check
- Design: no-replay versus `Buffer 30` over `sciq -> arc_challenge -> boolq` at seed `123`.
- Validation result: average replay-minus-no-replay validation delta was `+0.1222`.
- Test result: average replay-minus-no-replay test delta was `+0.1000`, although `boolq` test accuracy still favored no-replay.
- Forgetting result: maximum saved forgetting improved from `0.3333` to `0.2333`.
- Cautious conclusion: exp005 tentatively supports the Buffer 30 pattern, but the result is still limited to one additional seed.

### exp006: Replay Selection Strategy Comparison
- Design: no-replay, random replay, loss replay, and surprise replay over `sciq -> arc_challenge -> boolq` at seed `42` with replay buffer size `30`.
- Validation result: average final validation accuracy is tied across no-replay, loss replay, and surprise replay at `0.3889`, while random replay is slightly lower at `0.3778`.
- Test result: average final test accuracy is highest for loss replay and surprise replay at `0.3667`, above random replay at `0.3600` and no-replay at `0.3467`.
- PRE-PLAY artifacts: the replay-enabled runs generated `sample_signals`, `utility_labels`, and `human_review` candidate files, and exp006 also has experiment-level reviewed-label exports and scorer outputs.
- Scorer training result: `utility_scorer.py` was trained from reviewed utility labels with 57 usable labels, then retrained from the canonical expanded review file with 113 usable labels.
- Cautious conclusion: exp006 gives preliminary evidence that score-based replay selection may matter on average test accuracy, but the evidence remains lightweight.

### exp007: Exploratory Predicted Utility Replay
- Design: no replay, random replay, surprise replay, and `predicted_utility` replay over `sciq -> arc_challenge -> boolq` at seed `42`.
- Scorer source: the exp006 reviewed-label `LogisticRegression` scorer trained from 57 usable labels.
- Validation result: average final validation accuracy is highest for predicted utility at `0.4111`.
- Test result: average final test accuracy is highest for surprise replay at `0.3667`; predicted utility trails random and surprise.
- Cautious conclusion: exp007 shows that the PRE-PLAY scorer-loading pipeline works end to end, but the current scorer does not outperform the stronger replay baselines.

### exp008: Predicted Utility Scorer-Source Comparison
- Design: no replay, random replay, surprise replay, `predicted_utility` with the previous 57-label scorer, and `predicted_utility` with the expanded 113-label scorer over `sciq -> arc_challenge -> boolq` at seed `42`.
- Test result: average final test accuracy is highest for the expanded scorer at `0.4267`, above surprise replay, random replay, no replay, and the previous scorer.
- Forgetting result: the lowest maximum forgetting is a tie between no replay and the previous scorer, while the expanded scorer is highest.
- Cautious conclusion: exp008 is mixed. The expanded scorer improves average test accuracy, but more reviewed labels did not automatically produce a stronger predicted_utility replay policy on every metric.

### exp009: Predicted Utility Seed-123 Robustness Check
- Design: no replay, random replay, surprise replay, `predicted_utility` with the previous scorer, and `predicted_utility` with the expanded scorer over `sciq -> arc_challenge -> boolq` at seed `123`.
- Validation result: average final validation accuracy is highest for the expanded scorer.
- Test result: average final test accuracy is highest for random replay, while both predicted_utility runs tie below it.
- Forgetting result: the expanded scorer has the lowest maximum forgetting.
- Cautious conclusion: the scorer-source pattern from exp008 did not hold under seed `123`, so the result is exploratory and seed-sensitive.

### exp010: Random Replay Baseline Comparison
- Design: no replay, scored replay, and random replay over `sciq -> arc_challenge -> boolq`.
- Seed: `42`.
- Purpose: exp010 adds the direct random replay baseline so scored replay can be compared against random replay and no replay in the same single-seed setup.
- Final test means: no replay `0.3467`, scored replay `0.3667`, random replay `0.3400`.
- Forgetting means: no replay `0.0000`, scored replay `0.0222`, random replay `0.0444`.
- Cautious conclusion: scored replay is stronger than random replay under seed `42`, but this is a single-seed result and should not be treated as a robust conclusion.

### exp011: Multi-Seed Robustness Experiment
- Design: no replay, scored replay, and random replay over `sciq -> arc_challenge -> boolq`.
- Seeds: `13`, `42`, and `100`.
- Importance: exp011 is the current strongest main experiment because it compares the three core conditions across multiple seeds.
- Mean test accuracy: no replay `0.3644 +/- 0.0269`, scored replay `0.3489 +/- 0.0204`, random replay `0.3400 +/- 0.0067`.
- Mean forgetting: no replay `0.0519 +/- 0.0559`, scored replay `0.0444 +/- 0.0222`, random replay `0.0370 +/- 0.0064`.
- Cautious conclusion: no replay has the highest mean test accuracy; scored replay is slightly better than random replay; replay conditions show lower mean forgetting than no replay; results are mixed and seed-sensitive.

### exp012: Epoch Sensitivity Experiment
- Design: no replay, scored replay, and random replay over `sciq -> arc_challenge -> boolq`.
- Fixed seed: `42`.
- Epochs: `1`, `2`, and `3`.
- Mean test accuracy:
  - Epoch 1: no replay `0.3533`, scored replay `0.3667`, random replay `0.3400`.
  - Epoch 2: no replay `0.3533`, scored replay `0.3933`, random replay `0.3200`.
  - Epoch 3: no replay `0.3533`, scored replay `0.3800`, random replay `0.3533`.
- Cautious conclusion: scored replay is best at all three epoch settings under seed `42`, and the best scored replay result appears at epoch `2`; this is supplementary evidence only because it is single-seed.

## 5. Cross-Experiment Comparison
| Experiment | Replay Setting Tested | Validation Evidence | Test Evidence | Forgetting Evidence | Overall Interpretation |
| --- | --- | --- | --- | --- | --- |
| exp001 | Default replay setting | Average delta `-0.0111` | Average delta `-0.0333` | Max forgetting tie at `0.1000` | Mixed evidence; does not support replay overall |
| exp002 | Default replay setting | Average delta `-0.0111` | Average delta `+0.0133` | Max forgetting tie at `0.0667` | Mixed evidence; task-order-sensitive |
| exp003 | Default replay setting | Average delta `-0.0111` | Average delta `-0.0267` | Max forgetting tie at `0.1000` | Mixed evidence; does not support replay overall |
| exp004 | Buffer 30 / 60 / 100 | Buffer 30 best among replay settings; larger buffers worse | All replay settings trail no-replay on average test accuracy | Buffer 30 best among replay settings and ties no-replay on max forgetting | Mixed evidence; buffer-size-sensitive |
| exp005 | Buffer 30 at seed 123 | Average delta `+0.1222` | Average delta `+0.1000`, but one test task favors no-replay | Max forgetting improves from `0.3333` to `0.2333` | Tentative support for replay, but seed-sensitive |
| exp006 | Random / Loss / Surprise at Buffer 30 | No-replay, loss replay, and surprise replay tie on average validation accuracy | Loss replay and surprise replay are best on average test accuracy | No-replay has the lowest max forgetting; loss and surprise beat random | Preliminary evidence that replay selection strategy may matter |
| exp007 | Random / Surprise / Predicted Utility at Buffer 30 | Predicted utility is best on average validation accuracy | Surprise replay is best on average test accuracy; predicted utility trails random and surprise on average test | Final saved forgetting ties at `0.0000` across settings | The PRE-PLAY pipeline works, but the current scorer is not yet strong enough |
| exp008 | Random / Surprise / Predicted Utility with Previous vs Expanded scorer | Random and Predicted Previous tie for best average validation accuracy; Expanded is lower | Predicted Expanded is best on average test accuracy | No-replay and Predicted Previous tie for lowest max forgetting | Mixed scorer-source evidence; more labels did not automatically improve predicted_utility |
| exp009 | Random / Surprise / Predicted Utility with Previous vs Expanded scorer at seed 123 | Predicted Expanded is best on average validation accuracy | Random is best on average test accuracy; both predicted runs tie below it | Predicted Expanded has the lowest max forgetting | The scorer-source ranking is not stable across seeds |
| exp010 | No replay / scored replay / random replay | Scored replay has the best average validation accuracy | Scored replay beats no replay and random replay on average test accuracy | No replay has lowest forgetting; scored beats random | Useful direct random baseline, but single-seed only |
| exp011 | No replay / scored replay / random replay across seeds | Replay conditions have higher mean validation accuracy than no replay | No replay has highest mean test accuracy; scored is slightly above random | Replay conditions have lower mean forgetting than no replay | Current strongest result; mixed and seed-sensitive |
| exp012 | No replay / scored replay / random replay across epochs | Scored replay is strongest at 3 epochs and tied at 1-2 epochs | Scored replay is best at all three epoch settings | Forgetting increases relative to 1 epoch for all conditions | Useful supplementary epoch-sensitivity evidence, but single-seed |

## 6. Replay Effectiveness Analysis
- Replay helped most clearly in `exp005`, where `Buffer 30` improved average validation accuracy, average test accuracy, and maximum saved forgetting relative to no-replay.
- Replay helped partially in `exp002` on average test accuracy, but not on average validation accuracy or maximum forgetting.
- Replay hurt on average validation and test accuracy in `exp001` and `exp003`.
- In `exp006`, loss replay and surprise replay both outperform random replay on average test accuracy.
- In `exp007`, `predicted_utility` replay runs valid end to end, but it does not beat random or surprise on the saved summary metrics.
- In `exp008` and `exp009`, the scorer-source ranking changes across seeds, so predicted_utility remains exploratory.
- `exp010` adds the direct random replay baseline and shows scored replay above random replay at seed `42`.
- `exp011` is the strongest current robustness experiment: no replay has the highest mean test accuracy, scored replay is slightly stronger than random replay, and both replay conditions show lower mean forgetting than no replay.
- `exp012` is useful as supplementary epoch-sensitivity evidence: scored replay is best at each tested epoch under seed `42`, but that pattern has not been established across seeds.
- The saved evidence is not strong enough to claim that scored replay consistently improves performance.
- Replay may reduce forgetting under the lightweight setting more reliably than it improves final test accuracy.

## 7. Task-Order Sensitivity
`exp001` and `exp002` use the same three tasks but in different orders: `sciq -> arc_challenge -> boolq` versus `boolq -> arc_challenge -> sciq`. Under the original order, replay is slightly worse on average validation and test accuracy. Under the reversed order, replay is still slightly worse on average validation accuracy, but it becomes slightly better on average test accuracy. The task-level winners shift across the reversed sequence, suggesting that the replay effect is sensitive to task order rather than invariant across the same task set.

## 8. Buffer Size Analysis
`exp004` compares `Buffer 30`, `Buffer 60`, and `Buffer 100` against no-replay. `Buffer 30` is the strongest replay setting on average validation accuracy and on maximum forgetting among replay settings. Average test accuracy is tied across the three replay settings, and all of them remain below the no-replay baseline on average test accuracy. Larger replay buffers therefore do not help monotonically.

## 9. Seed Robustness Analysis
`exp004` and `exp005` already suggested seed sensitivity for replay. The same issue appears in the PRE-PLAY-specific runs: `exp008` and `exp009` keep the same task order, scorer sources, and replay budget, then change only the seed from `42` to `123`, and the scorer-source ranking changes.

`exp011` is the cleaner robustness test for the current no replay / scored replay / random replay comparison. Across seeds `13`, `42`, and `100`, no replay has the highest mean test accuracy (`0.3644 +/- 0.0269`), scored replay is second (`0.3489 +/- 0.0204`), and random replay is third (`0.3400 +/- 0.0067`). Replay looks better on mean forgetting, where no replay is `0.0519 +/- 0.0559`, scored replay is `0.0444 +/- 0.0222`, and random replay is `0.0370 +/- 0.0064`. This supports a mixed conclusion rather than a stable scored-replay win.

## 10. Epoch Sensitivity Analysis
`exp012` tests whether the relative value of replay changes when the number of training epochs changes. Under fixed seed `42`, scored replay has the highest mean test accuracy at `1`, `2`, and `3` epochs, with its best mean test accuracy at epoch `2` (`0.3933`). No replay is flat at `0.3533` across the three epoch settings, while random replay is lower at epochs `1` and `2` and ties no replay at epoch `3`.

This is useful supplementary evidence that training budget matters, but it is not a robustness result because it uses a single seed.

## 11. Audited Review Provenance
`exp011` and `exp012` attach `outputs/final_review/final_reviewed_dataset.csv` as audited replay / utility review provenance. The audited CSV is not used as a full train/validation/test replacement. Train, validation, and test samples remain loaded from `data/processed` JSONL files.

## 12. Main Findings
- Replay was operational in replay-enabled runs because replay score CSV files and non-empty replay buffer snapshots were saved.
- `exp001` and `exp003` do not show average validation or average test improvements from the default replay setting.
- Reversing task order in `exp002` changes the replay pattern, especially on test accuracy.
- In `exp004`, `Buffer 30` is the strongest replay setting among the tested buffer sizes, but no-replay remains strongest on average test accuracy.
- `exp005` is more favorable to `Buffer 30` than the seed `42` result, showing early seed sensitivity.
- `exp006` established the PRE-PLAY artifact path and suggests that score-based replay selection may matter.
- `exp007`, `exp008`, and `exp009` show that predicted_utility replay is technically valid but still exploratory and seed-sensitive.
- `exp010` adds the direct random replay baseline.
- `exp011` is the strongest current robustness experiment.
- `exp012` is useful as supplementary epoch-sensitivity evidence.
- Scored replay should not be claimed to consistently improve performance.
- Replay may reduce forgetting under the lightweight setting, even when it does not improve final test accuracy.

## 13. Limitations
- `exp011` only uses three seeds.
- `exp012` is single-seed.
- Earlier experiments use only one or two seeds.
- The saved subset sizes are lightweight: train=`100`, validation=`30`, test=`50`.
- Only three tasks are used in the continual learning sequence.
- Only one model appears in the saved files: `google/flan-t5-small`.
- Many experiments use only `1` epoch; `exp012` varies epochs but not seeds.
- Replay hyperparameter search is limited.
- `predicted_utility` uses small reviewed-label scorers and remains exploratory.
- Utility labels are proxy labels rather than exact causal replay utility.
- The audited CSV is used as replay / utility review provenance, not as a full train/validation/test replacement.
- No statistical significance analysis appears in the saved experiment files.
- Results should not be generalized too strongly.

## 14. Suggested Current Project Conclusion
`exp011` should be used as the main Results experiment. `exp012` should be used as supplementary epoch sensitivity analysis. `exp001-exp010` should be described as preliminary or development evidence that motivated the cleaner robustness comparison.

The overall conclusion should be mixed and cautious: scored replay is slightly stronger than random replay on average, but no replay has the highest mean test accuracy in `exp011`; replay may help forgetting more than final accuracy.

## 15. Paper-Writing Recommendation
- Use `exp011` as the primary quantitative result.
- Use `exp012` as supplementary evidence.
- Do not write that scored replay consistently improves performance.
- Clearly state that the audited CSV is used as reviewed replay / utility provenance rather than a full benchmark replacement.
- Present `exp001-exp010` as preliminary/development evidence and avoid over-weighting single-seed wins.

## 16. Reproducibility Checklist
- [x] exp001 summary available
- [x] exp002 summary available
- [x] exp003 summary available
- [x] exp004 summary available
- [x] exp005 summary available
- [x] exp006 summary available
- [x] exp007 summary available
- [x] exp008 summary available
- [x] exp009 summary available
- [x] exp010 summary available
- [x] exp011 summary available
- [x] exp012 summary available
- [x] exp011 seed comparison CSV available
- [x] exp012 epoch comparison CSV available
- [x] run_config.json files available
- [x] metrics files available
- [x] experiment_index.md available
- [x] latest main.py compatibility checked against exp001-exp012 configurations
- [x] overall_experiment_analysis.md updated with exp010-exp012

## 17. Notes
- This document was updated from saved experiment files only.
- No experiments were rerun while updating this report.
- No code files were modified while updating this report.
- `exp010` is the direct random replay baseline comparison.
- `exp011` is the current main robustness experiment.
- `exp012` is the current epoch-sensitivity supplement.
- Generation timestamp: `2026-06-05`.
