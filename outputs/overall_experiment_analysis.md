# Overall Experiment Analysis

## 1. Project-Level Experiment Overview
These saved experiments study replay-based continual learning on a lightweight sequence of QA tasks: `sciq`, `arc_challenge`, and `boolq`. The core comparison is replay versus no-replay, with additional checks on task order, replay buffer size, seed robustness, replay selection strategy, exploratory `predicted_utility` replay, scorer-source comparison for `predicted_utility`, and a second-seed robustness check for the predicted_utility scorer-source pattern. After the PRE-PLAY core infrastructure was added, `exp006` generated `sample_signals`, `utility_labels`, human-review candidate exports, and two experiment-level reviewed-label scorers, `exp007` used the first of those scorers for the first `predicted_utility` replay run, `exp008` compared the previous and expanded scorer sources directly, and `exp009` repeated that scorer-source comparison at seed `123`.

## 2. Experiment Roadmap
| Experiment | Purpose | Task Order | Main Comparison | Seed | Main Output Summary |
| --- | --- | --- | --- | --- | --- |
| exp001 | Original-order baseline | sciq -> arc_challenge -> boolq | Replay vs no-replay | 42 | [exp001_summary.md](experiments/exp001/exp001_summary.md) |
| exp002 | Reverse-order sensitivity check | boolq -> arc_challenge -> sciq | Replay vs no-replay under reversed order | 42 | [exp002_summary.md](experiments/exp002/exp002_summary.md) |
| exp003 | Additional baseline-style comparison | sciq -> arc_challenge -> boolq | Replay vs no-replay under the original order again | 42 | [exp003_summary.md](experiments/exp003/exp003_summary.md) |
| exp004 | Replay buffer size ablation | sciq -> arc_challenge -> boolq | No-replay vs Buffer 30 / 60 / 100 | 42 | [exp004_summary.md](experiments/exp004/exp004_summary.md) |
| exp005 | Seed robustness check | sciq -> arc_challenge -> boolq | No-replay vs Buffer 30 at a new seed | 123 | [exp005_summary.md](experiments/exp005/exp005_summary.md) |
| exp006 | Replay selection strategy comparison with PRE-PLAY artifacts | sciq -> arc_challenge -> boolq | No-replay vs Random / Loss / Surprise replay | 42 | [exp006_summary.md](experiments/exp006/exp006_summary.md) |
| exp007 | First predicted_utility replay experiment using the exp006 scorer | sciq -> arc_challenge -> boolq | No-replay vs Random / Surprise / Predicted Utility replay | 42 | [exp007_summary.md](experiments/exp007/exp007_summary.md) |
| exp008 | Predicted_utility scorer-source comparison | sciq -> arc_challenge -> boolq | No-replay vs Random / Surprise vs Predicted Utility with Previous / Expanded scorer | 42 | [exp008_summary.md](experiments/exp008/exp008_summary.md) |
| exp009 | Predicted_utility scorer-source robustness check | sciq -> arc_challenge -> boolq | No-replay vs Random / Surprise vs Predicted Utility with Previous / Expanded scorer | 123 | [exp009_summary.md](experiments/exp009/exp009_summary.md) |


## 3. Research Questions Across Experiments
1. Does replay improve retention and final performance?
2. Is replay sensitive to task order?
3. Does replay buffer size affect performance?
4. Is the best buffer-size result stable under another seed?
5. Under the same replay buffer budget, does replay sample selection strategy matter?
6. Can a human-review-trained utility scorer improve replay selection compared with random and surprise?
7. Does the expanded human-review-trained scorer improve predicted_utility replay compared with the previous 57-label scorer?
8. Are the exp008 predicted_utility scorer-source results stable when the random seed changes from 42 to 123?


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
- Relation to exp001: it uses the same task order and reaches a similar overall pattern, with replay still failing to show a consistent advantage.
- Cautious conclusion: this additional baseline-style comparison again gives mixed evidence and leans against a general replay benefit under the saved lightweight setting.

### exp004: Replay Buffer Size Ablation
- Design: one no-replay baseline plus replay buffer sizes `30`, `60`, and `100`, all over `sciq -> arc_challenge -> boolq` at seed `42`.
- Best average validation accuracy among replay settings: `Buffer 30`.
- Best average test accuracy among replay settings: tie across `Buffer 30`, `Buffer 60`, and `Buffer 100`, while no-replay still has the strongest average test accuracy overall.
- Lowest maximum forgetting among replay settings: `Buffer 30`; it also ties the no-replay baseline on maximum saved forgetting.
- Larger buffer size helped monotonically: no, larger buffers did not improve metrics in a monotonic way.
- Cautious conclusion: `Buffer 30` is the most promising replay setting in this saved sweep, but the overall evidence remains mixed.

### exp005: Seed Robustness Check
- Design: no-replay versus `Buffer 30` over `sciq -> arc_challenge -> boolq` at seed `123`.
- Validation result: average replay-minus-no-replay validation delta was `+0.1222`.
- Test result: average replay-minus-no-replay test delta was `+0.1000`, although `boolq` test accuracy still favored no-replay.
- Forgetting result: maximum saved forgetting improved from `0.3333` to `0.2333`.
- Relation to exp004: it is more favorable to `Buffer 30` than the seed `42` comparison in exp004.
- Cautious conclusion: exp005 tentatively supports the Buffer 30 pattern, but the result is still limited to one additional seed.

### exp006: Replay Selection Strategy Comparison
- Design: no-replay, random replay, loss replay, and surprise replay over `sciq -> arc_challenge -> boolq` at seed `42` with replay buffer size `30`.
- Validation result: average final validation accuracy is tied across no-replay, loss replay, and surprise replay at `0.3889`, while random replay is slightly lower at `0.3778`.
- Test result: average final test accuracy is highest for loss replay and surprise replay at `0.3667`, above random replay at `0.3600` and no-replay at `0.3467`.
- Forgetting result: no-replay has the lowest maximum saved forgetting at `0.0000`, while loss replay and surprise replay reduce maximum forgetting relative to random replay (`0.0667` vs `0.1000`).
- PRE-PLAY artifacts: the replay-enabled runs generated `sample_signals`, `utility_labels`, and `human_review` candidate files, and exp006 now also has experiment-level reviewed-label exports and scorer outputs.
- Scorer training result: `utility_scorer.py` was trained from `utility_labels_reviewed.csv` using `human_utility_label`, with 57 labelled rows (`keep`=22, `reject`=35), `LogisticRegression`, accuracy `0.8333`, precision `0.7500`, recall `0.8571`, F1 `0.8000`, and ROC AUC `0.8052`.
- Expanded scorer retraining result: `utility_scorer.py` was retrained from the canonical expanded review file using `final_replay_decision`, with 113 labelled rows (`keep`=53, `reject`=60), `LogisticRegression`, accuracy `0.7353`, precision `0.7333`, recall `0.6875`, F1 `0.7097`, and ROC AUC `0.7778`.
- Cautious conclusion: exp006 gives preliminary evidence that score-based replay selection may matter on average test accuracy, but the evidence remains lightweight and does not justify stronger claims yet.

### exp007: Exploratory Predicted Utility Replay
- Design: no replay, random replay, surprise replay, and `predicted_utility` replay over `sciq -> arc_challenge -> boolq` at seed `42` with replay buffer size `30`.
- Scorer source: the exp006 reviewed-label `LogisticRegression` scorer trained from `57` labelled rows (`keep`=`22`, `reject`=`35`).
- Validation result: average final validation accuracy is highest for predicted utility at `0.4111`, followed by no replay, random replay, and surprise replay tied at `0.3889`.
- Test result: average final test accuracy is highest for surprise replay at `0.3667`, followed by random replay at `0.3533`, no replay at `0.3467`, and predicted utility at `0.3333`.
- Forgetting result: the saved final forgetting values are all `0.0000`, so the final-task forgetting comparison is a tie across settings.
- Validity check: the predicted-utility run recorded exp006 scorer paths in `run_config.json`, and its selection-comparison files contain only `predicted_utility`, with selected sample sets differing from both random and surprise.
- Cautious conclusion: exp007 shows that the PRE-PLAY scorer-loading pipeline works end to end, but the current scorer does not outperform the stronger replay baselines under this lightweight exploratory setting.

### exp008: Predicted Utility Scorer-Source Comparison
- Design: no replay, random replay, surprise replay, `predicted_utility` with the previous 57-label scorer, and `predicted_utility` with the expanded 113-label scorer over `sciq -> arc_challenge -> boolq` at seed `42` with replay buffer size `30`.
- Scorer sources: the previous scorer is the exp006 `LogisticRegression` model trained on 57 usable labels; the expanded scorer is the exp006 `LogisticRegression` model trained on 113 usable labels.
- Validation result: average final validation accuracy is tied between random replay and the previous scorer at `0.4000`, while the expanded scorer is lower at `0.3444`.
- Test result: average final test accuracy is highest for the expanded scorer at `0.4267`, above surprise replay at `0.3667`, random replay at `0.3600`, no replay at `0.3533`, and the previous scorer at `0.3333`.
- Forgetting result: the lowest maximum forgetting is a tie between no replay and the previous scorer at `0.0000`, while the expanded scorer is highest at `0.1333`.
- Validity check: both predicted-utility runs recorded explicit scorer paths in `run_config.json`, saved replay-score CSVs and non-empty replay buffers, and their selection-comparison files contain only `predicted_utility`.
- Cautious conclusion: exp008 is mixed. The expanded scorer improves average test accuracy, but the previous scorer is stronger on average validation accuracy and forgetting, so more reviewed labels did not automatically produce a stronger predicted_utility replay policy.

### exp009: Predicted Utility Seed-123 Robustness Check
- Design: no replay, random replay, surprise replay, `predicted_utility` with the previous 57-label scorer, and `predicted_utility` with the expanded 113-label scorer over `sciq -> arc_challenge -> boolq` at seed `123` with replay buffer size `30`.
- Validation result: average final validation accuracy is highest for the expanded scorer at `0.4111`, above surprise replay at `0.3889`, random replay at `0.3778`, the previous scorer at `0.3778`, and no replay at `0.3444`.
- Test result: average final test accuracy is highest for random replay at `0.3600`, while both predicted_utility runs tie at `0.3533`.
- Forgetting result: the expanded scorer has the lowest maximum forgetting at `0.0333`; no replay is worst at `0.2333`.
- Validity check: both predicted-utility runs recorded explicit scorer paths in `run_config.json`, saved replay-score CSVs and non-empty replay buffers, and their selection-comparison files contain only `predicted_utility`.
- Cautious conclusion: the scorer-source pattern from exp008 did not hold under seed `123`. The expanded scorer becomes strongest on validation accuracy and forgetting, but it does not retain the best average test accuracy. The result is exploratory and seed-sensitive.

## 5. Cross-Experiment Comparison
| Experiment | Replay Setting Tested | Validation Evidence | Test Evidence | Forgetting Evidence | Overall Interpretation |
| --- | --- | --- | --- | --- | --- |
| exp001 | Default replay setting | Average delta `-0.0111` | Average delta `-0.0333` | Max forgetting tie at `0.1000` | Mixed evidence; does not support replay overall |
| exp002 | Default replay setting | Average delta `-0.0111` | Average delta `+0.0133` | Max forgetting tie at `0.0667` | Mixed evidence; task-order-sensitive |
| exp003 | Default replay setting | Average delta `-0.0111` | Average delta `-0.0267` | Max forgetting tie at `0.1000` | Mixed evidence; does not support replay overall |
| exp004 | Buffer 30 / 60 / 100 | Buffer 30 best among replay settings; larger buffers worse | All replay settings trail no-replay on average test accuracy | Buffer 30 best among replay settings and ties no-replay on max forgetting | Mixed evidence; buffer-size-sensitive |
| exp005 | Buffer 30 at seed 123 | Average delta `+0.1222` | Average delta `+0.1000`, but one test task favors no-replay | Max forgetting improves from `0.3333` to `0.2333` | Tentative support for replay, but seed-sensitive |
| exp006 | Random / Loss / Surprise at Buffer 30 | No-replay, loss replay, and surprise replay tie on average validation accuracy | Loss replay and surprise replay are best on average test accuracy | No-replay has the lowest max forgetting; loss and surprise beat random | Preliminary evidence that replay selection strategy may matter |
| exp007 | Random / Surprise / Predicted Utility at Buffer 30 | Predicted utility is best on average validation accuracy | Surprise replay is best on average test accuracy; predicted utility trails random and surprise on average test | Final saved forgetting ties at 0.0000 across settings | The PRE-PLAY pipeline works, but the current scorer is not yet strong enough |
| exp008 | Random / Surprise / Predicted Utility with Previous vs Expanded scorer | Random and Predicted Previous tie for best average validation accuracy; Expanded is lower | Predicted Expanded is best on average test accuracy | No-replay and Predicted Previous tie for lowest max forgetting | Mixed scorer-source evidence; more labels did not automatically improve predicted_utility |
| exp009 | Random / Surprise / Predicted Utility with Previous vs Expanded scorer at seed 123 | Predicted Expanded is best on average validation accuracy | Random is best on average test accuracy; both predicted runs tie below it | Predicted Expanded has the lowest max forgetting | The scorer-source ranking is not stable across seeds |

## 6. Replay Effectiveness Analysis
- Replay helped most clearly in `exp005`, where `Buffer 30` improved average validation accuracy, average test accuracy, and maximum saved forgetting relative to no-replay.
- Replay helped partially in `exp002` on average test accuracy, but not on average validation accuracy or maximum forgetting.
- Replay hurt on average validation and test accuracy in `exp001` and `exp003`.
- Replay did not consistently improve validation accuracy across experiments; only `exp004` `Buffer 30` and `exp005` show clearly positive average validation evidence.
- Replay did not consistently improve test accuracy across experiments; `exp001` and `exp003` are negative on average, `exp002` is only slightly positive, `exp004` replay settings all lag no-replay on average test accuracy, and `exp005` is the strongest positive case.
- Replay did not consistently reduce forgetting; `exp001` and `exp002` tie on maximum forgetting, `exp004` `Buffer 30` only ties the baseline while larger buffers are worse, and `exp005` is the clearest improvement.
- Within the fixed-buffer comparison in `exp006`, loss replay and surprise replay both outperform random replay on average test accuracy, while no-replay still has the lowest maximum forgetting.
- In `exp007`, `predicted_utility` replay runs valid end-to-end with the exp006 scorer, but it does not beat random or surprise on the saved summary metrics.
- In `exp008`, the scorer source clearly matters: the previous scorer is better on average validation accuracy and forgetting, while the expanded scorer is better on average test accuracy.
- In `exp009`, the scorer-source ranking changes again under seed `123`: the expanded scorer is best on average validation accuracy and maximum forgetting, while random replay is best on average test accuracy.
- The saved evidence is not strong enough to claim that replay is generally useful under this lightweight setup.

## 7. Task-Order Sensitivity
`exp001` and `exp002` use the same three tasks but in different orders: `sciq -> arc_challenge -> boolq` versus `boolq -> arc_challenge -> sciq`. Under the original order, replay is slightly worse on average validation and test accuracy. Under the reversed order, replay is still slightly worse on average validation accuracy, but it becomes slightly better on average test accuracy. Maximum saved forgetting is tied in both experiments, but the task-level winners shift across the reversed sequence. Taken together, the saved files suggest that the replay effect is sensitive to task order rather than invariant across the same task set.

## 8. Buffer Size Analysis
`exp004` compares `Buffer 30`, `Buffer 60`, and `Buffer 100` against no-replay. `Buffer 30` is the strongest replay setting on average validation accuracy and on maximum forgetting among replay settings. Average test accuracy is tied across the three replay settings, and all of them remain below the no-replay baseline on average test accuracy. Larger replay buffers therefore do not help monotonically, which makes `Buffer 30` the most sensible replay setting to prioritize next. This result is still limited because the sweep uses one seed, one epoch, one model, and only three buffer sizes.

## 9. Seed Robustness Analysis
`exp004` uses seed `42`, while `exp005` retests `Buffer 30` at seed `123`. At seed `42`, `Buffer 30` improves average validation accuracy slightly, does not improve average test accuracy, and only ties no-replay on maximum forgetting. At seed `123`, `Buffer 30` improves average validation accuracy, improves average test accuracy, and reduces maximum saved forgetting relative to no-replay. That shift makes `exp005` more favorable to replay than `exp004`, but it also suggests seed sensitivity because the strength of the replay effect changes noticeably across the two saved seeds.

The same issue appears in the PRE-PLAY-specific runs. `exp008` and `exp009` keep the same task order, scorer sources, and replay budget, then change only the seed from `42` to `123`. Under seed `42`, the expanded scorer is best on average test accuracy and the previous scorer is best on average validation accuracy and forgetting. Under seed `123`, the expanded scorer becomes best on average validation accuracy and forgetting, while random replay becomes best on average test accuracy. That reversal is direct evidence that the current predicted_utility result is seed-sensitive and not yet stable.

## 10. Main Findings
- Replay was operational in replay-enabled runs because replay score CSV files and non-empty replay buffer snapshots were saved.
- `exp001` and `exp003` do not show average validation or average test improvements from the default replay setting.
- Reversing task order in `exp002` changes the replay pattern, especially on test accuracy, which suggests task-order sensitivity.
- In `exp004`, `Buffer 30` is the strongest replay setting on average validation accuracy and the lowest maximum forgetting among replay settings.
- Larger replay buffers do not provide a monotonic benefit in `exp004`.
- No-replay remains the strongest average test setting in `exp004` despite the replay buffer sweep.
- `exp005` is more favorable to `Buffer 30` than the seed `42` result, but one test task still favors no-replay and only two seeds are available overall.
- `exp006` is the first experiment after the PRE-PLAY core infrastructure was added, and it successfully generated `sample_signals`, `utility_labels`, and `human_review` candidate files for replay-enabled runs.
- In `exp006`, loss replay and surprise replay both exceed random replay on average test accuracy, which is preliminary evidence that score-based replay selection may matter.
- In `exp006`, the first reviewed-label scorer was trained successfully.
- In `exp007`, that scorer was used for the first `predicted_utility` replay run, but the result remained exploratory and did not beat the stronger replay baselines.
- In `exp008`, comparing the previous and expanded scorer sources produced a mixed result rather than a clean win for the expanded scorer.
- In `exp009`, changing only the seed to `123` changes the scorer-source ranking again, which is direct evidence that the current predicted_utility result is not stable yet.

## 11. Limitations
- Only two seeds are tested so far: `42` and `123`.
- Only two predicted_utility seeds are available so far: `42` and `123`.
- All saved experiments use `1` epoch.
- The saved subset sizes are lightweight: train=`100`, val=`30`, test=`50`.
- Only three tasks are used in the continual learning sequence.
- Only one model appears in the saved files: `google/flan-t5-small`.
- Replay hyperparameter search is limited; only replay/no-replay and one small buffer-size sweep are saved.
- `exp007` is exploratory because it used the earlier scorer trained on only 57 labelled rows.
- `exp008` is also exploratory because it compares two small reviewed-label scorers under one seed and one lightweight training setting.
- Utility labels are proxy labels rather than exact causal replay utility.
- Human review is still limited in exp006 even after expansion: the canonical expanded review file has 132 reviewed rows and 113 usable `keep` / `reject` labels.
- The expanded scorer in exp006 still uses a small dataset by standard supervised-learning expectations, and its held-out metrics are lower than the initial 57-label scorer.
- No statistical significance analysis appears in the saved experiment files.
- These results come from lightweight saved settings and should not be generalized too strongly.

## 12. PRE-PLAY Extension: exp006 to exp009
- `exp006` compares replay selection strategies under a fixed replay buffer budget.
- It is the first experiment after the PRE-PLAY core infrastructure was added.
- It generated `sample_signals`, `utility_labels`, `human_review` candidate files, reviewed-label exports, and experiment-level scorer outputs.
- `utility_scorer.py` was trained from `utility_labels_reviewed.csv` using `human_utility_label`, with 57 labelled rows (`keep`=22, `reject`=35) and `LogisticRegression`.
- The saved scorer reached accuracy `0.8333`, precision `0.7500`, recall `0.8571`, F1 `0.8000`, and ROC AUC `0.8052`.
- The canonical expanded review file was then used to retrain an expanded scorer on 113 labelled rows (`keep`=53, `reject`=60), again with `LogisticRegression`.
- The expanded scorer reached accuracy `0.7353`, precision `0.7333`, recall `0.6875`, F1 `0.7097`, and ROC AUC `0.7778`, so more review labels did not automatically improve held-out metrics.
- `exp007` is the first `predicted_utility` replay experiment using that exp006 scorer.
- `exp007` still reflects the earlier 57-label scorer, not the newer expanded scorer.
- The predicted-utility run in exp007 was valid end to end: it recorded the exp006 scorer paths, used `predicted_utility` strategy labels in the selection-comparison files, and did not silently match the random or surprise selections.
- On saved metrics, exp007 did not show a performance win for the current scorer over random or surprise, so the current result should be treated as exploratory rather than confirmatory.
- `exp008` then compared `predicted_utility` replay under both scorer sources, with `no_replay`, `random`, and `surprise` as references.
- In exp008, the previous scorer tied the best average validation accuracy and the best maximum forgetting, while the expanded scorer produced the strongest average test accuracy.
- The scorer-source comparison in exp008 is mixed, so the current project state does not justify treating the expanded scorer as a clear upgrade yet.
- `exp009` repeated that scorer-source comparison at seed `123`.
- In exp009, the expanded scorer became the best average-validation and lowest-forgetting setting, while random replay became the best average-test setting.
- The exp008 pattern therefore did not replicate cleanly under the new seed.
- The next step is to consolidate the project report around `exp001` to `exp009`, then only return to more seeds or stronger feature design if that extra evidence is specifically needed.


## 13. Suggested Current Project Conclusion
Across the current lightweight experiments, replay still does not show a uniformly consistent advantage. `Buffer 30` remains a sensible replay setting to prioritize, exp006 established the PRE-PLAY review and scorer-training pipeline, exp007 confirmed that the reviewed-label scorer can drive a real `predicted_utility` replay run, and exp008 plus exp009 showed that scorer source materially changes predicted_utility behavior but that the ranking is not stable across seeds. The expanded 113-label scorer did not become a clean upgrade: it improved average test accuracy in exp008, then became the best average-validation and lowest-forgetting setting in exp009, while random replay remained the strongest average-test setting under seed `123`. More seeds, stronger training settings, better feature design, and a larger high-quality reviewed label set are still needed before making a strong project-level claim.


## 14. Reproducibility Checklist
- [x] exp001 summary available
- [x] exp002 summary available
- [x] exp003 summary available
- [x] exp004 summary available
- [x] exp005 summary available
- [x] exp006 summary available
- [x] exp007 summary available
- [x] exp008 summary available
- [x] exp009 summary available
- [x] run_config.json files available
- [x] metrics files available
- [x] experiment_index.md available
- [x] overall_experiment_analysis.md generated

## 15. Notes
- This document was generated from saved experiment files.
- `exp006` is the first saved experiment after the PRE-PLAY core infrastructure was added.
- `exp007` is the first saved `predicted_utility` replay experiment using the exp006 reviewed-label scorer.
- `exp008` is the first saved scorer-source comparison for `predicted_utility` replay.
- `exp009` is the saved seed-123 robustness check for the scorer-source comparison.
- No `exp001` to `exp008` training was rerun while updating this report.
- Generation timestamp: `2026-05-09`.
