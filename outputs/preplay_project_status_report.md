# PRE-PLAY Project Status Report

## 1. Current Project Goal
The current project goal is to predict which training samples are useful for future replay in continual learning. Concretely, the codebase now supports collecting sample-level training signals, exporting replay-utility proxies, adding human review labels, training a lightweight utility scorer, and using that scorer to drive a `predicted_utility` replay strategy during sequential fine-tuning.

## 2. Proposal Alignment
| Proposal Requirement | Current Implementation | Evidence from Experiments | Status |
| --- | --- | --- | --- |
| sequential fine-tuning | Multi-task sequential runs over `sciq -> arc_challenge -> boolq` and reverse order | `exp001`, `exp002`, `exp003`, `exp004`, `exp005` | Implemented |
| replay vs no-replay | Replay-enabled and no-replay baselines share the same training setup | `exp001`, `exp003`, `exp005` | Implemented |
| task order sensitivity | Same task set tested in original and reversed order | `exp002` vs `exp001` | Implemented |
| replay buffer size | Buffer-size ablation at fixed task order | `exp004` | Implemented |
| seed robustness | Same replay setting retested under another seed, including predicted_utility scorer-source runs | `exp005`, `exp009` | Partial |
| sample-level signals | `sample_signals/` saved for replay and no-replay runs | PRE-PLAY artifacts begin in `exp006` and continue in `exp007`, `exp008`, `exp009` | Implemented |
| human review labels | Combined, deduplicated, and expanded review files with `keep` / `reject` / `unsure` decisions | `exp006/human_review/`, especially `replay_review_candidates_reviewed_fixed.csv` and `replay_review_candidates_expanded_review_target.csv` | Implemented |
| utility scorer training | Reviewed labels exported and used to train experiment-level scorers | `exp006/scorer_outputs/` and `exp006/scorer_outputs_expanded/` | Implemented |
| predicted utility replay | `predicted_utility` strategy loads scorer artifacts and selects replay samples | `exp007`, `exp008`, `exp009` | Implemented |
| scorer-source comparison | `predicted_utility` can load explicit scorer and feature paths from CLI, enabling direct comparison of scorer sources | `exp008`, `exp009` | Implemented |

## 3. Experiment Timeline
| Experiment | Purpose | Main Comparison | Main Finding | Relevance to Proposal |
| --- | --- | --- | --- | --- |
| `exp001` | Original-order baseline | replay vs no-replay | replay is mixed and slightly worse on average | establishes baseline continual-learning setup |
| `exp002` | Task-order sensitivity | reverse-order replay vs no-replay | replay pattern changes under reversed order | shows task order matters |
| `exp003` | Additional baseline check | replay vs no-replay | similar mixed result to `exp001` | confirms baseline instability |
| `exp004` | Buffer-size ablation | no-replay vs buffer 30 / 60 / 100 | buffer size matters, but larger is not always better | tests replay budget design |
| `exp005` | Seed robustness | no-replay vs buffer 30 at seed 123 | replay looks stronger under another seed | shows seed sensitivity |
| `exp006` | Strategy comparison plus PRE-PLAY artifact generation | no-replay vs random / loss / surprise | loss and surprise beat random on average test; PRE-PLAY pipeline artifacts generated | first full PRE-PLAY data-generation step |
| `exp007` | First predicted utility replay test | no-replay vs random / surprise / predicted utility | predicted utility has best average validation, but not best average test | first end-to-end PRE-PLAY prototype run |
| `exp008` | Scorer-source comparison | previous scorer vs expanded scorer, with references | previous scorer is better on validation and forgetting; expanded scorer is better on average test | tests whether more reviewed labels improve predicted utility |
| `exp009` | Predicted utility robustness check | no-replay vs random / surprise / predicted utility previous / predicted utility expanded | scorer-source ranking changes under seed 123 | tests whether predicted utility scorer-source conclusions are seed-stable |

## 4. Current Main Findings
- Replay is not consistently better than no-replay across the saved experiments.
- Replay buffer size matters, but the effect is not monotonic. `Buffer 30` is the best replay setting in the current buffer sweep.
- Changing the seed changes the result materially. `exp005` is more favorable to replay than the seed-42 runs.
- Replay selection strategies differ. In `exp006`, loss and surprise outperform random on average test accuracy.
- Predicted utility replay improves some metrics but not all. In `exp007`, it has the best average validation accuracy, but not the best average test accuracy.
- The previous 57-label scorer and the expanded 113-label scorer give mixed results. In `exp008`, the previous scorer ties the best validation and forgetting behavior, while the expanded scorer gives the best average test accuracy.
- `exp009` shows that the scorer-source pattern is not stable yet. Under seed `123`, the expanded scorer is best on validation and forgetting, but random replay is best on average test accuracy.

## 5. PRE-PLAY Core Pipeline Status
| Pipeline Step | State | Output Files | Limitations |
| --- | --- | --- | --- |
| `sample_signals` | Implemented | run-level `sample_signals/*.csv` in `exp006`, `exp007`, `exp008`, `exp009` | signals are still lightweight and based on one-epoch small-subset runs |
| `utility_labels` | Implemented | run-level `utility_labels/*.csv`; experiment-level `exp006/utility_labels/utility_labels_reviewed.csv` | labels are proxies or human-reviewed decisions, not exact causal future utility |
| `human_review` | Implemented | `exp006/human_review/replay_review_candidates_reviewed_fixed.csv`, `replay_review_candidates_reviewed_by_strategy.csv`, `replay_review_candidates_expanded_review_target.csv` | only 132 reviewed rows total, with 113 usable `keep` / `reject` labels |
| `utility_scorer` | Implemented | `exp006/scorer_outputs/`, `exp006/scorer_checkpoints/`, `exp006/scorer_outputs_expanded/`, `exp006/scorer_checkpoints_expanded/` | current model is a simple `LogisticRegression`; scorer quality is still uncertain |
| `predicted_utility` replay | Implemented | `exp007`, `exp008`, and `exp009` run folders, `run_config.json` scorer paths, `replay_selection_comparison/*.csv` | current evidence is exploratory and mixed rather than decisive |

Pipeline summary:
`sample_signals -> utility_labels -> human_review -> utility_scorer -> predicted_utility replay` is now working end to end. The main weakness is not missing infrastructure; it is label scale, feature quality, and experimental robustness.

## 6. exp007 to exp009 Interpretation
`exp007` is the first direct PRE-PLAY test. It shows that `predicted_utility` can load the exp006 scorer correctly, avoid fallback, and produce the best average validation accuracy (`0.4111`). But it does not produce the best average test accuracy; surprise replay remains stronger on test (`0.3667` vs `0.3333`).

`exp008` extends that result by comparing scorer sources. The previous scorer, trained on 57 usable labels, ties the best average validation accuracy (`0.4000`) and the best maximum forgetting (`0.0000`). The expanded scorer, trained on 113 usable labels, produces the best average test accuracy (`0.4267`) but worse validation accuracy and worse forgetting. The result is therefore mixed. More reviewed labels changed the policy, but did not create a uniformly better scorer.

`exp009` repeats that scorer-source comparison under seed `123`. Here, the expanded scorer becomes the best average-validation setting (`0.4111`) and the lowest-forgetting setting (`0.0333`), while random replay becomes the best average-test setting (`0.3600`). That reversal means the `exp008` scorer-source pattern is not stable yet.

## 7. Current Limitations
- Only two seeds have been used for `predicted_utility` experiments so far: `42` and `123`.
- All saved runs use one epoch.
- Train / val / test subset sizes are lightweight: `100 / 30 / 50`.
- The human-reviewed dataset is still small by supervised-learning standards.
- Human review labels are not exact causal future utility labels.
- The current scorer is a simple `LogisticRegression`.
- The current feature set may be insufficient for a stronger utility predictor.
- There is no statistical significance analysis in the saved experiment suite.

## 8. Does the Current Code Meet the Proposal?
Yes, at the prototype level.

- It meets the baseline and prototype implementation requirements.
- It now implements a first working PRE-PLAY prototype.
- It does not yet provide strong evidence that `predicted_utility` replay is consistently superior.

The code and experiment structure align with the proposal's intended workflow. The remaining gap is empirical strength, not missing core functionality.

## 9. Recommended Next Step
Stop running new experiments temporarily and write the project report section based on `exp001` to `exp009`, unless more seeds are specifically required.

The current code and experiment suite already cover the baseline replay checks, the PRE-PLAY data pipeline, scorer training, scorer-source comparison, and one additional predicted_utility robustness seed. The bottleneck is now interpretation quality rather than missing runs.

## 10. If Another Robustness Check Is Needed Later
- keep the task sequence fixed at `sciq -> arc_challenge -> boolq`
- compare:
  - no-replay
  - random replay
  - surprise replay
  - predicted_utility with previous scorer
  - predicted_utility with expanded scorer
- add one new seed only if the report requires more evidence on stability

That would extend the existing `exp008` and `exp009` template without changing the project structure.

## 11. Final Project-Level Conclusion
The project now has a working PRE-PLAY prototype: the code can generate sample-level replay signals, export and review replay candidates, train reviewed-label utility scorers, and run `predicted_utility` replay from saved scorer artifacts. The experiments show that this pipeline is operational and capable of changing replay behavior in measurable ways. At the same time, the empirical result is still mixed. Replay itself is not consistently superior across the benchmark suite, and the current predicted-utility scorer does not yet dominate strong baselines such as surprise replay or random replay across seeds. The right project-level conclusion is that the implementation now matches the proposal's prototype goals, but stronger evidence still depends on better review labels, better features, and more robust evaluation.
