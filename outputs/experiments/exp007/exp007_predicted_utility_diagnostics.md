# exp007 Predicted Utility Diagnostics

## 1. Scope
This report diagnoses the `predicted_utility` replay run in `exp007` using only saved files from:

- `outputs/experiments/exp007/exp007_no_replay_seed42_3tasks/`
- `outputs/experiments/exp007/exp007_replay_random_buffer30_seed42_3tasks/`
- `outputs/experiments/exp007/exp007_replay_surprise_buffer30_seed42_3tasks/`
- `outputs/experiments/exp007/exp007_replay_predicted_utility_buffer30_seed42_3tasks/`
- `outputs/experiments/exp006/scorer_outputs/`
- `outputs/experiments/exp006/human_review/`
- `outputs/experiments/exp006/utility_labels/`

## 2. Did `predicted_utility` actually use the exp006 scorer?
Yes.

Saved evidence:

| Check | Result |
| --- | --- |
| `run_config.json` strategy | `predicted_utility` |
| `run_config.json` replay buffer size | `30` |
| `run_config.json` seed | `42` |
| Scorer model path | `outputs/experiments/exp006/scorer_checkpoints/scorer_model.joblib` |
| Feature columns path | `outputs/experiments/exp006/scorer_outputs/feature_columns.json` |
| `feature_columns.json` model name | `logistic_regression` |
| `feature_columns.json` label source | `human_review` |
| Reviewed utility-label rows available in `utility_labels_reviewed.csv` | `57` usable labels (`22` keep / `35` reject) |

The saved config points to the exp006 scorer artifacts directly, and the exp006 feature metadata records a human-review-trained `logistic_regression` scorer.

## 3. Did it silently fall back to random, loss, or surprise?
No silent fallback is visible in the saved files.

Evidence:

1. Every `replay_selection_comparison/*.csv` file in the predicted-utility run records `replay_selection_strategy=predicted_utility`.
2. The predicted-utility run selected a different set of samples than the random run.
3. The predicted-utility run also selected a different set of samples than the surprise run.
4. Within the predicted-utility selection files, `replay_selection_score` is not numerically identical to either `loss` or `surprise_score`.

Direct comparison against a loss-based exp007 run is not possible because exp007 did not include a loss replay setting. But the saved predicted-utility score column is distinct from the raw `loss` column, which rules out a direct loss fallback.

## 4. Metric Comparison

### Average validation accuracy
| Setting | Average validation accuracy |
| --- | --- |
| No Replay | `0.3889` |
| Random | `0.3889` |
| Surprise | `0.3889` |
| Predicted Utility | `0.4111` |

### Average test accuracy
| Setting | Average test accuracy |
| --- | --- |
| No Replay | `0.3467` |
| Random | `0.3533` |
| Surprise | `0.3667` |
| Predicted Utility | `0.3333` |

### Maximum forgetting
| Setting | Maximum forgetting |
| --- | --- |
| No Replay | `0.0000` |
| Random | `0.1000` |
| Surprise | `0.0667` |
| Predicted Utility | `0.0000` |

Summary:

- `predicted_utility` has the best average validation accuracy.
- `predicted_utility` does not beat surprise on average test accuracy.
- `predicted_utility` ties no-replay for the lowest saved maximum forgetting.

## 5. Predicted-Utility Score Diagnostics

### Saved score files
| Location | Count | Notes |
| --- | --- | --- |
| `replay_scores/*.csv` | `3` | These files contain base `loss` and `surprise_score` signals. |
| `replay_selection_comparison/*.csv` | `3` | These files contain the actual `replay_selection_score` and `was_selected_for_replay` fields used for selection. |

For the predicted-utility run, the actual predicted replay score is stored in `replay_selection_comparison/*.csv`, not in `replay_scores/*.csv`.

### Score range and mean
| Task | Candidates | Selected | Score range | Mean score | Mean selected score | Mean unselected score |
| --- | --- | --- | --- | --- | --- | --- |
| `sciq` | `100` | `30` | `0.2230` to `0.8943` | `0.4495` | `0.5795` | `0.3938` |
| `arc_challenge` | `100` | `30` | `0.0693` to `0.8386` | `0.3053` | `0.4375` | `0.2487` |
| `boolq` | `100` | `30` | `0.2824` to `0.5702` | `0.3503` | `0.4503` | `0.3074` |
| **Overall** | `300` | `90` | `0.0693` to `0.8943` | `0.3684` | `0.4891` | `0.3166` |

The selected examples consistently have higher mean predicted scores than the unselected examples, which indicates the scorer was actively ranking samples rather than selecting arbitrarily.

### Top selected examples
Top scored selected examples across the predicted-utility run:

| Task | sample_id | Predicted score | exp006 review decision |
| --- | --- | --- | --- |
| `sciq` | `sciq_train_000001` | `0.8943` | `reject` |
| `sciq` | `sciq_train_000032` | `0.8477` | `reject` |
| `arc_challenge` | `arc_challenge_train_000045` | `0.8386` | `keep` |
| `sciq` | `sciq_train_000055` | `0.8136` | `reject` |
| `sciq` | `sciq_train_000079` | `0.7903` | `keep` |
| `arc_challenge` | `arc_challenge_train_000034` | `0.7132` | `reject` |
| `sciq` | `sciq_train_000037` | `0.7123` | `keep` |
| `sciq` | `sciq_train_000090` | `0.6233` | `keep` |
| `sciq` | `sciq_train_000089` | `0.6124` | `unsure` |

This high-score list is mixed: some top selections align with exp006 `keep` decisions, but several are known `reject` examples. That indicates the scorer is not yet cleanly calibrated.

## 6. Replay Buffer Overlap
Exact overlap can be computed from `replay_selection_comparison/*.csv` because each file records `was_selected_for_replay` for every candidate.

### Predicted utility vs random
| Task | Overlap count | Jaccard overlap |
| --- | --- | --- |
| `sciq` | `9 / 30` | `0.1765` |
| `arc_challenge` | `13 / 30` | `0.2766` |
| `boolq` | `7 / 30` | `0.1321` |
| **Overall** | `29 / 90` | `0.1921` |

### Predicted utility vs surprise
| Task | Overlap count | Jaccard overlap |
| --- | --- | --- |
| `sciq` | `27 / 30` | `0.8182` |
| `arc_challenge` | `21 / 30` | `0.5385` |
| `boolq` | `27 / 30` | `0.8182` |
| **Overall** | `75 / 90` | `0.7143` |

Interpretation:

- `predicted_utility` is strongly different from `random`.
- `predicted_utility` overlaps heavily with `surprise`, especially on `sciq` and `boolq`.
- The main divergence from surprise happens on `arc_challenge`, where only `21` of `30` selected samples overlap.

## 7. Alignment with exp006 Human Review Labels
The predicted-utility selected set contains `90` samples total. Of those, `33` match samples that were manually reviewed in exp006, and `57` do not have human review labels yet.

### Predicted-utility selected samples vs exp006 review decisions
| Review bucket | Count |
| --- | --- |
| `keep` | `17` |
| `reject` | `14` |
| `unsure` | `2` |
| `not_reviewed` | `57` |

### By task
| Task | keep | reject | unsure | not reviewed |
| --- | --- | --- | --- | --- |
| `sciq` | `7` | `3` | `2` | `18` |
| `arc_challenge` | `6` | `8` | `0` | `16` |
| `boolq` | `4` | `3` | `0` | `23` |

### Comparison with other replay strategies on the same exp006 reviewed subset
| Strategy | keep | reject | unsure | not reviewed |
| --- | --- | --- | --- | --- |
| Random | `15` | `10` | `2` | `63` |
| Surprise | `17` | `16` | `1` | `56` |
| Predicted Utility | `17` | `14` | `2` | `57` |

Relative to surprise:

- `predicted_utility` keeps the same number of reviewed `keep` samples (`17`).
- `predicted_utility` selects slightly fewer reviewed `reject` samples (`14` vs `16`).
- The difference is modest, not dramatic.

Relative to random:

- `predicted_utility` is selecting a substantially different sample set.
- But the reviewed-label advantage is not clean enough to say it is clearly better calibrated than random from the current reviewed subset alone.

## 8. Does `predicted_utility` select meaningfully different samples?
Yes, but the answer depends on the baseline.

Against random:

- Yes, clearly.
- Only `29` of `90` selected samples overlap overall.
- That means `61` selected samples differ between the two strategies.

Against surprise:

- Only partly.
- `75` of `90` selected samples overlap overall.
- The strategy is mostly shadowing surprise on `sciq` and `boolq`, but it makes more noticeable changes on `arc_challenge`.

One useful detail is the `predicted_utility`-only vs `surprise`-only swap set:

- `predicted_utility` selects `15` samples that surprise does not.
- `surprise` selects `15` samples that predicted utility does not.
- Among reviewed samples in those swap sets, the predicted-only side includes `3` reviewed rejects and `1` unsure, while the surprise-only side includes `5` reviewed rejects.

That suggests the scorer is making a real ranking change and may be slightly reducing obviously bad surprise selections, but the effect is still small and not yet translated into better test accuracy.

## 9. Diagnostic Conclusion
The saved files support these conclusions:

1. `predicted_utility` did use the exp006 scorer correctly.
2. It did not silently fall back to random, loss, or surprise.
3. It improves average validation accuracy over no-replay, random, and surprise.
4. It does not improve average test accuracy over surprise.
5. It ties the lowest maximum forgetting with no replay.
6. It selects a meaningfully different set from random, but a strongly overlapping set with surprise.
7. Its highest-scored selected samples still include several exp006 reviewed rejects, so scorer calibration is still rough.

## 10. Cautious Interpretation
- `predicted_utility` improves average validation accuracy.
- `predicted_utility` does not improve average test accuracy over surprise.
- `predicted_utility` ties the lowest saved maximum forgetting.
- The result is promising enough to justify deeper diagnostics and a stronger follow-up experiment.
- The result is still exploratory because the scorer was trained on only `57` labelled rows, and `57` of the `90` predicted-utility selections were not reviewed in exp006.

Recommendation:

- This does support moving to a stronger `exp008` if the goal is a better controlled follow-up.
- It does **not** support claiming that `predicted_utility` is already better than surprise.
