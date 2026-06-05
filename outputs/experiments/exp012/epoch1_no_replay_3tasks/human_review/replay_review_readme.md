# Replay Review Guide

This export is a human-review queue built from proxy utility labels (`proxy_replay_selection_v1`).
It is not an exact causal estimate of future replay utility.

## Automatic Tags
- `high_value_candidate`: selected for replay by the proxy label and also high on utility_score within the same task.
- `selected_for_replay`: selected for replay by the proxy label, but not in the top 30 percent on utility_score within the same task.
- `needs_human_review`: not selected for replay by the proxy label, but high on early_loss or surprise_score within the same task.
- `low_priority`: none of the higher-priority conditions were triggered.

## Manual Review
- `replay_review_candidates.csv` is the original auto-generated export.
- `replay_review_candidates_reviewed.csv` is the file intended for manual editing.
- Edit only `replay_review_candidates_reviewed.csv`.
- Keep `replay_review_candidates.csv` unchanged for reproducibility.
- Read the input and target text together.
- Check whether the sample looks useful for future replay, low quality, redundant, or narrowly task-specific.
- If an existing reviewed copy already contains non-pending review statuses, the exporter preserves it and writes a timestamped reviewed copy instead.
- Update `human_review_status`, `human_review_label`, `human_review_notes`, and `final_replay_decision` directly in `replay_review_candidates_reviewed.csv`.

## Recommended `human_review_label` Values
- `useful_for_replay`
- `low_quality`
- `ambiguous`
- `redundant`
- `task_specific`
- `general_knowledge`
- `unsure`

## Recommended `final_replay_decision` Values
- `keep`
- `reject`
- `unsure`