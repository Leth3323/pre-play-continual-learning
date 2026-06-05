# Final Audited Dataset Report

## Source
- Source root: `/Users/chenxiangkai/PycharmProjects/Pre_play`
- Source CSV files were read only; no source CSV files were modified.

## Output Files
- `final_reviewed_dataset.csv`
- `reviewed_gold.csv`
- `review_conflicts.csv`
- `audit_report.md`

## Counts
- Unique records: 300
- Human-reviewed gold records: 132
- Auto-reviewed records: 168
- Conflict rows: 0

## Review Status Counts
- `auto_reviewed`: 168
- `human_reviewed`: 132

## Final Decision Counts
- `keep`: 111
- `reject`: 170
- `unsure`: 19

## Decision Source Counts
- `auto_expanded_scorer`: 168
- `human_expanded_review`: 132

## Replay Strategy Row Counts
- `loss`: 300
- `predicted_utility`: 1500
- `random`: 1200
- `surprise`: 2400

## Method
- `human_expanded_review` records take precedence over other reviewed sources.
- `human_fixed_review` and `human_utility_label_review` are used only when no expanded review exists.
- Remaining records are assigned by the saved expanded human-review scorer.
- Auto-reviewed records are explicitly marked with `review_status=auto_reviewed` and include `decision_confidence`.
