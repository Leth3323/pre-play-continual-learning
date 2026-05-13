# Scorer Training Summary

## Training Setup
- Label source: `expanded_human_review_final_replay_decision`
- Model: `logistic_regression`
- Split method: `train_test`
- Training rows loaded: 113
- Rows used for fitting: 79
- Rows used for evaluation: 34

## Label Distribution
- Positive labels (`keep`): 53
- Negative labels (`reject`): 60

## Feature Columns
- Numeric: `early_loss`, `surprise_score`, `replay_selection_score`, `utility_score`, `token_count`
- Categorical: `task_name`, `auto_replay_tag`, `appeared_in_strategies`, `review_priority`

## Metrics
- `accuracy`: 0.7353
- `precision`: 0.7333
- `recall`: 0.6875
- `f1`: 0.7097
- `roc_auc`: 0.7778

## Outputs
- Metrics: `scorer_metrics.csv`
- Predictions: `scorer_predictions.csv`
- Feature metadata: `feature_columns.json`
- Model checkpoint: `scorer_model.joblib`
