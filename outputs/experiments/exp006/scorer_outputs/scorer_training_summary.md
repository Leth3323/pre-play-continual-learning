# Scorer Training Summary

## Training Setup
- Label source: `human_review`
- Model: `logistic_regression`
- Split method: `train_test`
- Training rows loaded: 57
- Rows used for fitting: 39
- Rows used for evaluation: 18

## Label Distribution
- Positive labels (`keep`): 22
- Negative labels (`reject`): 35

## Feature Columns
- Numeric: `early_loss`, `surprise_score`, `replay_selection_score`, `utility_score`, `token_count`
- Categorical: `task_name`, `auto_replay_tag`, `appeared_in_strategies`

## Metrics
- `accuracy`: 0.8333
- `precision`: 0.7500
- `recall`: 0.8571
- `f1`: 0.8000
- `roc_auc`: 0.8052

## Outputs
- Metrics: `scorer_metrics.csv`
- Predictions: `scorer_predictions.csv`
- Feature metadata: `feature_columns.json`
- Model checkpoint: `scorer_model.joblib`

## Warning
Small labelled dataset: 57 rows available for training and evaluation.
