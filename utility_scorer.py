from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


NUMERIC_FEATURE_COLUMNS = [
    "early_loss",
    "surprise_score",
    "replay_selection_score",
    "utility_score",
    "token_count",
]
CATEGORICAL_FEATURE_COLUMNS = ["task_name", "auto_replay_tag", "appeared_in_strategies"]
RAW_FEATURE_COLUMNS = [*NUMERIC_FEATURE_COLUMNS, *CATEGORICAL_FEATURE_COLUMNS]
EXPANDED_CATEGORICAL_FEATURE_COLUMNS = [*CATEGORICAL_FEATURE_COLUMNS, "review_priority"]
EXPANDED_RAW_FEATURE_COLUMNS = [*NUMERIC_FEATURE_COLUMNS, *EXPANDED_CATEGORICAL_FEATURE_COLUMNS]


def _safe_read_csv(input_path: Path) -> pd.DataFrame:
    if not input_path.exists() or input_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(input_path)
    except EmptyDataError:
        return pd.DataFrame()


def _get_human_review_label_paths(human_review_dir: Path) -> list[Path]:
    reviewed_paths = sorted(human_review_dir.glob("replay_review_candidates_reviewed*.csv"))
    if reviewed_paths:
        return reviewed_paths

    return sorted(
        path
        for path in human_review_dir.glob("replay_review_candidates*.csv")
        if not path.name.startswith("replay_review_candidates_reviewed")
    )


def _get_reviewed_utility_label_paths(utility_label_dir: Path) -> list[Path]:
    return sorted(utility_label_dir.glob("utility_labels_reviewed*.csv"))


def _get_proxy_utility_label_paths(utility_label_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in utility_label_dir.glob("utility_labels_*.csv")
        if not path.name.startswith("utility_labels_reviewed")
    )


def _build_feature_frame(
    records_df: pd.DataFrame,
    *,
    numeric_feature_columns: list[str] | None = None,
    categorical_feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    numeric_feature_columns = numeric_feature_columns or NUMERIC_FEATURE_COLUMNS
    categorical_feature_columns = categorical_feature_columns or CATEGORICAL_FEATURE_COLUMNS
    feature_df = pd.DataFrame(index=records_df.index)
    for column_name in numeric_feature_columns:
        source_column = column_name
        if source_column not in records_df.columns and source_column == "early_loss" and "loss" in records_df.columns:
            source_column = "loss"
        feature_df[column_name] = pd.to_numeric(records_df.get(source_column), errors="coerce")

    for column_name in categorical_feature_columns:
        column_values = records_df.get(column_name)
        if column_values is None:
            column_values = pd.Series(["unknown"] * len(records_df), index=records_df.index)
        feature_df[column_name] = column_values.fillna("unknown").astype(str)
    return feature_df


def _load_model_dependencies() -> dict[str, Any]:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    try:
        import joblib
    except ImportError:
        joblib = None

    return {
        "ColumnTransformer": ColumnTransformer,
        "RandomForestClassifier": RandomForestClassifier,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "joblib": joblib,
    }


def _load_human_review_labels(human_review_dir: Path) -> pd.DataFrame:
    review_paths = _get_human_review_label_paths(human_review_dir)
    if not review_paths:
        return pd.DataFrame()

    review_frames = [_safe_read_csv(path) for path in review_paths]
    review_frames = [frame for frame in review_frames if not frame.empty]
    if not review_frames:
        return pd.DataFrame()

    review_df = pd.concat(review_frames, ignore_index=True)
    if "final_replay_decision" not in review_df.columns:
        return pd.DataFrame()

    final_decisions = review_df["final_replay_decision"].fillna("").astype(str).str.strip().str.lower()
    review_df = review_df.loc[final_decisions.isin({"keep", "reject"})].copy()
    if review_df.empty:
        return review_df

    review_df["target_label"] = final_decisions.loc[review_df.index].map({"keep": 1, "reject": 0})
    review_df["label_source"] = "human_review"
    return review_df[["sample_id", "task_name", "target_label", "label_source"]]


def _load_reviewed_utility_label_training_frame(utility_label_dir: Path) -> pd.DataFrame:
    reviewed_label_paths = _get_reviewed_utility_label_paths(utility_label_dir)
    if not reviewed_label_paths:
        return pd.DataFrame()

    reviewed_frames = [_safe_read_csv(path) for path in reviewed_label_paths]
    reviewed_frames = [frame for frame in reviewed_frames if not frame.empty]
    if not reviewed_frames:
        return pd.DataFrame()

    reviewed_df = pd.concat(reviewed_frames, ignore_index=True)
    if "human_utility_label" not in reviewed_df.columns:
        return pd.DataFrame()

    reviewed_df = reviewed_df.copy()
    reviewed_df["target_label"] = pd.to_numeric(reviewed_df["human_utility_label"], errors="coerce")
    if "label_source" not in reviewed_df.columns:
        reviewed_df["label_source"] = "human_review"
    else:
        reviewed_df["label_source"] = reviewed_df["label_source"].fillna("human_review").astype(str)
    return reviewed_df


def _load_training_frame(run_dir: Path, use_human_review_labels: bool) -> pd.DataFrame:
    utility_label_dir = run_dir / "utility_labels"
    utility_label_paths = _get_proxy_utility_label_paths(utility_label_dir)
    utility_frames = [_safe_read_csv(path) for path in utility_label_paths]
    utility_frames = [frame for frame in utility_frames if not frame.empty]

    if use_human_review_labels:
        reviewed_utility_df = _load_reviewed_utility_label_training_frame(utility_label_dir)
        if not reviewed_utility_df.empty:
            return reviewed_utility_df

    if not utility_frames:
        return pd.DataFrame()

    utility_df = pd.concat(utility_frames, ignore_index=True)
    if use_human_review_labels:
        review_df = _load_human_review_labels(run_dir / "human_review")
        if review_df.empty:
            return pd.DataFrame()
        training_df = utility_df.merge(review_df, on=["sample_id", "task_name"], how="inner")
        return training_df

    utility_df["target_label"] = pd.to_numeric(utility_df.get("utility_label"), errors="coerce")
    utility_df["label_source"] = "proxy_utility_label"
    return utility_df


def _write_warning(output_dir: Path, warning_text: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    warning_path = output_dir / "scorer_training_warning.txt"
    warning_path.write_text(warning_text, encoding="utf-8")
    return warning_path


def _build_reliability_warning(training_df: pd.DataFrame, split_used: str) -> str | None:
    warning_lines: list[str] = []
    label_counts = training_df["target_label"].value_counts()
    if len(training_df) < 100:
        warning_lines.append(
            f"Small labelled dataset: {len(training_df)} rows available for training and evaluation."
        )
    if not label_counts.empty and int(label_counts.min()) < 20:
        warning_lines.append(
            f"Minority class is small: minimum class count is {int(label_counts.min())}."
        )
    if split_used == "train_only":
        warning_lines.append("Evaluation used the training rows directly, so metrics are optimistic.")
    if not warning_lines:
        return None
    return "\n".join(warning_lines)


def _prepare_review_decision_training_frame(
    review_df: pd.DataFrame,
    *,
    label_source: str,
) -> pd.DataFrame:
    if review_df.empty or "final_replay_decision" not in review_df.columns:
        return pd.DataFrame()

    decision_series = review_df["final_replay_decision"].fillna("").astype(str).str.strip().str.lower()
    training_df = review_df.loc[decision_series.isin({"keep", "reject"})].copy()
    if training_df.empty:
        return pd.DataFrame()

    training_df["target_label"] = decision_series.loc[training_df.index].map({"keep": 1, "reject": 0})
    training_df["human_utility_label"] = training_df["target_label"]
    training_df["label_source"] = label_source
    return training_df


def _write_training_summary(
    output_dir: Path,
    *,
    label_source: str,
    model_name: str,
    split_used: str,
    num_rows: int,
    num_train_rows: int,
    num_eval_rows: int,
    num_positive_labels: int,
    num_negative_labels: int,
    metrics: dict[str, float],
    warning_text: str | None,
    model_path: Path | None,
    numeric_feature_columns: list[str],
    categorical_feature_columns: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "scorer_training_summary.md"
    lines = [
        "# Scorer Training Summary",
        "",
        "## Training Setup",
        f"- Label source: `{label_source}`",
        f"- Model: `{model_name}`",
        f"- Split method: `{split_used}`",
        f"- Training rows loaded: {num_rows}",
        f"- Rows used for fitting: {num_train_rows}",
        f"- Rows used for evaluation: {num_eval_rows}",
        "",
        "## Label Distribution",
        f"- Positive labels (`keep`): {num_positive_labels}",
        f"- Negative labels (`reject`): {num_negative_labels}",
        "",
        "## Feature Columns",
        f"- Numeric: {', '.join(f'`{column}`' for column in numeric_feature_columns)}",
        f"- Categorical: {', '.join(f'`{column}`' for column in categorical_feature_columns)}",
        "",
        "## Metrics",
    ]
    for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if metric_name in metrics:
            lines.append(f"- `{metric_name}`: {metrics[metric_name]:.4f}")
    lines.extend(
        [
            "",
            "## Outputs",
            "- Metrics: `scorer_metrics.csv`",
            "- Predictions: `scorer_predictions.csv`",
            "- Feature metadata: `feature_columns.json`",
            f"- Model checkpoint: `{model_path.name}`" if model_path is not None else "- Model checkpoint: not written (`joblib` unavailable)",
        ]
    )
    if warning_text:
        lines.extend(
            [
                "",
                "## Warning",
                warning_text,
            ]
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _train_scorer_from_training_frame(
    training_df: pd.DataFrame,
    *,
    label_source: str,
    output_dir: Path,
    checkpoint_dir: Path,
    model_name: str = "logistic_regression",
    numeric_feature_columns: list[str] | None = None,
    categorical_feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    numeric_feature_columns = numeric_feature_columns or NUMERIC_FEATURE_COLUMNS
    categorical_feature_columns = categorical_feature_columns or CATEGORICAL_FEATURE_COLUMNS
    raw_feature_columns = [*numeric_feature_columns, *categorical_feature_columns]

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if training_df.empty:
        warning_text = f"No training rows were available for label source `{label_source}`."
        warning_path = _write_warning(output_dir, warning_text)
        summary_path = _write_training_summary(
            output_dir,
            label_source=label_source,
            model_name=model_name,
            split_used="none",
            num_rows=0,
            num_train_rows=0,
            num_eval_rows=0,
            num_positive_labels=0,
            num_negative_labels=0,
            metrics={},
            warning_text=warning_text,
            model_path=None,
            numeric_feature_columns=numeric_feature_columns,
            categorical_feature_columns=categorical_feature_columns,
        )
        return {"status": "warning", "warning_path": warning_path, "summary_path": summary_path}

    training_df = training_df.dropna(subset=["target_label"]).copy()
    training_df["target_label"] = training_df["target_label"].astype(int)

    if training_df["target_label"].nunique() < 2 or len(training_df) < 4:
        warning_text = "Not enough label variation or total rows to train a classifier safely."
        warning_path = _write_warning(output_dir, warning_text)
        summary_path = _write_training_summary(
            output_dir,
            label_source=label_source,
            model_name=model_name,
            split_used="none",
            num_rows=len(training_df),
            num_train_rows=0,
            num_eval_rows=0,
            num_positive_labels=int(training_df["target_label"].sum()),
            num_negative_labels=int((1 - training_df["target_label"]).sum()),
            metrics={},
            warning_text=warning_text,
            model_path=None,
            numeric_feature_columns=numeric_feature_columns,
            categorical_feature_columns=categorical_feature_columns,
        )
        return {"status": "warning", "warning_path": warning_path, "summary_path": summary_path}

    deps = _load_model_dependencies()
    pipeline_class = deps["Pipeline"]
    column_transformer_class = deps["ColumnTransformer"]
    simple_imputer_class = deps["SimpleImputer"]
    one_hot_encoder_class = deps["OneHotEncoder"]
    logistic_regression_class = deps["LogisticRegression"]
    random_forest_classifier_class = deps["RandomForestClassifier"]

    classifier: Any
    if model_name == "random_forest":
        classifier = random_forest_classifier_class(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )
    else:
        classifier = logistic_regression_class(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        model_name = "logistic_regression"

    preprocessor = column_transformer_class(
        transformers=[
            (
                "num",
                pipeline_class(
                    steps=[
                        ("imputer", simple_imputer_class(strategy="median")),
                    ]
                ),
                numeric_feature_columns,
            ),
            (
                "cat",
                pipeline_class(
                    steps=[
                        ("imputer", simple_imputer_class(strategy="most_frequent")),
                        ("onehot", one_hot_encoder_class(handle_unknown="ignore")),
                    ]
                ),
                categorical_feature_columns,
            ),
        ]
    )
    pipeline = pipeline_class(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )

    feature_df = _build_feature_frame(
        training_df,
        numeric_feature_columns=numeric_feature_columns,
        categorical_feature_columns=categorical_feature_columns,
    )
    label_series = training_df["target_label"]

    split_used = "train_only"
    evaluation_index = training_df.index
    fit_index = training_df.index
    if len(training_df) >= 10 and training_df["target_label"].value_counts().min() >= 2:
        train_test_split = deps["train_test_split"]
        split_used = "train_test"
        train_indices, test_indices = train_test_split(
            training_df.index.to_list(),
            test_size=0.3,
            random_state=42,
            stratify=label_series,
        )
        fit_index = pd.Index(train_indices)
        pipeline.fit(feature_df.loc[train_indices], label_series.loc[train_indices])
        evaluation_index = pd.Index(test_indices)
    else:
        pipeline.fit(feature_df, label_series)

    probability_matrix = pipeline.predict_proba(feature_df)
    prediction_scores = probability_matrix[:, 1] if probability_matrix.shape[1] > 1 else probability_matrix[:, 0]
    predicted_labels = (prediction_scores >= 0.5).astype(int)

    accuracy_score = deps["accuracy_score"]
    f1_score = deps["f1_score"]
    precision_score = deps["precision_score"]
    recall_score = deps["recall_score"]
    roc_auc_score = deps["roc_auc_score"]

    evaluation_truth = label_series.loc[evaluation_index]
    evaluation_predictions = pd.Series(predicted_labels, index=training_df.index).loc[evaluation_index]
    evaluation_scores = pd.Series(prediction_scores, index=training_df.index).loc[evaluation_index]

    accuracy = float(accuracy_score(evaluation_truth, evaluation_predictions))
    precision = float(precision_score(evaluation_truth, evaluation_predictions, zero_division=0))
    recall = float(recall_score(evaluation_truth, evaluation_predictions, zero_division=0))
    f1 = float(f1_score(evaluation_truth, evaluation_predictions, zero_division=0))

    metrics_rows = [
        {"metric": "label_source", "value": label_source},
        {"metric": "model_name", "value": model_name},
        {"metric": "split_used", "value": split_used},
        {"metric": "num_rows", "value": len(training_df)},
        {"metric": "num_train_rows", "value": len(fit_index)},
        {"metric": "num_eval_rows", "value": len(evaluation_index)},
        {"metric": "num_positive_labels", "value": int(label_series.sum())},
        {"metric": "num_negative_labels", "value": int((1 - label_series).sum())},
        {"metric": "accuracy", "value": accuracy},
        {"metric": "precision", "value": precision},
        {"metric": "recall", "value": recall},
        {"metric": "f1", "value": f1},
    ]
    metrics_summary: dict[str, float] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if evaluation_truth.nunique() > 1:
        roc_auc = float(roc_auc_score(evaluation_truth, evaluation_scores))
        metrics_rows.append(
            {
                "metric": "roc_auc",
                "value": roc_auc,
            }
        )
        metrics_summary["roc_auc"] = roc_auc

    predictions_df = training_df.copy()
    predictions_df["predicted_label"] = predicted_labels
    predictions_df["predicted_utility_score"] = prediction_scores

    metrics_path = output_dir / "scorer_metrics.csv"
    predictions_path = output_dir / "scorer_predictions.csv"
    feature_columns_path = output_dir / "feature_columns.json"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    feature_columns_path.write_text(
        json.dumps(
            {
                "numeric_features": numeric_feature_columns,
                "categorical_features": categorical_feature_columns,
                "raw_feature_columns": raw_feature_columns,
                "model_name": model_name,
                "label_source": label_source,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model_path: Path | None = None
    if deps["joblib"] is not None:
        model_path = checkpoint_dir / "scorer_model.joblib"
        deps["joblib"].dump(pipeline, model_path)

    warning_text = _build_reliability_warning(training_df, split_used)
    warning_path: Path | None = None
    if warning_text:
        warning_path = _write_warning(output_dir, warning_text)

    summary_path = _write_training_summary(
        output_dir,
        label_source=label_source,
        model_name=model_name,
        split_used=split_used,
        num_rows=len(training_df),
        num_train_rows=len(fit_index),
        num_eval_rows=len(evaluation_index),
        num_positive_labels=int(label_series.sum()),
        num_negative_labels=int((1 - label_series).sum()),
        metrics=metrics_summary,
        warning_text=warning_text,
        model_path=model_path,
        numeric_feature_columns=numeric_feature_columns,
        categorical_feature_columns=categorical_feature_columns,
    )

    return {
        "status": "ok",
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "feature_columns_path": feature_columns_path,
        "model_path": model_path,
        "warning_path": warning_path,
        "summary_path": summary_path,
    }


def train_utility_scorer(
    run_dir: Path,
    *,
    use_human_review_labels: bool = False,
    model_name: str = "logistic_regression",
) -> dict[str, Any]:
    """Train a lightweight proxy-utility scorer for one run directory."""
    training_df = _load_training_frame(run_dir, use_human_review_labels=use_human_review_labels)
    label_source = "human_review" if use_human_review_labels else "proxy_utility_label"
    return _train_scorer_from_training_frame(
        training_df,
        label_source=label_source,
        output_dir=run_dir / "scorer_outputs",
        checkpoint_dir=run_dir / "scorer_checkpoints",
        model_name=model_name,
        numeric_feature_columns=NUMERIC_FEATURE_COLUMNS,
        categorical_feature_columns=CATEGORICAL_FEATURE_COLUMNS,
    )


def train_utility_scorer_from_review_csv(
    review_csv_path: Path,
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    model_name: str = "logistic_regression",
) -> dict[str, Any]:
    review_df = _safe_read_csv(review_csv_path)
    training_df = _prepare_review_decision_training_frame(
        review_df,
        label_source="expanded_human_review_final_replay_decision",
    )
    return _train_scorer_from_training_frame(
        training_df,
        label_source="expanded_human_review_final_replay_decision",
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        numeric_feature_columns=NUMERIC_FEATURE_COLUMNS,
        categorical_feature_columns=EXPANDED_CATEGORICAL_FEATURE_COLUMNS,
    )


def predict_utility_scores(
    score_rows: list[dict[str, Any]],
    model_path: Path,
    feature_columns_path: Path,
) -> list[dict[str, Any]]:
    """Predict utility scores for scored samples using a saved scorer pipeline."""
    deps = _load_model_dependencies()
    if deps["joblib"] is None:
        raise ImportError("joblib is required to load a saved utility scorer model.")

    if not model_path.exists():
        raise FileNotFoundError(f"Predicted utility strategy requires a saved model at: {model_path}")
    if not feature_columns_path.exists():
        raise FileNotFoundError(f"Predicted utility strategy requires saved feature metadata at: {feature_columns_path}")

    metadata = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    pipeline = deps["joblib"].load(model_path)
    score_df = pd.DataFrame(score_rows)
    if score_df.empty:
        return []

    if "early_loss" not in score_df.columns and "loss" in score_df.columns:
        score_df["early_loss"] = score_df["loss"]

    numeric_feature_columns = metadata.get("numeric_features", NUMERIC_FEATURE_COLUMNS)
    categorical_feature_columns = metadata.get("categorical_features", CATEGORICAL_FEATURE_COLUMNS)
    feature_df = _build_feature_frame(
        score_df,
        numeric_feature_columns=list(numeric_feature_columns),
        categorical_feature_columns=list(categorical_feature_columns),
    )
    probability_matrix = pipeline.predict_proba(feature_df)
    prediction_scores = probability_matrix[:, 1] if probability_matrix.shape[1] > 1 else probability_matrix[:, 0]

    prediction_rows: list[dict[str, Any]] = []
    for row, prediction_score in zip(score_rows, prediction_scores):
        prediction_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "task_name": row.get("task_name"),
                "predicted_utility_score": float(prediction_score),
            }
        )
    return prediction_rows
