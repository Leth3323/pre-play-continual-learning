from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd

from data_loader import InstructionSample
from utility_scorer import predict_utility_scores


REPLAY_SELECTION_STRATEGIES = (
    "random",
    "loss",
    "surprise",
    "predicted_utility",
)


def _safe_selection_score(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("-inf")
    return float(value)


def _with_predicted_utility_scores(
    score_rows: list[dict[str, Any]],
    run_dir: Path,
    *,
    utility_scorer_path: Path | None = None,
    utility_feature_columns_path: Path | None = None,
) -> list[dict[str, Any]]:
    model_path, feature_columns_path = resolve_predicted_utility_artifact_paths(
        run_dir,
        utility_scorer_path=utility_scorer_path,
        utility_feature_columns_path=utility_feature_columns_path,
    )
    prediction_rows = predict_utility_scores(score_rows, model_path, feature_columns_path)
    prediction_lookup = {
        str(row.get("sample_id")): row.get("predicted_utility_score")
        for row in prediction_rows
    }
    enriched_rows: list[dict[str, Any]] = []
    for row in score_rows:
        enriched_row = dict(row)
        sample_id = str(row.get("sample_id"))
        if sample_id not in prediction_lookup:
            raise ValueError(
                f"Predicted utility strategy could not find a predicted score for sample_id={sample_id!r}."
            )
        enriched_row["replay_selection_score"] = float(prediction_lookup[sample_id])
        enriched_rows.append(enriched_row)
    return enriched_rows


def resolve_predicted_utility_artifact_paths(
    run_dir: Path,
    *,
    utility_scorer_path: Path | None = None,
    utility_feature_columns_path: Path | None = None,
) -> tuple[Path, Path]:
    if (utility_scorer_path is None) != (utility_feature_columns_path is None):
        raise ValueError(
            "Predicted utility strategy requires both utility_scorer_path and "
            "utility_feature_columns_path when either is provided."
        )

    if utility_scorer_path is not None and utility_feature_columns_path is not None:
        checked_text = (
            f"model={utility_scorer_path}, "
            f"features={utility_feature_columns_path}"
        )
        if not utility_scorer_path.exists() or not utility_feature_columns_path.exists():
            raise FileNotFoundError(
                "Predicted utility strategy could not find the required scorer artifacts "
                "from the explicitly provided paths.\n"
                "Checked:\n"
                f"{checked_text}"
            )
        return utility_scorer_path, utility_feature_columns_path

    experiment_dir = run_dir.parent
    candidate_pairs = [
        (
            experiment_dir / "scorer_checkpoints" / "scorer_model.joblib",
            experiment_dir / "scorer_outputs" / "feature_columns.json",
        ),
        (
            experiment_dir.parent / "exp006" / "scorer_checkpoints" / "scorer_model.joblib",
            experiment_dir.parent / "exp006" / "scorer_outputs" / "feature_columns.json",
        ),
    ]

    checked_paths: list[str] = []
    for model_path, feature_columns_path in candidate_pairs:
        checked_paths.append(f"model={model_path}, features={feature_columns_path}")
        if model_path.exists() and feature_columns_path.exists():
            return model_path, feature_columns_path

    checked_text = "\n".join(checked_paths)
    raise FileNotFoundError(
        "Predicted utility strategy could not find the required scorer artifacts.\n"
        "Checked:\n"
        f"{checked_text}"
    )


def attach_replay_selection_scores(
    score_rows: list[dict[str, Any]],
    strategy: str,
    *,
    task_name: str,
    seed: int,
    run_dir: Path | None = None,
    utility_scorer_path: Path | None = None,
    utility_feature_columns_path: Path | None = None,
    allow_missing_predicted_utility: bool = False,
) -> list[dict[str, Any]]:
    """Attach a replay-selection score to each scored sample."""
    if strategy not in REPLAY_SELECTION_STRATEGIES:
        raise ValueError(
            f"Unsupported replay selection strategy: {strategy}. "
            f"Expected one of {', '.join(REPLAY_SELECTION_STRATEGIES)}."
        )

    if strategy == "predicted_utility":
        if run_dir is None:
            raise ValueError("Predicted utility strategy requires a run directory for scorer artifacts.")
        try:
            return _with_predicted_utility_scores(
                score_rows,
                run_dir,
                utility_scorer_path=utility_scorer_path,
                utility_feature_columns_path=utility_feature_columns_path,
            )
        except (FileNotFoundError, ImportError):
            if allow_missing_predicted_utility:
                enriched_rows = []
                for row in score_rows:
                    enriched_row = dict(row)
                    enriched_row["replay_selection_score"] = None
                    enriched_rows.append(enriched_row)
                return enriched_rows
            raise

    enriched_rows: list[dict[str, Any]] = []
    rng = random.Random(f"{seed}:{task_name}:{strategy}")
    for row in score_rows:
        enriched_row = dict(row)
        if strategy == "random":
            enriched_row["replay_selection_score"] = rng.random()
        elif strategy == "loss":
            enriched_row["replay_selection_score"] = row.get("loss")
        else:
            enriched_row["replay_selection_score"] = row.get("surprise_score", row.get("loss"))
        enriched_rows.append(enriched_row)
    return enriched_rows


def select_samples_for_replay(
    samples: list[InstructionSample],
    scored_rows: list[dict[str, Any]],
    top_k: int,
) -> list[InstructionSample]:
    """Select the highest-scoring replay candidates under the configured strategy."""
    if top_k <= 0 or not samples or not scored_rows:
        return []

    score_lookup = {
        str(row.get("sample_id")): _safe_selection_score(row.get("replay_selection_score"))
        for row in scored_rows
    }
    ranked_samples = sorted(
        samples,
        key=lambda sample: (
            score_lookup.get(sample.sample_id, float("-inf")),
            sample.sample_id,
        ),
        reverse=True,
    )
    return ranked_samples[:top_k]


def write_replay_selection_comparison(
    output_dir: Path,
    task_name: str,
    scored_rows: list[dict[str, Any]],
    selected_sample_ids: set[str],
    strategy: str,
) -> Path:
    """Write a lightweight per-task view of selection scores and replay decisions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task_name}_selection_comparison.csv"
    comparison_rows: list[dict[str, Any]] = []
    for row in scored_rows:
        enriched_row = dict(row)
        enriched_row["replay_selection_strategy"] = strategy
        enriched_row["was_selected_for_replay"] = int(str(row.get("sample_id")) in selected_sample_ids)
        comparison_rows.append(enriched_row)
    pd.DataFrame(comparison_rows).to_csv(output_path, index=False)
    return output_path
