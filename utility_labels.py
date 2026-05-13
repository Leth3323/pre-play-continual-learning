from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


UTILITY_LABEL_COLUMNS = [
    "sample_id",
    "task_name",
    "split",
    "source_dataset",
    "input_text",
    "target_text",
    "token_count",
    "early_loss",
    "surprise_score",
    "replay_selection_score",
    "was_selected_for_replay",
    "utility_label",
    "utility_score",
    "label_method",
]

HUMAN_REVIEW_COLUMNS = [
    "sample_id",
    "task_name",
    "split",
    "source_dataset",
    "input_text",
    "target_text",
    "token_count",
    "early_loss",
    "surprise_score",
    "replay_selection_score",
    "utility_score",
    "utility_label",
    "was_selected_for_replay",
    "auto_replay_tag",
    "auto_tag_reason",
    "human_review_status",
    "human_review_label",
    "human_review_notes",
    "final_replay_decision",
]

PROXY_LABEL_METHOD = "proxy_replay_selection_v1"
HUMAN_REVIEW_CANDIDATE_FILE_NAME = "replay_review_candidates.csv"
HUMAN_REVIEW_REVIEWED_FILE_NAME = "replay_review_candidates_reviewed.csv"
AUTO_REPLAY_TAGS = [
    "high_value_candidate",
    "selected_for_replay",
    "needs_human_review",
    "low_priority",
]


def _safe_read_csv(input_path: Path) -> pd.DataFrame:
    if not input_path.exists() or input_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(input_path)
    except EmptyDataError:
        return pd.DataFrame()


def _normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _first_non_null(series: pd.Series) -> Any:
    for value in series:
        if pd.notna(value):
            return value
    return None


def _compute_utility_score(row: pd.Series) -> Any:
    for column_name in ("replay_selection_score", "surprise_score", "early_loss"):
        value = row.get(column_name)
        if pd.notna(value):
            return value
    return None


def generate_proxy_utility_labels(
    sample_signals_path: Path,
    output_dir: Path,
) -> Path:
    """Aggregate sample-level signals into one proxy utility label per sample."""
    output_dir.mkdir(parents=True, exist_ok=True)
    task_name = sample_signals_path.stem.replace("sample_signals_", "", 1)
    output_path = output_dir / f"utility_labels_{task_name}.csv"

    signals_df = _safe_read_csv(sample_signals_path)
    if signals_df.empty:
        pd.DataFrame(columns=UTILITY_LABEL_COLUMNS).to_csv(output_path, index=False)
        return output_path

    sort_columns = [column for column in ("epoch", "step") if column in signals_df.columns]
    if sort_columns:
        signals_df = signals_df.sort_values(sort_columns, kind="stable")

    grouped = signals_df.groupby("sample_id", dropna=False, sort=False)
    utility_df = grouped.agg(
        {
            "task_name": "first",
            "split": "first",
            "source_dataset": "first",
            "input_text": "first",
            "target_text": "first",
            "token_count": _first_non_null,
            "loss": _first_non_null,
            "surprise_score": _first_non_null,
            "replay_selection_score": _first_non_null,
            "was_added_to_replay_buffer": "max",
        }
    ).reset_index()

    utility_df = utility_df.rename(
        columns={
            "loss": "early_loss",
            "was_added_to_replay_buffer": "was_selected_for_replay",
        }
    )
    utility_df["was_selected_for_replay"] = (
        pd.to_numeric(utility_df["was_selected_for_replay"], errors="coerce").fillna(0).astype(int)
    )
    utility_df["utility_label"] = utility_df["was_selected_for_replay"]
    utility_df["utility_score"] = utility_df.apply(_compute_utility_score, axis=1)
    utility_df["label_method"] = PROXY_LABEL_METHOD

    utility_df = utility_df.reindex(columns=UTILITY_LABEL_COLUMNS)
    utility_df.to_csv(output_path, index=False)
    return output_path


def _mark_top_thirty_percent(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(False, index=df.index)
    numeric_series = pd.to_numeric(df[column_name], errors="coerce")
    valid_series = numeric_series.dropna()
    if valid_series.empty:
        return pd.Series(False, index=df.index)
    threshold = valid_series.quantile(0.7)
    return numeric_series.ge(threshold).fillna(False)


def _build_auto_tag_reason(row: pd.Series) -> str:
    reason_parts: list[str] = []
    if row.get("utility_label") == 1:
        reason_parts.append("proxy utility label indicates the sample was selected for replay")
    else:
        reason_parts.append("proxy utility label indicates the sample was not selected for replay")

    if row.get("utility_score_is_high", False):
        reason_parts.append("utility_score is in the top 30% within the task")
    elif pd.notna(row.get("utility_score")):
        reason_parts.append("utility_score is available but not in the top 30% within the task")
    else:
        reason_parts.append("utility_score is unavailable")

    loss_reason_parts: list[str] = []
    if row.get("early_loss_is_high", False):
        loss_reason_parts.append("early_loss is in the top 30% within the task")
    elif pd.notna(row.get("early_loss")):
        loss_reason_parts.append("early_loss is available")

    if row.get("surprise_score_is_high", False):
        loss_reason_parts.append("surprise_score is in the top 30% within the task")
    elif pd.notna(row.get("surprise_score")):
        loss_reason_parts.append("surprise_score is available")

    if loss_reason_parts:
        reason_parts.append("; ".join(loss_reason_parts))
    else:
        reason_parts.append("high-loss and surprise fields are unavailable")

    return ". ".join(reason_parts) + "."


def _assign_auto_replay_tags(utility_df: pd.DataFrame) -> pd.DataFrame:
    if utility_df.empty:
        return utility_df

    tagged_groups: list[pd.DataFrame] = []
    for _, task_df in utility_df.groupby("task_name", dropna=False, sort=False):
        task_df = task_df.copy()
        task_df["utility_score_is_high"] = _mark_top_thirty_percent(task_df, "utility_score")
        task_df["early_loss_is_high"] = _mark_top_thirty_percent(task_df, "early_loss")
        task_df["surprise_score_is_high"] = _mark_top_thirty_percent(task_df, "surprise_score")

        task_df["auto_replay_tag"] = "low_priority"
        high_value_mask = (task_df["utility_label"] == 1) & task_df["utility_score_is_high"]
        selected_mask = (task_df["utility_label"] == 1) & ~task_df["utility_score_is_high"]
        review_mask = (task_df["utility_label"] == 0) & (
            task_df["early_loss_is_high"] | task_df["surprise_score_is_high"]
        )

        task_df.loc[high_value_mask, "auto_replay_tag"] = "high_value_candidate"
        task_df.loc[selected_mask, "auto_replay_tag"] = "selected_for_replay"
        task_df.loc[review_mask, "auto_replay_tag"] = "needs_human_review"
        task_df["auto_tag_reason"] = task_df.apply(_build_auto_tag_reason, axis=1)
        tagged_groups.append(task_df)

    return pd.concat(tagged_groups, ignore_index=True)


def _preferred_human_review_output_path(human_review_dir: Path) -> Path:
    return human_review_dir / HUMAN_REVIEW_CANDIDATE_FILE_NAME


def _preferred_reviewed_human_review_output_path(human_review_dir: Path) -> Path:
    return human_review_dir / HUMAN_REVIEW_REVIEWED_FILE_NAME


def _has_non_pending_reviews(review_df: pd.DataFrame) -> bool:
    if "human_review_status" not in review_df.columns:
        return False
    statuses = review_df["human_review_status"].fillna("").astype(str).str.strip().str.lower()
    return any(status and status != "pending" for status in statuses)


def _review_copy_can_be_overwritten(review_df: pd.DataFrame) -> bool:
    if "human_review_status" not in review_df.columns:
        return False
    statuses = review_df["human_review_status"].fillna("").astype(str).str.strip().str.lower()
    return bool(((statuses == "") | (statuses == "pending")).all())


def _timestamped_human_review_path(human_review_dir: Path, file_stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return human_review_dir / f"{file_stem}_{timestamp}.csv"


def _resolve_human_review_output_path(human_review_dir: Path) -> Path:
    output_path = _preferred_human_review_output_path(human_review_dir)
    if not output_path.exists():
        return output_path

    existing_df = _safe_read_csv(output_path)
    if not _has_non_pending_reviews(existing_df):
        return output_path

    return _timestamped_human_review_path(human_review_dir, "replay_review_candidates")


def create_review_copy(candidate_csv_path: Path) -> Path:
    """Create or preserve the manual-review copy that mirrors the generated candidate CSV."""
    reviewed_output_path = _preferred_reviewed_human_review_output_path(candidate_csv_path.parent)
    if not reviewed_output_path.exists():
        shutil.copyfile(candidate_csv_path, reviewed_output_path)
        return reviewed_output_path

    existing_reviewed_df = _safe_read_csv(reviewed_output_path)
    if _review_copy_can_be_overwritten(existing_reviewed_df):
        shutil.copyfile(candidate_csv_path, reviewed_output_path)
        return reviewed_output_path

    timestamped_reviewed_path = _timestamped_human_review_path(
        candidate_csv_path.parent,
        "replay_review_candidates_reviewed",
    )
    shutil.copyfile(candidate_csv_path, timestamped_reviewed_path)
    print(
        "Warning: preserved manually reviewed file "
        f"`{reviewed_output_path.name}` because it contains non-pending human_review_status values. "
        f"Wrote the new auto-generated review copy to `{timestamped_reviewed_path.name}`."
    )
    return timestamped_reviewed_path


def export_human_review_candidates(
    utility_label_paths: list[Path],
    human_review_dir: Path,
) -> tuple[Path, Path]:
    """Aggregate utility labels into one human-review export with auto tags."""
    human_review_dir.mkdir(parents=True, exist_ok=True)
    review_output_path = _resolve_human_review_output_path(human_review_dir)
    readme_output_path = human_review_dir / "replay_review_readme.md"

    utility_frames = [_safe_read_csv(path) for path in utility_label_paths]
    utility_frames = [frame for frame in utility_frames if not frame.empty]
    if utility_frames:
        utility_df = pd.concat(utility_frames, ignore_index=True)
    else:
        utility_df = pd.DataFrame(columns=UTILITY_LABEL_COLUMNS)

    utility_df = _assign_auto_replay_tags(utility_df)
    utility_df["human_review_status"] = "pending"
    utility_df["human_review_label"] = ""
    utility_df["human_review_notes"] = ""
    utility_df["final_replay_decision"] = ""
    utility_df = utility_df.rename(columns={"utility_label": "utility_label"})
    utility_df = utility_df.reindex(columns=HUMAN_REVIEW_COLUMNS)
    utility_df.to_csv(review_output_path, index=False)
    create_review_copy(review_output_path)

    readme_output_path.write_text(
        "\n".join(
            [
                "# Replay Review Guide",
                "",
                "This export is a human-review queue built from proxy utility labels (`proxy_replay_selection_v1`).",
                "It is not an exact causal estimate of future replay utility.",
                "",
                "## Automatic Tags",
                "- `high_value_candidate`: selected for replay by the proxy label and also high on utility_score within the same task.",
                "- `selected_for_replay`: selected for replay by the proxy label, but not in the top 30 percent on utility_score within the same task.",
                "- `needs_human_review`: not selected for replay by the proxy label, but high on early_loss or surprise_score within the same task.",
                "- `low_priority`: none of the higher-priority conditions were triggered.",
                "",
                "## Manual Review",
                f"- `{HUMAN_REVIEW_CANDIDATE_FILE_NAME}` is the original auto-generated export.",
                f"- `{HUMAN_REVIEW_REVIEWED_FILE_NAME}` is the file intended for manual editing.",
                f"- Edit only `{HUMAN_REVIEW_REVIEWED_FILE_NAME}`.",
                f"- Keep `{HUMAN_REVIEW_CANDIDATE_FILE_NAME}` unchanged for reproducibility.",
                "- Read the input and target text together.",
                "- Check whether the sample looks useful for future replay, low quality, redundant, or narrowly task-specific.",
                (
                    "- If an existing reviewed copy already contains non-pending review statuses, "
                    "the exporter preserves it and writes a timestamped reviewed copy instead."
                ),
                (
                    f"- Update `human_review_status`, `human_review_label`, `human_review_notes`, and "
                    f"`final_replay_decision` directly in `{HUMAN_REVIEW_REVIEWED_FILE_NAME}`."
                ),
                "",
                "## Recommended `human_review_label` Values",
                "- `useful_for_replay`",
                "- `low_quality`",
                "- `ambiguous`",
                "- `redundant`",
                "- `task_specific`",
                "- `general_knowledge`",
                "- `unsure`",
                "",
                "## Recommended `final_replay_decision` Values",
                "- `keep`",
                "- `reject`",
                "- `unsure`",
            ]
        ),
        encoding="utf-8",
    )
    return review_output_path, readme_output_path
