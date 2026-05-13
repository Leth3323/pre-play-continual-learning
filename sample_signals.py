from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SAMPLE_SIGNAL_COLUMNS = [
    "sample_id",
    "task_name",
    "split",
    "source_dataset",
    "input_text",
    "target_text",
    "token_count",
    "epoch",
    "step",
    "loss",
    "surprise_score",
    "replay_selection_strategy",
    "was_added_to_replay_buffer",
    "replay_selection_score",
]


def enrich_and_filter_sample_signals(
    sample_signal_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    current_task_name: str,
    replay_selection_strategy: str,
    selected_sample_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Keep only the current task rows and attach replay-selection metadata."""
    selected_ids = selected_sample_ids or set()
    score_lookup = {str(row.get("sample_id")): row for row in score_rows}

    enriched_rows: list[dict[str, Any]] = []
    for row in sample_signal_rows:
        if str(row.get("task_name")) != current_task_name:
            continue

        sample_id = str(row.get("sample_id", ""))
        score_row = score_lookup.get(sample_id, {})
        enriched_row = {column: row.get(column) for column in SAMPLE_SIGNAL_COLUMNS}
        enriched_row["token_count"] = (
            row.get("token_count")
            if row.get("token_count") is not None
            else score_row.get("token_count")
        )
        enriched_row["surprise_score"] = (
            score_row.get("surprise_score")
            if score_row.get("surprise_score") is not None
            else row.get("surprise_score")
        )
        enriched_row["replay_selection_strategy"] = replay_selection_strategy
        enriched_row["was_added_to_replay_buffer"] = int(sample_id in selected_ids)
        enriched_row["replay_selection_score"] = score_row.get("replay_selection_score")
        enriched_rows.append(enriched_row)

    return enriched_rows


def write_sample_signals(
    output_dir: Path,
    task_name: str,
    sample_signal_rows: list[dict[str, Any]],
) -> Path:
    """Write one sample-signals CSV for a task."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sample_signals_{task_name}.csv"
    pd.DataFrame(sample_signal_rows, columns=SAMPLE_SIGNAL_COLUMNS).to_csv(output_path, index=False)
    return output_path
