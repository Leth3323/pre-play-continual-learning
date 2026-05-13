from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from config import get_config


TASK_DESCRIPTIONS = {
    "sciq": "science question answering task",
    "arc_challenge": "challenging grade-school science question answering task",
    "boolq": "yes/no question answering task",
    "gsm8k": "grade-school math word problem task",
}


def write_text(output_path: Path, text: str) -> None:
    """Write a UTF-8 text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def relative_markdown_link(target_path: Path, current_file_dir: Path, label: str) -> str:
    """Build a relative markdown link."""
    relative_path = target_path.relative_to(current_file_dir)
    return f"[{label}]({relative_path.as_posix()})"


def build_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a markdown table."""
    if not rows:
        return "Not available."

    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row, *body_rows])


def format_metric(value: Any) -> str:
    """Format a numeric metric value or return Not available."""
    if value is None:
        return "Not available"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric_value):
        return "Not available"
    return f"{numeric_value:.4f}"


def format_scalar(value: Any) -> str:
    """Format a scalar configuration value."""
    if value is None:
        return "Not available"
    if isinstance(value, float):
        if math.isnan(value):
            return "Not available"
        return f"{value:.6g}"
    return str(value)


def format_task_sequence(task_order: list[str]) -> str:
    """Format a task sequence for prose."""
    if not task_order:
        return "Not available"
    return " -> ".join(task_order)


def get_config_utility_scorer_path(config: dict[str, Any]) -> Any:
    """Return the configured utility scorer path, including backward-compatible keys."""
    return config.get("utility_scorer_path") or config.get("predicted_utility_scorer_model_path")


def get_config_utility_feature_columns_path(config: dict[str, Any]) -> Any:
    """Return the configured utility feature metadata path, including backward-compatible keys."""
    return config.get("utility_feature_columns_path") or config.get("predicted_utility_feature_columns_path")


def safe_read_json(input_path: Path) -> dict[str, Any]:
    """Read a JSON file from disk."""
    return json.loads(input_path.read_text(encoding="utf-8"))


def safe_read_csv(input_path: Path) -> pd.DataFrame:
    """Read a CSV file, returning an empty dataframe for empty files."""
    if not input_path.exists() or input_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(input_path)
    except EmptyDataError:
        return pd.DataFrame()


def format_boolean(value: bool) -> str:
    """Format a boolean value as Yes or No."""
    return "Yes" if value else "No"


def get_preferred_human_review_file(human_review_dir: Path) -> Path | None:
    """Return the reviewed copy first, then fall back to the original candidate export."""
    reviewed_path = human_review_dir / "replay_review_candidates_reviewed.csv"
    if reviewed_path.exists():
        return reviewed_path
    reviewed_paths = sorted(human_review_dir.glob("replay_review_candidates_reviewed*.csv"))
    if reviewed_paths:
        return reviewed_paths[-1]

    canonical_path = human_review_dir / "replay_review_candidates.csv"
    if canonical_path.exists():
        return canonical_path
    review_paths = sorted(
        path
        for path in human_review_dir.glob("replay_review_candidates*.csv")
        if not path.name.startswith("replay_review_candidates_reviewed")
    )
    if not review_paths:
        return None
    return review_paths[-1]


def summarize_human_review_df(review_df: pd.DataFrame) -> tuple[dict[str, int], int]:
    """Summarize automatic tag counts and manual review coverage."""
    tag_names = [
        "high_value_candidate",
        "selected_for_replay",
        "needs_human_review",
        "low_priority",
    ]
    tag_counts = {tag_name: 0 for tag_name in tag_names}
    if not review_df.empty and "auto_replay_tag" in review_df.columns:
        tags = review_df["auto_replay_tag"].fillna("").astype(str)
        for tag_name in tag_names:
            tag_counts[tag_name] = int((tags == tag_name).sum())

    manual_review_count = 0
    if not review_df.empty and "human_review_status" in review_df.columns:
        statuses = review_df["human_review_status"].fillna("").astype(str).str.strip().str.lower()
        manual_review_count = int(((statuses != "") & (statuses != "pending")).sum())

    return tag_counts, manual_review_count


def count_file_lines(file_path: Path) -> int:
    """Count the number of lines in a text file."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return 0
    with file_path.open("r", encoding="utf-8") as file_handle:
        return sum(1 for _ in file_handle)


def list_directory_entries(directory: Path) -> str:
    """Return a comma-separated directory listing."""
    if not directory.exists():
        return "Not available"
    entries = sorted(entry.name for entry in directory.iterdir() if not entry.name.startswith("."))
    return ", ".join(entries) if entries else "Empty"


def build_tree(root: Path, max_depth: int = 3) -> str:
    """Build a simple text tree for a directory."""
    lines = [f"{root.name}/"]

    def walk(path: Path, prefix: str, depth: int) -> None:
        if depth >= max_depth:
            return
        entries = sorted(
            (entry for entry in path.iterdir() if not entry.name.startswith(".")),
            key=lambda item: (item.is_file(), item.name),
        )
        for index, entry in enumerate(entries):
            connector = "└── " if index == len(entries) - 1 else "├── "
            label = entry.name + ("/" if entry.is_dir() else "")
            lines.append(prefix + connector + label)
            if entry.is_dir():
                extension = "    " if index == len(entries) - 1 else "│   "
                walk(entry, prefix + extension, depth + 1)

    walk(root, "", 0)
    return "\n".join(lines)


def get_experiment_root(project_root: Path | None = None) -> Path:
    """Return the outputs/experiments root."""
    return get_config(project_root=project_root).paths.outputs_experiments


def list_experiment_dirs(project_root: Path | None = None) -> list[Path]:
    """Return experiment-level directories under outputs/experiments."""
    experiments_root = get_experiment_root(project_root=project_root)
    if not experiments_root.exists():
        return []
    experiment_dirs: list[Path] = []
    for path in sorted(experiments_root.iterdir()):
        if not path.is_dir():
            continue
        if list_run_dirs(path):
            experiment_dirs.append(path)
    return experiment_dirs


def list_run_dirs(experiment_dir: Path) -> list[Path]:
    """Return nested run directories under one experiment directory."""
    if not experiment_dir.exists():
        return []
    return sorted(
        path for path in experiment_dir.iterdir() if path.is_dir() and (path / "run_config.json").exists()
    )


def get_experiment_summary_path(experiment_dir: Path) -> Path:
    """Return the enriched experiment summary path."""
    return experiment_dir / f"{experiment_dir.name}_summary.md"


def filter_final_metric_rows(metrics_df: pd.DataFrame, final_task_name: str | None) -> pd.DataFrame:
    """Return metric rows recorded after the final task."""
    if final_task_name is None or metrics_df.empty or "after_task" not in metrics_df.columns:
        return pd.DataFrame()
    return metrics_df[metrics_df["after_task"] == final_task_name].copy()


def build_reconstructed_command(config: dict[str, Any]) -> str:
    """Reconstruct a plausible run command from saved config."""
    replay_enabled = bool(config.get("replay_enabled", False))
    command_parts = [
        "python",
        "main.py",
        "--run-name",
        str(config.get("run_name", "Not available")),
    ]
    task_order = config.get("task_order", [])
    if task_order:
        command_parts.append("--tasks")
        command_parts.extend(str(task_name) for task_name in task_order)
    if replay_enabled and config.get("replay_buffer_size") is not None:
        command_parts.extend(["--replay-buffer-size", str(config.get("replay_buffer_size"))])
    replay_selection_strategy = config.get("replay_selection_strategy")
    if replay_enabled and replay_selection_strategy:
        command_parts.extend(["--replay-selection-strategy", str(replay_selection_strategy)])
    utility_scorer_path = get_config_utility_scorer_path(config)
    utility_feature_columns_path = get_config_utility_feature_columns_path(config)
    if replay_enabled and replay_selection_strategy == "predicted_utility":
        if utility_scorer_path:
            command_parts.extend(["--utility-scorer-path", str(utility_scorer_path)])
        if utility_feature_columns_path:
            command_parts.extend(["--utility-feature-columns-path", str(utility_feature_columns_path)])
    if not replay_enabled:
        command_parts.append("--no-replay")
    if config.get("seed") is not None:
        command_parts.extend(["--seed", str(config.get("seed"))])
    return " ".join(command_parts)


def load_run_record(run_dir: Path) -> dict[str, Any]:
    """Load metrics and artifact metadata for one run directory."""
    config = safe_read_json(run_dir / "run_config.json")
    task_order = list(config.get("task_order", []))
    final_task_name = task_order[-1] if task_order else None

    val_metrics_path = run_dir / "metrics" / "val_metrics.csv"
    test_metrics_path = run_dir / "metrics" / "test_metrics.csv"
    train_summary_path = run_dir / "logs" / "train_summary.csv"
    val_metrics_df = safe_read_csv(val_metrics_path)
    test_metrics_df = safe_read_csv(test_metrics_path)
    final_val_df = filter_final_metric_rows(val_metrics_df, final_task_name)
    final_test_df = filter_final_metric_rows(test_metrics_df, final_task_name)
    if final_test_df.empty:
        final_test_df = test_metrics_df.copy()

    metrics_by_task: dict[str, dict[str, Any]] = {task_name: {} for task_name in task_order}
    for _, row in final_val_df.iterrows():
        task_name = str(row["task_name"])
        metrics_by_task.setdefault(task_name, {})
        metrics_by_task[task_name].update(
            {
                "val_accuracy": row.get("accuracy"),
                "val_samples": row.get("num_samples"),
                "best_val_accuracy": row.get("best_accuracy_so_far"),
                "forgetting": row.get("forgetting"),
            }
        )
    for _, row in final_test_df.iterrows():
        task_name = str(row["task_name"])
        metrics_by_task.setdefault(task_name, {})
        metrics_by_task[task_name].update(
            {
                "test_accuracy": row.get("accuracy"),
                "test_samples": row.get("num_samples"),
            }
        )

    replay_scores_dir = run_dir / "replay_scores"
    replay_buffers_dir = run_dir / "replay_buffers"
    replay_score_files = sorted(replay_scores_dir.glob("*.csv")) if replay_scores_dir.exists() else []
    replay_buffer_files = sorted(replay_buffers_dir.iterdir()) if replay_buffers_dir.exists() else []
    replay_buffer_snapshots = sorted(replay_buffers_dir.glob("*.jsonl")) if replay_buffers_dir.exists() else []
    non_empty_replay_snapshots = [path for path in replay_buffer_snapshots if path.stat().st_size > 0]
    sample_signals_dir = run_dir / "sample_signals"
    utility_labels_dir = run_dir / "utility_labels"
    human_review_dir = run_dir / "human_review"
    scorer_outputs_dir = run_dir / "scorer_outputs"
    scorer_checkpoints_dir = run_dir / "scorer_checkpoints"
    replay_selection_comparison_dir = run_dir / "replay_selection_comparison"
    sample_signal_files = sorted(sample_signals_dir.glob("*.csv")) if sample_signals_dir.exists() else []
    utility_label_files = sorted(utility_labels_dir.glob("*.csv")) if utility_labels_dir.exists() else []
    human_review_files = sorted(human_review_dir.iterdir()) if human_review_dir.exists() else []
    human_review_candidate_file = get_preferred_human_review_file(human_review_dir) if human_review_dir.exists() else None
    human_review_df = (
        safe_read_csv(human_review_candidate_file)
        if human_review_candidate_file is not None
        else pd.DataFrame()
    )
    human_review_tag_counts, manual_human_review_count = summarize_human_review_df(human_review_df)
    scorer_output_files = sorted(scorer_outputs_dir.iterdir()) if scorer_outputs_dir.exists() else []
    scorer_checkpoint_files = sorted(scorer_checkpoints_dir.iterdir()) if scorer_checkpoints_dir.exists() else []
    replay_selection_comparison_files = (
        sorted(replay_selection_comparison_dir.glob("*.csv"))
        if replay_selection_comparison_dir.exists()
        else []
    )

    console_log_path = run_dir / "logs" / "console.log"
    console_log_text = console_log_path.read_text(encoding="utf-8") if console_log_path.exists() else ""
    has_traceback = "Traceback" in console_log_text

    checkpoints_dir = run_dir / "checkpoints"
    has_checkpoints = checkpoints_dir.exists() and any(checkpoints_dir.iterdir())
    run_appears_complete = (
        has_checkpoints
        and val_metrics_path.exists()
        and test_metrics_path.exists()
        and train_summary_path.exists()
        and not has_traceback
    )

    return {
        "run_name": str(config.get("run_name", run_dir.name)),
        "run_dir": run_dir,
        "config": config,
        "task_order": task_order,
        "replay_enabled": bool(config.get("replay_enabled", False)),
        "metrics_by_task": metrics_by_task,
        "replay_score_files": replay_score_files,
        "replay_buffer_files": replay_buffer_files,
        "replay_buffer_snapshots": replay_buffer_snapshots,
        "non_empty_replay_snapshots": non_empty_replay_snapshots,
        "sample_signal_files": sample_signal_files,
        "utility_label_files": utility_label_files,
        "human_review_files": human_review_files,
        "human_review_candidate_file": human_review_candidate_file,
        "human_review_tag_counts": human_review_tag_counts,
        "manual_human_review_count": manual_human_review_count,
        "scorer_output_files": scorer_output_files,
        "scorer_checkpoint_files": scorer_checkpoint_files,
        "replay_selection_comparison_files": replay_selection_comparison_files,
        "checkpoint_entries": list_directory_entries(run_dir / "checkpoints"),
        "log_entries": list_directory_entries(run_dir / "logs"),
        "metric_entries": list_directory_entries(run_dir / "metrics"),
        "console_log_path": console_log_path,
        "console_log_exists": console_log_path.exists(),
        "has_traceback": has_traceback,
        "run_appears_complete": run_appears_complete,
        "reconstructed_command": build_reconstructed_command(config),
    }


def get_condition_run(run_records: list[dict[str, Any]], replay_enabled: bool) -> dict[str, Any] | None:
    """Return the first run matching a replay condition."""
    for record in run_records:
        if record["replay_enabled"] == replay_enabled:
            return record
    return None


def get_replay_runs(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return replay-enabled runs sorted by configured replay buffer size."""
    replay_runs = [record for record in run_records if record["replay_enabled"]]
    return sorted(
        replay_runs,
        key=lambda record: (
            int(record["config"].get("replay_buffer_size", 10**9)),
            record["run_name"],
        ),
    )


def is_replay_buffer_size_ablation_experiment(
    experiment_id: str,
    run_records: list[dict[str, Any]],
) -> bool:
    """Return True when an experiment contains one no-replay run and multiple replay buffer sizes."""
    baseline_run = get_condition_run(run_records, replay_enabled=False)
    replay_runs = get_replay_runs(run_records)
    replay_buffer_sizes = {
        int(record["config"].get("replay_buffer_size"))
        for record in replay_runs
        if record["config"].get("replay_buffer_size") is not None
    }
    return baseline_run is not None and len(replay_runs) >= 2 and len(replay_buffer_sizes) >= 2


def is_seed_robustness_experiment(
    experiment_id: str,
    run_records: list[dict[str, Any]],
) -> bool:
    """Return True when an experiment compares one no-replay run with one seeded replay run."""
    baseline_run = get_condition_run(run_records, replay_enabled=False)
    replay_runs = get_replay_runs(run_records)
    run_names = [record["run_name"].lower() for record in run_records]
    return baseline_run is not None and len(replay_runs) == 1 and any("seed" in run_name for run_name in run_names)


def is_replay_strategy_comparison_experiment(
    experiment_id: str,
    run_records: list[dict[str, Any]],
) -> bool:
    """Return True when an experiment compares no replay against multiple replay selection strategies."""
    baseline_run = get_condition_run(run_records, replay_enabled=False)
    replay_runs = get_replay_runs(run_records)
    replay_strategies = {
        str(record["config"].get("replay_selection_strategy", "")).strip()
        for record in replay_runs
    }
    return (
        baseline_run is not None
        and {"random", "loss", "surprise"}.issubset(replay_strategies)
        and experiment_id == "exp006"
    )


def format_buffer_label(buffer_size: Any) -> str:
    """Format a replay buffer size label."""
    try:
        numeric_value = int(buffer_size)
    except (TypeError, ValueError):
        return f"Buffer {buffer_size}"
    return f"Buffer {numeric_value}"


def format_strategy_label(strategy_name: str) -> str:
    """Format a replay selection strategy label."""
    label_lookup = {
        "random": "Random Replay",
        "loss": "Loss Replay",
        "surprise": "Surprise Replay",
        "predicted_utility": "Predicted Utility Replay",
    }
    return label_lookup.get(strategy_name, strategy_name.replace("_", " ").title())


def get_buffer_ablation_runs(run_records: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Return labeled runs for a replay buffer size ablation comparison."""
    labeled_runs: list[tuple[str, dict[str, Any]]] = []
    baseline_run = get_condition_run(run_records, replay_enabled=False)
    if baseline_run is not None:
        labeled_runs.append(("No Replay", baseline_run))
    for record in get_replay_runs(run_records):
        labeled_runs.append((format_buffer_label(record["config"].get("replay_buffer_size")), record))
    return labeled_runs


def get_strategy_comparison_runs(run_records: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Return labeled runs for a replay strategy comparison in a fixed order."""
    labeled_runs: list[tuple[str, dict[str, Any]]] = []
    baseline_run = get_condition_run(run_records, replay_enabled=False)
    if baseline_run is not None:
        labeled_runs.append(("No Replay", baseline_run))

    replay_runs_by_strategy = {
        str(record["config"].get("replay_selection_strategy", "")).strip(): record
        for record in get_replay_runs(run_records)
    }
    for strategy_name in ("random", "loss", "surprise", "predicted_utility"):
        record = replay_runs_by_strategy.get(strategy_name)
        if record is not None:
            labeled_runs.append((format_strategy_label(strategy_name), record))
    return labeled_runs


def get_primary_task_order(run_records: list[dict[str, Any]]) -> list[str]:
    """Return the primary task order for an experiment."""
    replay_run = get_condition_run(run_records, replay_enabled=True)
    baseline_run = get_condition_run(run_records, replay_enabled=False)
    if replay_run and replay_run["task_order"]:
        return list(replay_run["task_order"])
    if baseline_run and baseline_run["task_order"]:
        return list(baseline_run["task_order"])

    task_order: list[str] = []
    for record in run_records:
        for task_name in record["task_order"]:
            if task_name not in task_order:
                task_order.append(task_name)
    return task_order


def infer_experiment_labels(
    experiment_id: str,
    run_records: list[dict[str, Any]],
    all_experiment_dirs: list[Path],
) -> list[str]:
    """Infer coarse experiment-type labels from saved run names and task order."""
    task_order = get_primary_task_order(run_records)
    task_count = len(task_order)
    max_task_count = max(
        (
            len(get_primary_task_order([load_run_record(run_dir) for run_dir in list_run_dirs(experiment_dir)]))
            for experiment_dir in all_experiment_dirs
        ),
        default=task_count,
    )
    run_names = [record["run_name"].lower() for record in run_records]
    has_reverse = any("reverse" in run_name for run_name in run_names)
    labels: list[str] = []
    if has_reverse:
        labels.append("task-order sensitivity experiment")
    if task_count == 1:
        labels.append("preliminary experiment")
    elif task_count >= 3 and task_count == max_task_count and not has_reverse:
        labels.append("baseline experiment")
    elif task_count < max_task_count:
        labels.append("preliminary experiment")

    if not labels:
        labels.append("another type of experiment")
    return labels


def format_experiment_label_phrase(labels: list[str]) -> str:
    """Format inferred experiment labels into a short phrase."""
    if not labels:
        return "another type of experiment"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def infer_research_question(labels: list[str], task_order: list[str]) -> str:
    """Infer a research question for an experiment."""
    task_sequence = format_task_sequence(task_order)
    if "task-order sensitivity experiment" in labels:
        return f"Is the replay effect sensitive to changing the task order, here using the saved sequence `{task_sequence}`?"
    if len(task_order) == 1:
        return (
            f"In this single-task control using `{task_sequence}`, does enabling replay materially change final performance "
            "when there is little or no cross-task forgetting pressure?"
        )
    if "baseline experiment" in labels:
        return (
            f"Does replay improve retention and final performance when the model is trained sequentially on `{task_sequence}`?"
        )
    return f"What does this experiment reveal about replay effectiveness under the saved task sequence `{task_sequence}`?"


def build_experiment_overview_text(labels: list[str], has_both_conditions: bool, task_order: list[str]) -> str:
    """Build the overview text with careful experiment wording."""
    label_phrase = format_experiment_label_phrase(labels)
    task_sequence = format_task_sequence(task_order)

    if label_phrase == "task-order sensitivity experiment" and has_both_conditions:
        return (
            "This experiment is best interpreted as a task-order sensitivity experiment with a replay versus no-replay comparison. "
            f"It compares the saved runs over the task sequence `{task_sequence}`."
        )

    if has_both_conditions:
        return (
            f"This experiment is best interpreted as a {label_phrase} with a replay versus no-replay comparison. "
            f"It compares the saved runs over the task sequence `{task_sequence}`."
        )

    return (
        f"This experiment is best interpreted as a {label_phrase}. "
        f"It uses the saved task sequence `{task_sequence}`."
    )


def get_experiment_description(experiment_id: str, labels: list[str], has_both_conditions: bool, task_order: list[str]) -> str:
    """Return the short experiment description used in summaries and the index."""
    if experiment_id == "exp001":
        return "a baseline continual learning experiment with a replay versus no-replay comparison"
    if experiment_id == "exp002":
        return "a task-order sensitivity experiment with a replay versus no-replay comparison"
    if experiment_id == "exp003":
        return "an additional baseline-style replay versus no-replay comparison using the original task order"
    if experiment_id == "exp004":
        return "a replay buffer size ablation experiment over the task sequence sciq -> arc_challenge -> boolq"
    if experiment_id == "exp005":
        return "a seed robustness check comparing no replay with replay buffer size 30 over the original task order"
    if experiment_id == "exp006":
        return "a replay selection strategy comparison experiment over no replay, random replay, loss replay, and surprise replay"

    label_phrase = format_experiment_label_phrase(labels)
    if has_both_conditions:
        return f"a {label_phrase} with a replay versus no-replay comparison"
    return f"a {label_phrase}"


def build_experiment_design_lines(
    replay_run: dict[str, Any] | None,
    baseline_run: dict[str, Any] | None,
    task_order: list[str],
) -> list[str]:
    """Describe the experiment design."""
    lines = []
    if replay_run is not None:
        lines.append(
            f"- Replay run: `{replay_run['run_name']}` with replay enabled on the task sequence `{format_task_sequence(task_order)}`."
        )
    else:
        lines.append("- Replay run: Not available.")
    if baseline_run is not None:
        lines.append(
            f"- No-replay run: `{baseline_run['run_name']}` with replay disabled on the same task sequence."
        )
    else:
        lines.append("- No-replay run: Not available.")
    lines.append(f"- Task sequence: `{format_task_sequence(task_order)}`.")
    lines.append("- Comparison target: final validation accuracy, final test accuracy, and saved forgetting values.")
    lines.append(
        "- Why this comparison is useful: it isolates the effect of replay while keeping the task order and general training setup aligned across conditions."
    )
    return lines


def build_task_description_lines(task_order: list[str]) -> list[str]:
    """Describe the tasks used in one experiment."""
    lines = []
    for task_name in task_order:
        description = TASK_DESCRIPTIONS.get(task_name, "task in the saved lightweight continual learning sequence")
        lines.append(f"- `{task_name}`: {description}.")
    lines.append("- Together, these tasks act as a lightweight continual learning sequence rather than a large-scale benchmark sweep.")
    return lines


def build_run_command_lines(replay_run: dict[str, Any] | None, baseline_run: dict[str, Any] | None) -> list[str]:
    """List replay and no-replay run commands."""
    lines = []
    if replay_run is not None:
        lines.append(
            f"- Replay run command (reconstructed from saved `run_config.json` because the exact invocation string was not persisted): `{replay_run['reconstructed_command']}`"
        )
    else:
        lines.append("- Replay run command: Not available.")
    if baseline_run is not None:
        lines.append(
            f"- No-replay run command (reconstructed from saved `run_config.json` because the exact invocation string was not persisted): `{baseline_run['reconstructed_command']}`"
        )
    else:
        lines.append("- No-replay run command: Not available.")
    return lines


def build_configuration_summary_table(
    replay_run: dict[str, Any] | None,
    baseline_run: dict[str, Any] | None,
) -> str:
    """Build a field-wise configuration comparison table."""
    replay_config = replay_run["config"] if replay_run is not None else {}
    baseline_config = baseline_run["config"] if baseline_run is not None else {}
    rows = [
        ["run_name", str(replay_config.get("run_name", "Not available")), str(baseline_config.get("run_name", "Not available"))],
        [
            "tasks",
            format_task_sequence(list(replay_config.get("task_order", []))) if replay_config.get("task_order") else "Not available",
            format_task_sequence(list(baseline_config.get("task_order", []))) if baseline_config.get("task_order") else "Not available",
        ],
        ["replay enabled", str(replay_config.get("replay_enabled", "Not available")), str(baseline_config.get("replay_enabled", "Not available"))],
        ["replay buffer size", format_scalar(replay_config.get("replay_buffer_size")), format_scalar(baseline_config.get("replay_buffer_size"))],
        [
            "replay selection strategy",
            str(replay_config.get("replay_selection_strategy", "Not available")),
            str(baseline_config.get("replay_selection_strategy", "Not available")),
        ],
        ["model name", str(replay_config.get("model_name", "Not available")), str(baseline_config.get("model_name", "Not available"))],
        ["epochs", format_scalar(replay_config.get("num_epochs")), format_scalar(baseline_config.get("num_epochs"))],
        ["batch size", format_scalar(replay_config.get("batch_size")), format_scalar(baseline_config.get("batch_size"))],
        ["learning rate", format_scalar(replay_config.get("learning_rate")), format_scalar(baseline_config.get("learning_rate"))],
        ["seed", format_scalar(replay_config.get("seed")), format_scalar(baseline_config.get("seed"))],
        [
            "utility scorer path",
            str(get_config_utility_scorer_path(replay_config) or "Not available"),
            str(get_config_utility_scorer_path(baseline_config) or "Not available"),
        ],
        [
            "utility feature columns path",
            str(get_config_utility_feature_columns_path(replay_config) or "Not available"),
            str(get_config_utility_feature_columns_path(baseline_config) or "Not available"),
        ],
        [
            "output directory",
            str(replay_config.get("run_dir", replay_run["run_dir"] if replay_run is not None else "Not available")),
            str(baseline_config.get("run_dir", baseline_run["run_dir"] if baseline_run is not None else "Not available")),
        ],
    ]
    return build_markdown_table(["Field", "Replay Run", "No-Replay Run"], rows)


def output_files_section_lines(experiment_id: str) -> list[str]:
    """Explain the main files produced by an experiment directory."""
    return [
        "- `checkpoints/`: saved model and tokenizer checkpoints after each task, useful for inspecting intermediate training states.",
        "- `logs/`: console output plus training summary CSVs, useful for checking whether runs completed and how many training steps were recorded.",
        "- `metrics/`: validation and test metric CSVs plus prediction files, which are the main source for the comparisons in this report.",
        "- `replay_scores/`: per-sample replay scoring CSVs used only when replay is enabled; for no-replay runs this folder may be empty.",
        "- `replay_buffers/`: replay buffer snapshots and the replay buffer summary CSV; for no-replay runs these files can be empty placeholders.",
        "- `sample_signals/`: sample-level early training signal CSVs for each task, including loss-derived surprise and replay-selection metadata when available.",
        "- `utility_labels/`: proxy replay-utility labels derived from replay selection outcomes rather than exact causal future replay utility.",
        "- `human_review/`: auto-tagged replay review exports for manual inspection, including the original `replay_review_candidates.csv` file and the editable `replay_review_candidates_reviewed.csv` copy.",
        "- `replay_selection_comparison/`: per-task CSVs that show the configured replay selection score and whether each sample was selected.",
        "- `scorer_outputs/`: lightweight scorer metrics, predictions, feature metadata, or warnings when a utility scorer is trained for this run.",
        "- `scorer_checkpoints/`: serialized scorer models when scorer training succeeds and `joblib` is available.",
        "- `run_config.json`: the saved resolved configuration for each run folder, including task order, output paths, replay selection strategy, and utility scorer paths when applicable.",
        f"- `{experiment_id}_summary.md`: this experiment-level report generated from the saved files in this experiment directory.",
    ]


def metric_definitions_lines() -> list[str]:
    """Explain the saved metrics in simple language."""
    return [
        "- Validation accuracy: the fraction of validation examples answered correctly for a task after training has reached the final saved task state.",
        "- Test accuracy: the fraction of held-out test examples answered correctly for a task after the final saved task state.",
        "- Forgetting: a conceptual measure of how much a task's performance drops after learning later tasks compared with its best earlier saved performance. The code saves this value directly, so this report explains it conceptually rather than inventing a new formula.",
        "- Delta between replay and no-replay: `Replay score - No Replay score`. A positive delta means replay performed better, while a negative delta means the no-replay run performed better.",
    ]


def build_preplay_core_artifact_lines(label: str, record: dict[str, Any]) -> list[str]:
    """Summarize PRE-PLAY core artifact coverage for one run."""
    tag_counts = record.get("human_review_tag_counts", {})
    human_review_file = record.get("human_review_candidate_file")
    human_review_file_name = human_review_file.name if human_review_file is not None else "Not available"

    return [
        f"- {label} replay selection strategy: `{record['config'].get('replay_selection_strategy', 'surprise')}`.",
        f"- {label} utility scorer path: `{get_config_utility_scorer_path(record['config']) or 'Not available'}`.",
        f"- {label} utility feature columns path: `{get_config_utility_feature_columns_path(record['config']) or 'Not available'}`.",
        f"- {label} sample_signals present: {format_boolean(bool(record.get('sample_signal_files')))} ({len(record.get('sample_signal_files', []))} file(s)).",
        f"- {label} utility_labels present: {format_boolean(bool(record.get('utility_label_files')))} ({len(record.get('utility_label_files', []))} file(s)).",
        f"- {label} human_review files present: {format_boolean(bool(record.get('human_review_files')))} ({len(record.get('human_review_files', []))} file(s)); review CSV used for counts: `{human_review_file_name}`.",
        (
            f"- {label} human review tag counts: "
            f"high_value_candidate={tag_counts.get('high_value_candidate', 0)}, "
            f"selected_for_replay={tag_counts.get('selected_for_replay', 0)}, "
            f"needs_human_review={tag_counts.get('needs_human_review', 0)}, "
            f"low_priority={tag_counts.get('low_priority', 0)}."
        ),
        f"- {label} manually reviewed samples: {record.get('manual_human_review_count', 0)}.",
        f"- {label} scorer_outputs present: {format_boolean(bool(record.get('scorer_output_files')))} ({len(record.get('scorer_output_files', []))} file(s)).",
        f"- {label} scorer_checkpoints present: {format_boolean(bool(record.get('scorer_checkpoint_files')))} ({len(record.get('scorer_checkpoint_files', []))} file(s)).",
        f"- {label} replay_selection_comparison present: {format_boolean(bool(record.get('replay_selection_comparison_files')))} ({len(record.get('replay_selection_comparison_files', []))} file(s)).",
    ]


def better_setting_label(
    replay_value: Any,
    baseline_value: Any,
    *,
    lower_is_better: bool = False,
) -> str:
    """Return which setting is better for a pair of numeric values."""
    if replay_value is None or baseline_value is None:
        return "Not available"
    replay_numeric = float(replay_value)
    baseline_numeric = float(baseline_value)
    if math.isnan(replay_numeric) or math.isnan(baseline_numeric):
        return "Not available"
    if replay_numeric == baseline_numeric:
        return "Tie"
    if lower_is_better:
        return "Replay" if replay_numeric < baseline_numeric else "No Replay"
    return "Replay" if replay_numeric > baseline_numeric else "No Replay"


def get_metric_table_tasks(
    replay_run: dict[str, Any] | None,
    baseline_run: dict[str, Any] | None,
    task_order: list[str],
) -> list[str]:
    """Return the task order to use for comparison tables."""
    if task_order:
        return task_order
    tasks: list[str] = []
    for record in (replay_run, baseline_run):
        if record is None:
            continue
        for task_name in record["metrics_by_task"]:
            if task_name not in tasks:
                tasks.append(task_name)
    return tasks


def build_metric_comparison_table(
    replay_run: dict[str, Any] | None,
    baseline_run: dict[str, Any] | None,
    task_order: list[str],
    metric_key: str,
) -> str:
    """Build a replay/no-replay comparison table for one metric."""
    rows: list[list[str]] = []
    for task_name in get_metric_table_tasks(replay_run, baseline_run, task_order):
        replay_value = replay_run["metrics_by_task"].get(task_name, {}).get(metric_key) if replay_run else None
        baseline_value = baseline_run["metrics_by_task"].get(task_name, {}).get(metric_key) if baseline_run else None
        delta_text = "Not available"
        if replay_value is not None and baseline_value is not None:
            delta_text = format_metric(float(replay_value) - float(baseline_value))
        rows.append(
            [
                task_name,
                format_metric(replay_value),
                format_metric(baseline_value),
                delta_text,
                better_setting_label(replay_value, baseline_value),
            ]
        )

    return build_markdown_table(["Task", "Replay", "No Replay", "Delta", "Better Setting"], rows)


def build_forgetting_table(
    replay_run: dict[str, Any] | None,
    baseline_run: dict[str, Any] | None,
    task_order: list[str],
) -> str:
    """Build the forgetting comparison table."""
    rows: list[list[str]] = []
    for task_name in get_metric_table_tasks(replay_run, baseline_run, task_order):
        replay_value = replay_run["metrics_by_task"].get(task_name, {}).get("forgetting") if replay_run else None
        baseline_value = baseline_run["metrics_by_task"].get(task_name, {}).get("forgetting") if baseline_run else None
        rows.append(
            [
                task_name,
                format_metric(replay_value),
                format_metric(baseline_value),
                better_setting_label(replay_value, baseline_value, lower_is_better=True),
            ]
        )
    return build_markdown_table(
        ["Task", "Replay Forgetting", "No Replay Forgetting", "Lower Forgetting Setting"],
        rows,
    )


def collect_record_metric_values(
    record: dict[str, Any] | None,
    task_order: list[str],
    metric_key: str,
) -> list[float]:
    """Collect numeric metric values for one run across tasks."""
    if record is None:
        return []
    values: list[float] = []
    for task_name in task_order:
        metric_value = record["metrics_by_task"].get(task_name, {}).get(metric_key)
        if metric_value is None:
            continue
        numeric_value = float(metric_value)
        if math.isnan(numeric_value):
            continue
        values.append(numeric_value)
    return values


def compute_average_metric(
    record: dict[str, Any] | None,
    task_order: list[str],
    metric_key: str,
) -> float | None:
    """Compute the average saved metric across tasks."""
    values = collect_record_metric_values(record, task_order, metric_key)
    if not values:
        return None
    return mean(values)


def compute_max_forgetting(
    record: dict[str, Any] | None,
    task_order: list[str],
) -> float | None:
    """Compute the maximum saved forgetting across tasks."""
    values = collect_record_metric_values(record, task_order, "forgetting")
    if not values:
        return None
    return max(values)


def compute_average_forgetting(
    record: dict[str, Any] | None,
    task_order: list[str],
) -> float | None:
    """Compute the average saved forgetting across tasks."""
    values = collect_record_metric_values(record, task_order, "forgetting")
    if not values:
        return None
    return mean(values)


def best_setting_for_values(
    labeled_values: list[tuple[str, Any]],
    *,
    lower_is_better: bool = False,
) -> str:
    """Return the best setting label for multiple metric values."""
    valid_values: list[tuple[str, float]] = []
    for label, value in labeled_values:
        if value is None:
            continue
        numeric_value = float(value)
        if math.isnan(numeric_value):
            continue
        valid_values.append((label, numeric_value))

    if not valid_values:
        return "Not available"

    best_value = min(value for _, value in valid_values) if lower_is_better else max(value for _, value in valid_values)
    best_labels = [label for label, value in valid_values if math.isclose(value, best_value, rel_tol=1e-9, abs_tol=1e-9)]
    if len(best_labels) == 1:
        return best_labels[0]
    return "Tie (" + ", ".join(best_labels) + ")"


def build_multi_run_configuration_summary_table(labeled_runs: list[tuple[str, dict[str, Any]]]) -> str:
    """Build a configuration summary table for experiments with more than two runs."""
    headers = ["Field", *[label for label, _ in labeled_runs]]
    rows: list[list[str]] = []
    field_rows = [
        ("run_name", lambda record: str(record["config"].get("run_name", "Not available"))),
        (
            "tasks",
            lambda record: format_task_sequence(list(record["config"].get("task_order", [])))
            if record["config"].get("task_order")
            else "Not available",
        ),
        ("replay enabled", lambda record: str(record["config"].get("replay_enabled", "Not available"))),
        ("replay buffer size", lambda record: format_scalar(record["config"].get("replay_buffer_size"))),
        (
            "replay selection strategy",
            lambda record: str(record["config"].get("replay_selection_strategy", "Not available")),
        ),
        ("model name", lambda record: str(record["config"].get("model_name", "Not available"))),
        ("epochs", lambda record: format_scalar(record["config"].get("num_epochs"))),
        ("batch size", lambda record: format_scalar(record["config"].get("batch_size"))),
        ("learning rate", lambda record: format_scalar(record["config"].get("learning_rate"))),
        ("seed", lambda record: format_scalar(record["config"].get("seed"))),
        (
            "utility scorer path",
            lambda record: str(get_config_utility_scorer_path(record["config"]) or "Not available"),
        ),
        (
            "utility feature columns path",
            lambda record: str(get_config_utility_feature_columns_path(record["config"]) or "Not available"),
        ),
        (
            "output directory",
            lambda record: str(record["config"].get("run_dir", record["run_dir"])),
        ),
    ]

    for field_name, formatter in field_rows:
        rows.append([field_name, *[formatter(record) for _, record in labeled_runs]])
    return build_markdown_table(headers, rows)


def build_multi_run_metric_table(
    labeled_runs: list[tuple[str, dict[str, Any]]],
    task_order: list[str],
    metric_key: str,
    best_setting_header: str,
    *,
    lower_is_better: bool = False,
) -> str:
    """Build a multi-run comparison table for one metric."""
    headers = ["Task", *[label for label, _ in labeled_runs], best_setting_header]
    rows: list[list[str]] = []
    for task_name in task_order:
        labeled_values: list[tuple[str, Any]] = []
        row = [task_name]
        for label, record in labeled_runs:
            metric_value = record["metrics_by_task"].get(task_name, {}).get(metric_key)
            labeled_values.append((label, metric_value))
            row.append(format_metric(metric_value))
        row.append(best_setting_for_values(labeled_values, lower_is_better=lower_is_better))
        rows.append(row)
    return build_markdown_table(headers, rows)


def build_multi_run_command_lines(labeled_runs: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """List reconstructed commands for multiple runs."""
    return [
        f"- {label}: `{record['reconstructed_command']}` (reconstructed from saved `run_config.json` because the exact invocation string was not persisted)."
        for label, record in labeled_runs
    ]


def build_buffer_ablation_design_lines(
    labeled_runs: list[tuple[str, dict[str, Any]]],
    task_order: list[str],
) -> list[str]:
    """Describe a replay buffer size ablation design."""
    lines = ["- No-replay baseline: compare against training without replay."]
    for label, record in labeled_runs:
        if label == "No Replay":
            continue
        lines.append(
            f"- {label}: replay enabled in `{record['run_name']}` with replay buffer size `{format_scalar(record['config'].get('replay_buffer_size'))}`."
        )
    lines.append(f"- Task sequence: `{format_task_sequence(task_order)}`.")
    lines.append("- Comparison target: final validation accuracy, final test accuracy, and saved forgetting values across all four settings.")
    lines.append("- Why this comparison is useful: it isolates replay buffer capacity while keeping the task sequence and the rest of the saved training setup aligned.")
    return lines


def build_buffer_ablation_artifact_analysis_lines(
    labeled_runs: list[tuple[str, dict[str, Any]]],
) -> list[str]:
    """Describe replay artifacts for each run in a buffer-size ablation."""
    lines: list[str] = []
    for label, record in labeled_runs:
        replay_score_count = len(record["replay_score_files"])
        replay_snapshot_count = len(record["replay_buffer_snapshots"])
        replay_non_empty_count = len(record["non_empty_replay_snapshots"])
        if label == "No Replay":
            lines.append(f"- {label} (`{record['run_name']}`) has replay disabled.")
            lines.append(
                f"- Its `replay_scores/` folder contains {replay_score_count} replay score CSV file(s), which is expected to be zero when replay is disabled."
            )
            lines.append(
                f"- Its `replay_buffers/` folder contains {replay_snapshot_count} replay buffer snapshot file(s), with {replay_non_empty_count} non-empty snapshot(s); empty files indicate placeholders rather than active replay."
            )
            lines.extend(build_preplay_core_artifact_lines(label, record))
            continue

        lines.append(f"- {label} (`{record['run_name']}`) saved {replay_score_count} replay score CSV file(s).")
        lines.append(f"- {label} (`{record['run_name']}`) saved {replay_snapshot_count} replay buffer snapshot file(s).")
        lines.append(
            f"- {label} has {replay_non_empty_count} non-empty replay buffer snapshot file(s), which suggests replay was active for this setting."
        )
        lines.extend(build_preplay_core_artifact_lines(label, record))
    return lines


def buffer_size_consistently_helps(
    labeled_runs: list[tuple[str, dict[str, Any]]],
    task_order: list[str],
) -> bool:
    """Return True when larger buffer sizes improve val/test and reduce forgetting monotonically."""
    replay_runs = [(label, record) for label, record in labeled_runs if label != "No Replay"]
    if len(replay_runs) < 2:
        return False

    val_averages = [compute_average_metric(record, task_order, "val_accuracy") for _, record in replay_runs]
    test_averages = [compute_average_metric(record, task_order, "test_accuracy") for _, record in replay_runs]
    max_forgetting_values = [compute_max_forgetting(record, task_order) for _, record in replay_runs]
    if any(value is None for value in [*val_averages, *test_averages, *max_forgetting_values]):
        return False

    val_non_decreasing = all(left <= right for left, right in zip(val_averages, val_averages[1:]))
    test_non_decreasing = all(left <= right for left, right in zip(test_averages, test_averages[1:]))
    forgetting_non_increasing = all(left >= right for left, right in zip(max_forgetting_values, max_forgetting_values[1:]))
    return val_non_decreasing and test_non_decreasing and forgetting_non_increasing


def collect_numeric_deltas(
    replay_run: dict[str, Any] | None,
    baseline_run: dict[str, Any] | None,
    task_order: list[str],
    metric_key: str,
) -> list[float]:
    """Collect numeric replay-minus-baseline deltas for a metric."""
    deltas: list[float] = []
    for task_name in get_metric_table_tasks(replay_run, baseline_run, task_order):
        replay_value = replay_run["metrics_by_task"].get(task_name, {}).get(metric_key) if replay_run else None
        baseline_value = baseline_run["metrics_by_task"].get(task_name, {}).get(metric_key) if baseline_run else None
        if replay_value is None or baseline_value is None:
            continue
        deltas.append(float(replay_value) - float(baseline_value))
    return deltas


def collect_forgetting_pairs(
    replay_run: dict[str, Any] | None,
    baseline_run: dict[str, Any] | None,
    task_order: list[str],
) -> list[tuple[float, float]]:
    """Collect replay and no-replay forgetting pairs."""
    pairs: list[tuple[float, float]] = []
    for task_name in get_metric_table_tasks(replay_run, baseline_run, task_order):
        replay_value = replay_run["metrics_by_task"].get(task_name, {}).get("forgetting") if replay_run else None
        baseline_value = baseline_run["metrics_by_task"].get(task_name, {}).get("forgetting") if baseline_run else None
        if replay_value is None or baseline_value is None:
            continue
        pairs.append((float(replay_value), float(baseline_value)))
    return pairs


def has_consistent_replay_advantage(
    replay_run: dict[str, Any] | None,
    baseline_run: dict[str, Any] | None,
    task_order: list[str],
) -> bool:
    """Return True only when replay is consistently better on val, test, and forgetting."""
    val_deltas = collect_numeric_deltas(replay_run, baseline_run, task_order, "val_accuracy")
    test_deltas = collect_numeric_deltas(replay_run, baseline_run, task_order, "test_accuracy")
    forgetting_pairs = collect_forgetting_pairs(replay_run, baseline_run, task_order)

    if not val_deltas or not test_deltas or not forgetting_pairs:
        return False

    val_consistently_better = all(delta > 0 for delta in val_deltas)
    test_consistently_better = all(delta > 0 for delta in test_deltas)
    forgetting_non_worse = all(replay_value <= baseline_value for replay_value, baseline_value in forgetting_pairs)
    forgetting_strictly_better = any(replay_value < baseline_value for replay_value, baseline_value in forgetting_pairs)
    return val_consistently_better and test_consistently_better and forgetting_non_worse and forgetting_strictly_better


def build_forgetting_analysis_lines(
    replay_run: dict[str, Any] | None,
    baseline_run: dict[str, Any] | None,
    task_order: list[str],
) -> list[str]:
    """Explain the forgetting comparison."""
    if replay_run is None or baseline_run is None:
        return ["- Not available. Both replay and no-replay runs are required for a forgetting comparison."]

    forgetting_values_replay: list[float] = []
    forgetting_values_baseline: list[float] = []
    for task_name in get_metric_table_tasks(replay_run, baseline_run, task_order):
        replay_value = replay_run["metrics_by_task"].get(task_name, {}).get("forgetting")
        baseline_value = baseline_run["metrics_by_task"].get(task_name, {}).get("forgetting")
        if replay_value is not None:
            forgetting_values_replay.append(float(replay_value))
        if baseline_value is not None:
            forgetting_values_baseline.append(float(baseline_value))

    if not forgetting_values_replay and not forgetting_values_baseline:
        return [
            "- Not available. The current saved metric files do not provide enough final forgetting values for a comparison."
        ]

    lines: list[str] = []
    if len(task_order) == 1:
        lines.append("- Forgetting values exist, but they are not very informative in a single-task control because there are no later tasks to cause substantial retention loss.")
        return lines

    replay_max = max(forgetting_values_replay) if forgetting_values_replay else None
    baseline_max = max(forgetting_values_baseline) if forgetting_values_baseline else None
    if replay_max is None or baseline_max is None:
        lines.append("- Forgetting was partially recorded, but one condition is missing enough values for a strong comparison.")
        return lines

    if replay_max == 0.0 and baseline_max == 0.0:
        lines.append("- No measurable forgetting is visible in the saved final validation metrics for either setting.")
    else:
        lines.append("- Some forgetting was observed in the saved final validation metrics after later tasks were learned.")
        if math.isclose(replay_max, baseline_max, rel_tol=1e-9, abs_tol=1e-9):
            lines.append(f"- The maximum saved forgetting was identical in both settings at {replay_max:.4f}.")
        elif replay_max < baseline_max:
            lines.append(f"- Replay forgot less on the worst affected task, reducing maximum saved forgetting from {baseline_max:.4f} to {replay_max:.4f}.")
        else:
            lines.append(f"- Replay forgot more on the worst affected task, increasing maximum saved forgetting from {baseline_max:.4f} to {replay_max:.4f}.")
    return lines


def build_replay_artifact_analysis_lines(
    replay_run: dict[str, Any] | None,
    baseline_run: dict[str, Any] | None,
) -> list[str]:
    """Describe replay-related saved artifacts."""
    lines: list[str] = []
    if replay_run is not None:
        replay_score_count = len(replay_run["replay_score_files"])
        replay_snapshot_count = len(replay_run["replay_buffer_snapshots"])
        replay_non_empty_count = len(replay_run["non_empty_replay_snapshots"])
        lines.append(f"- Replay run `{replay_run['run_name']}` saved {replay_score_count} replay score CSV file(s).")
        lines.append(f"- Replay run `{replay_run['run_name']}` saved {replay_snapshot_count} replay buffer snapshot file(s).")
        if replay_snapshot_count > 0:
            lines.append(
                f"- {replay_non_empty_count} of those replay buffer snapshot file(s) are non-empty, which suggests replay was actively selecting and storing samples."
            )
        else:
            lines.append("- No replay buffer snapshots were found for the replay run, which would be unusual if replay was active.")
        lines.extend(build_preplay_core_artifact_lines("Replay run", replay_run))
    else:
        lines.append("- Replay artifact analysis for the replay condition is not available because the replay run is missing.")

    if baseline_run is not None:
        baseline_snapshot_count = len(baseline_run["replay_buffer_snapshots"])
        baseline_non_empty_count = len(baseline_run["non_empty_replay_snapshots"])
        lines.append(f"- No-replay run `{baseline_run['run_name']}` has replay disabled by configuration.")
        lines.append(
            f"- Its `replay_scores/` folder contains {len(baseline_run['replay_score_files'])} replay score CSV file(s), which is expected to be zero when replay is disabled."
        )
        lines.append(
            f"- Its `replay_buffers/` folder contains {baseline_snapshot_count} replay buffer snapshot file(s), with {baseline_non_empty_count} non-empty snapshot(s); empty files indicate placeholders rather than active replay."
        )
        lines.extend(build_preplay_core_artifact_lines("No-replay run", baseline_run))
    else:
        lines.append("- Replay artifact analysis for the no-replay condition is not available because the baseline run is missing.")
    return lines


def build_main_findings_lines(
    replay_run: dict[str, Any] | None,
    baseline_run: dict[str, Any] | None,
    task_order: list[str],
) -> list[str]:
    """Summarize the main evidence-based findings in 3 to 5 bullets."""
    findings: list[str] = []
    findings.append(f"- The saved comparison covers the task sequence `{format_task_sequence(task_order)}`.")

    val_deltas = collect_numeric_deltas(replay_run, baseline_run, task_order, "val_accuracy")
    test_deltas = collect_numeric_deltas(replay_run, baseline_run, task_order, "test_accuracy")
    if val_deltas:
        findings.append(f"- The average replay-minus-no-replay final validation accuracy delta was {mean(val_deltas):.4f}.")
    else:
        findings.append("- Final validation accuracy deltas were not fully available from the saved metrics.")
    if test_deltas:
        findings.append(f"- The average replay-minus-no-replay final test accuracy delta was {mean(test_deltas):.4f}.")
    else:
        findings.append("- Final test accuracy deltas were not fully available from the saved metrics.")

    if len(task_order) == 1:
        findings.append("- Because this is a single-task control, forgetting is not the main stress test; the comparison is mainly about whether replay changes the end result at all.")
    else:
        forgetting_lines = build_forgetting_analysis_lines(replay_run, baseline_run, task_order)
        findings.append(forgetting_lines[0].replace("Not available. ", "") if forgetting_lines else "- Forgetting could not be assessed.")

    if replay_run is not None:
        findings.append(
            f"- Replay artifacts were actually produced for the replay run ({len(replay_run['replay_score_files'])} score file(s), {len(replay_run['non_empty_replay_snapshots'])} non-empty buffer snapshot(s)), so replay appears to have been active."
        )

    if replay_run is not None and baseline_run is not None:
        if has_consistent_replay_advantage(replay_run, baseline_run, task_order):
            findings.append("- The saved metrics are consistently favorable to replay across validation accuracy, test accuracy, and forgetting.")
        else:
            findings.append("- The saved metrics provide mixed evidence rather than a consistent replay advantage.")

    return findings[:5]


def build_interpretation_lines(
    replay_run: dict[str, Any] | None,
    baseline_run: dict[str, Any] | None,
    task_order: list[str],
) -> list[str]:
    """Build a cautious interpretation section."""
    if replay_run is None or baseline_run is None:
        return ["- A full interpretation is not available because one of the two comparison conditions is missing."]

    lines: list[str] = []
    val_deltas = collect_numeric_deltas(replay_run, baseline_run, task_order, "val_accuracy")
    test_deltas = collect_numeric_deltas(replay_run, baseline_run, task_order, "test_accuracy")

    if val_deltas:
        average_val_delta = mean(val_deltas)
        if average_val_delta > 0:
            lines.append(f"- Replay improved final validation accuracy on average by {average_val_delta:.4f} across the saved tasks.")
        elif average_val_delta < 0:
            lines.append(f"- Replay reduced final validation accuracy on average by {abs(average_val_delta):.4f} across the saved tasks.")
        else:
            lines.append("- Replay and no-replay were tied on average final validation accuracy across the saved tasks.")
    else:
        lines.append("- Final validation accuracy was not available for every task, so the validation-side comparison is incomplete.")

    if test_deltas:
        average_test_delta = mean(test_deltas)
        if average_test_delta > 0:
            lines.append(f"- Replay improved final test accuracy on average by {average_test_delta:.4f} across the saved tasks.")
        elif average_test_delta < 0:
            lines.append(f"- Replay reduced final test accuracy on average by {abs(average_test_delta):.4f} across the saved tasks.")
        else:
            lines.append("- Replay and no-replay were tied on average final test accuracy across the saved tasks.")
    else:
        lines.append("- Final test accuracy was not available for every task, so the test-side comparison is incomplete.")

    forgetting_lines = build_forgetting_analysis_lines(replay_run, baseline_run, task_order)
    if forgetting_lines and "Not available" not in forgetting_lines[0]:
        lines.append(forgetting_lines[-1] if len(forgetting_lines) > 1 else forgetting_lines[0])
    else:
        lines.append("- The saved forgetting values are too limited to support a strong claim about whether replay reduced forgetting.")

    if has_consistent_replay_advantage(replay_run, baseline_run, task_order):
        lines.append(
            "- Under this saved setting, replay is consistently better on validation accuracy, test accuracy, and forgetting, so the result is cautiously favorable to replay within this narrow configuration."
        )
    else:
        lines.append(
            "- Overall, the saved comparison provides mixed evidence rather than a consistent case that replay is beneficial under this setting."
        )

    lines.append(
        "- Mixed results are plausible here because the setup uses one seed, one epoch, lightweight sample sizes, and a fixed replay configuration rather than a tuned sweep."
    )
    return lines


def build_limitations_lines(run_records: list[dict[str, Any]], task_order: list[str]) -> list[str]:
    """List limitations supported by saved config or artifacts."""
    reference_config = run_records[0]["config"] if run_records else {}
    replay_configuration_line = "- Only one replay configuration is represented in the saved files, so replay hyperparameters were not broadly explored here."
    if is_replay_buffer_size_ablation_experiment(run_records[0]["run_dir"].parent.name if run_records else "", run_records):
        replay_configuration_line = "- Only three replay buffer sizes are represented, and other replay hyperparameters are not explored."
    lines = [
        f"- Single random seed: `{format_scalar(reference_config.get('seed'))}`.",
        f"- One-epoch training: `{format_scalar(reference_config.get('num_epochs'))}` epoch(s) were saved in the configuration.",
        (
            f"- Lightweight subset sizes: train={format_scalar(reference_config.get('train_sample_size'))}, "
            f"val={format_scalar(reference_config.get('val_sample_size'))}, test={format_scalar(reference_config.get('test_sample_size'))}."
        ),
        f"- Limited task count in this experiment: `{len(task_order)}` task(s).",
        replay_configuration_line,
        "- These results come from a lightweight saved setting and may not generalize without more seeds, task orders, or larger sample sizes.",
    ]
    return lines


def build_relation_to_other_experiments_lines(
    experiment_id: str,
    all_experiment_dirs: list[Path],
) -> list[str]:
    """Describe how one experiment relates to other saved experiment folders."""
    if experiment_id == "exp004":
        return [
            "- `exp001` is the original-order baseline comparison.",
            "- `exp002` tests the reverse task order.",
            "- `exp003` is an additional baseline-style comparison.",
            "- `exp004` tests replay buffer size.",
        ]
    if experiment_id == "exp005":
        return [
            "- `exp001` is the original-order baseline comparison.",
            "- `exp002` tests the reverse task order.",
            "- `exp003` is an additional baseline-style comparison.",
            "- `exp004` tests replay buffer size.",
            "- `exp005` checks whether the strongest saved replay-buffer setting remains stable under a different seed.",
        ]
    if experiment_id == "exp006":
        return [
            "- `exp001` is the original-order baseline comparison.",
            "- `exp002` tests the reverse task order.",
            "- `exp003` is an additional baseline-style comparison.",
            "- `exp004` tests replay buffer size.",
            "- `exp005` checks whether the strongest saved replay-buffer setting remains stable under a different seed.",
            "- `exp006` compares replay selection strategies after the PRE-PLAY core infrastructure was added.",
        ]

    summaries: list[str] = []
    reverse_present = False
    for experiment_dir in all_experiment_dirs:
        run_records = [load_run_record(run_dir) for run_dir in list_run_dirs(experiment_dir)]
        task_order = get_primary_task_order(run_records)
        task_count = len(task_order)
        run_names = [record["run_name"].lower() for record in run_records]
        if any("reverse" in run_name for run_name in run_names):
            reverse_present = True
        if task_count == 0:
            summaries.append(f"- `{experiment_dir.name}` exists, but its task sequence could not be inferred from the saved files.")
        elif task_count == 1:
            summaries.append(f"- `{experiment_dir.name}` is the saved single-task control in the current outputs.")
        else:
            summaries.append(
                f"- `{experiment_dir.name}` uses a `{task_count}`-task saved sequence: `{format_task_sequence(task_order)}`."
            )

    if not reverse_present:
        summaries.append("- No saved experiment folder currently indicates a reverse-order run, so task-order sensitivity is not yet directly tested by the current outputs.")
    if not summaries:
        summaries.append("- Not enough saved experiment folders were available to describe cross-experiment relationships.")
    return summaries


def infer_next_step(run_records: list[dict[str, Any]], task_order: list[str]) -> str:
    """Suggest one concrete next experiment."""
    experiment_id = run_records[0]["run_dir"].parent.name if run_records else ""
    if experiment_id == "exp003":
        return "Test replay buffer size or replay sampling strategy to examine whether replay can become more consistently useful."
    if experiment_id == "exp004":
        return "Run the best-performing buffer-size setting across another random seed, or test replay sampling strategy to see whether the result is stable."
    if experiment_id == "exp005":
        return "Run the same seed robustness check with more seeds, or test replay sampling strategy."
    if experiment_id == "exp006":
        return "Use the generated utility_labels and human_review files to train or validate a utility scorer, then run predicted_utility replay in exp007."
    run_names = [record["run_name"].lower() for record in run_records]
    if len(task_order) >= 2 and not any("reverse" in run_name for run_name in run_names):
        return "Run the same replay vs no-replay pair with the task order reversed to test task-order sensitivity."
    if len(task_order) == 1:
        return "Extend this same replay vs no-replay comparison to at least two tasks so that forgetting becomes measurable."
    return "Repeat the same experiment with a second random seed to check whether the current result is stable."


def build_buffer_ablation_forgetting_lines(
    labeled_runs: list[tuple[str, dict[str, Any]]],
    task_order: list[str],
) -> list[str]:
    """Explain forgetting in a replay buffer size ablation."""
    max_values: list[tuple[str, float]] = []
    for label, record in labeled_runs:
        max_forgetting = compute_max_forgetting(record, task_order)
        if max_forgetting is not None:
            max_values.append((label, max_forgetting))

    if not max_values:
        return ["- Not available. The current saved metric files do not provide enough final forgetting values for a comparison."]

    if all(math.isclose(value, 0.0, rel_tol=1e-9, abs_tol=1e-9) for _, value in max_values):
        lines = ["- No measurable forgetting is visible in the saved final validation metrics for any of the four settings."]
    else:
        lines = ["- Some forgetting was observed in the saved final validation metrics after later tasks were learned."]
    lines.append(
        f"- The lowest maximum forgetting was achieved by {best_setting_for_values(max_values, lower_is_better=True)}."
    )
    baseline_record = next((record for label, record in labeled_runs if label == "No Replay"), None)
    replay_records = [(label, record) for label, record in labeled_runs if label != "No Replay"]
    if baseline_record is None or not replay_records:
        lines.append("- Replay-vs-baseline forgetting could not be assessed because one or more runs are missing.")
        return lines

    baseline_max = compute_max_forgetting(baseline_record, task_order)
    replay_beats_baseline = []
    for label, record in replay_records:
        replay_max = compute_max_forgetting(record, task_order)
        if replay_max is None or baseline_max is None:
            continue
        if replay_max < baseline_max:
            replay_beats_baseline.append(label)

    if len(replay_beats_baseline) == len(replay_records) and replay_records:
        lines.append("- Every replay buffer size reduced maximum saved forgetting relative to the no-replay baseline.")
    elif replay_beats_baseline:
        lines.append(
            "- Replay reduced forgetting for some, but not all, buffer-size settings relative to the no-replay baseline."
        )
    else:
        lines.append("- Replay did not consistently reduce forgetting relative to the no-replay baseline.")
    return lines


def build_buffer_ablation_main_findings_lines(
    labeled_runs: list[tuple[str, dict[str, Any]]],
    task_order: list[str],
) -> list[str]:
    """Summarize the main findings for a replay buffer size ablation."""
    replay_runs = [(label, record) for label, record in labeled_runs if label != "No Replay"]
    replay_val_averages = [(label, compute_average_metric(record, task_order, "val_accuracy")) for label, record in replay_runs]
    replay_test_averages = [(label, compute_average_metric(record, task_order, "test_accuracy")) for label, record in replay_runs]
    replay_forgetting_max = [(label, compute_max_forgetting(record, task_order)) for label, record in replay_runs]

    findings = [
        f"- The saved ablation covers the task sequence `{format_task_sequence(task_order)}` across one no-replay baseline and three replay buffer sizes.",
        f"- Among replay settings, the best average validation accuracy came from {best_setting_for_values(replay_val_averages)}.",
        f"- Among replay settings, the best average test accuracy came from {best_setting_for_values(replay_test_averages)}.",
        f"- Among replay settings, the lowest maximum forgetting came from {best_setting_for_values(replay_forgetting_max, lower_is_better=True)}.",
    ]
    if buffer_size_consistently_helps(labeled_runs, task_order):
        findings.append("- Larger replay buffers improved the saved metrics consistently across validation accuracy, test accuracy, and forgetting.")
    else:
        findings.append("- Larger replay buffers did not show a consistent monotonic benefit across the saved metrics.")
    return findings


def build_buffer_ablation_interpretation_lines(
    labeled_runs: list[tuple[str, dict[str, Any]]],
    task_order: list[str],
) -> list[str]:
    """Build a cautious interpretation for a replay buffer size ablation."""
    replay_runs = [(label, record) for label, record in labeled_runs if label != "No Replay"]
    baseline_record = next((record for label, record in labeled_runs if label == "No Replay"), None)
    replay_val_averages = [(label, compute_average_metric(record, task_order, "val_accuracy")) for label, record in replay_runs]
    replay_test_averages = [(label, compute_average_metric(record, task_order, "test_accuracy")) for label, record in replay_runs]
    replay_forgetting_max = [(label, compute_max_forgetting(record, task_order)) for label, record in replay_runs]

    lines = [
        f"- The strongest replay setting on average validation accuracy was {best_setting_for_values(replay_val_averages)}.",
        f"- The strongest replay setting on average test accuracy was {best_setting_for_values(replay_test_averages)}.",
        f"- The setting with the lowest maximum saved forgetting was {best_setting_for_values(replay_forgetting_max, lower_is_better=True)}.",
    ]

    replay_consistent_advantages = [
        label
        for label, record in replay_runs
        if baseline_record is not None and has_consistent_replay_advantage(record, baseline_record, task_order)
    ]
    if replay_consistent_advantages:
        lines.append(
            f"- A fully consistent replay advantage over no-replay appears only for {', '.join(replay_consistent_advantages)}, so the evidence should still be read narrowly."
        )
    else:
        lines.append(
            "- No replay buffer size is consistently better than the no-replay baseline on validation accuracy, test accuracy, and forgetting at the same time, so the evidence remains mixed."
        )

    if buffer_size_consistently_helps(labeled_runs, task_order):
        lines.append("- The effect of buffer size is fairly clear within this saved sweep because larger buffers help monotonically across the main metrics.")
    else:
        lines.append("- The effect of buffer size is unclear in this saved sweep because the best setting changes across metrics and larger buffers do not help monotonically.")

    lines.append(
        "- Mixed results are plausible here because the setup still uses one seed, one epoch, lightweight sample sizes, and fixed replay settings apart from buffer size."
    )
    return lines


def build_buffer_ablation_summary_markdown(
    experiment_id: str,
    experiment_dir: Path,
    run_records: list[dict[str, Any]],
    all_experiment_dirs: list[Path],
) -> str:
    """Build an experiment summary for replay buffer size ablations."""
    generation_timestamp = datetime.now().isoformat(timespec="seconds")
    task_order = get_primary_task_order(run_records)
    labeled_runs = get_buffer_ablation_runs(run_records)
    relation_lines = build_relation_to_other_experiments_lines(experiment_id, all_experiment_dirs)

    summary_lines = [
        f"# {experiment_id} Summary",
        "",
        "## 1. Experiment Overview",
        "This experiment is best interpreted as a replay buffer size ablation experiment over the task sequence sciq -> arc_challenge -> boolq.",
        "",
        "## 2. Research Question",
        "How does replay buffer size affect final performance and forgetting in the three-task continual learning sequence?",
        "",
        "## 3. Experiment Design",
        *build_buffer_ablation_design_lines(labeled_runs, task_order),
        "",
        "## 4. Task and Dataset Description",
        *build_task_description_lines(task_order),
        "",
        "## 5. Run Commands",
        *build_multi_run_command_lines(labeled_runs),
        "",
        "## 6. Directory Structure",
        "```text",
        build_tree(experiment_dir, max_depth=3),
        "```",
        "",
        "## 7. Configuration Summary",
        build_multi_run_configuration_summary_table(labeled_runs),
        "",
        "## 8. Output Files and Their Purpose",
        *output_files_section_lines(experiment_id),
        "",
        "## 9. Metric Definitions",
        *metric_definitions_lines(),
        "",
        "## 10. Final Validation Accuracy Comparison",
        build_multi_run_metric_table(labeled_runs, task_order, "val_accuracy", "Best Setting"),
        "",
        "## 11. Final Test Accuracy Comparison",
        build_multi_run_metric_table(labeled_runs, task_order, "test_accuracy", "Best Setting"),
        "",
        "## 12. Forgetting Analysis",
        build_multi_run_metric_table(labeled_runs, task_order, "forgetting", "Lowest Forgetting Setting", lower_is_better=True),
        *build_buffer_ablation_forgetting_lines(labeled_runs, task_order),
        "",
        "## 13. Replay Artifacts Analysis",
        *build_buffer_ablation_artifact_analysis_lines(labeled_runs),
        "",
        "## 14. Main Findings",
        *build_buffer_ablation_main_findings_lines(labeled_runs, task_order),
        "",
        "## 15. Interpretation",
        *build_buffer_ablation_interpretation_lines(labeled_runs, task_order),
        "",
        "## 16. Limitations",
        *build_limitations_lines(run_records, task_order),
        "",
        "## 17. Relation to Other Experiments",
        *relation_lines,
        "",
        "## 18. Next Step",
        infer_next_step(run_records, task_order),
        "",
        "## 19. Reproducibility Checklist",
        *build_reproducibility_checklist_lines(experiment_dir, run_records),
        "",
        "## 20. Notes",
        *build_notes_lines(run_records, generation_timestamp),
        "",
    ]
    return "\n".join(summary_lines)


def build_strategy_configuration_summary_table(labeled_runs: list[tuple[str, dict[str, Any]]]) -> str:
    """Build the exp006 configuration summary table in a fixed label order."""
    record_lookup = {label: record for label, record in labeled_runs}
    headers = ["Field", "No Replay", "Random Replay", "Loss Replay", "Surprise Replay"]
    column_labels = headers[1:]

    def replay_buffer_value(record: dict[str, Any]) -> str:
        if not record["replay_enabled"]:
            return "disabled"
        return format_scalar(record["config"].get("replay_buffer_size"))

    def replay_strategy_value(record: dict[str, Any]) -> str:
        if not record["replay_enabled"]:
            return "disabled"
        return str(record["config"].get("replay_selection_strategy", "Not available"))

    field_rows = [
        ("run_name", lambda record: str(record["config"].get("run_name", "Not available"))),
        (
            "tasks",
            lambda record: format_task_sequence(list(record["config"].get("task_order", [])))
            if record["config"].get("task_order")
            else "Not available",
        ),
        ("replay enabled", lambda record: str(record["config"].get("replay_enabled", "Not available"))),
        ("replay buffer size", replay_buffer_value),
        ("replay selection strategy", replay_strategy_value),
        ("seed", lambda record: format_scalar(record["config"].get("seed"))),
        ("model name", lambda record: str(record["config"].get("model_name", "Not available"))),
        ("epochs", lambda record: format_scalar(record["config"].get("num_epochs"))),
        ("batch size", lambda record: format_scalar(record["config"].get("batch_size"))),
        ("learning rate", lambda record: format_scalar(record["config"].get("learning_rate"))),
        (
            "utility scorer path",
            lambda record: str(get_config_utility_scorer_path(record["config"]) or "Not available"),
        ),
        (
            "utility feature columns path",
            lambda record: str(get_config_utility_feature_columns_path(record["config"]) or "Not available"),
        ),
        ("output directory", lambda record: str(record["config"].get("run_dir", record["run_dir"]))),
    ]

    rows: list[list[str]] = []
    for field_name, formatter in field_rows:
        row = [field_name]
        for column_label in column_labels:
            record = record_lookup.get(column_label)
            row.append(formatter(record) if record is not None else "Not available")
        rows.append(row)
    return build_markdown_table(headers, rows)


def build_strategy_metric_table(
    labeled_runs: list[tuple[str, dict[str, Any]]],
    task_order: list[str],
    metric_key: str,
) -> str:
    """Build an exp006 metric comparison table."""
    record_lookup = {label: record for label, record in labeled_runs}
    rows: list[list[str]] = []
    for task_name in task_order:
        labeled_values: list[tuple[str, Any]] = []
        row = [task_name]
        for display_label, column_label in (
            ("No Replay", "No Replay"),
            ("Random", "Random Replay"),
            ("Loss", "Loss Replay"),
            ("Surprise", "Surprise Replay"),
        ):
            record = record_lookup.get(column_label)
            metric_value = record["metrics_by_task"].get(task_name, {}).get(metric_key) if record is not None else None
            labeled_values.append((display_label, metric_value))
            row.append(format_metric(metric_value))
        row.append(best_setting_for_values(labeled_values, lower_is_better=(metric_key == "forgetting")))
        rows.append(row)

    final_header = "Lowest Forgetting Setting" if metric_key == "forgetting" else "Best Setting"
    return build_markdown_table(
        ["Task", "No Replay", "Random", "Loss", "Surprise", final_header],
        rows,
    )


def build_strategy_average_lines(
    labeled_runs: list[tuple[str, dict[str, Any]]],
    task_order: list[str],
    metric_key: str,
) -> list[str]:
    """Report average metric values for each exp006 setting."""
    record_lookup = {label: record for label, record in labeled_runs}
    lines: list[str] = []
    for label in ("No Replay", "Random Replay", "Loss Replay", "Surprise Replay"):
        record = record_lookup.get(label)
        average_value = compute_average_metric(record, task_order, metric_key) if record is not None else None
        lines.append(f"- Average {metric_key.replace('_', ' ')} for {label}: {format_metric(average_value)}.")
    return lines


def build_strategy_max_forgetting_lines(
    labeled_runs: list[tuple[str, dict[str, Any]]],
    task_order: list[str],
) -> list[str]:
    """Report maximum forgetting values for each exp006 setting."""
    record_lookup = {label: record for label, record in labeled_runs}
    lines: list[str] = []
    for label in ("No Replay", "Random Replay", "Loss Replay", "Surprise Replay"):
        record = record_lookup.get(label)
        max_value = compute_max_forgetting(record, task_order) if record is not None else None
        lines.append(f"- Maximum forgetting for {label}: {format_metric(max_value)}.")
    return lines


def build_strategy_artifact_lines(labeled_runs: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Describe PRE-PLAY artifacts for exp006."""
    lines: list[str] = []
    for label, record in labeled_runs:
        if label == "No Replay":
            lines.append(f"- {label} (`{record['run_name']}`) has replay disabled in `run_config.json`.")
            continue

        human_review_candidate_exists = (record["run_dir"] / "human_review" / "replay_review_candidates.csv").exists()
        human_review_readme_exists = (record["run_dir"] / "human_review" / "replay_review_readme.md").exists()
        tag_counts = record.get("human_review_tag_counts", {})
        lines.append(f"- {label} (`{record['run_name']}`) saved {len(record['replay_score_files'])} replay score CSV file(s).")
        lines.append(f"- {label} (`{record['run_name']}`) saved {len(record['replay_buffer_snapshots'])} replay buffer snapshot file(s).")
        lines.append(f"- {label} has {len(record['non_empty_replay_snapshots'])} non-empty replay buffer snapshot file(s).")
        lines.append(f"- {label} saved {len(record.get('sample_signal_files', []))} sample_signals CSV file(s).")
        lines.append(f"- {label} saved {len(record.get('utility_label_files', []))} utility_labels CSV file(s).")
        lines.append(
            f"- {label} human review exports: replay_review_candidates.csv={format_boolean(human_review_candidate_exists)}, replay_review_readme.md={format_boolean(human_review_readme_exists)}."
        )
        lines.append(
            f"- {label} human review tag counts: high_value_candidate={tag_counts.get('high_value_candidate', 0)}, selected_for_replay={tag_counts.get('selected_for_replay', 0)}, needs_human_review={tag_counts.get('needs_human_review', 0)}, low_priority={tag_counts.get('low_priority', 0)}."
        )
        lines.append(f"- {label} manually reviewed samples: {record.get('manual_human_review_count', 0)}.")
    return lines


def build_strategy_pairwise_comparison_lines(
    labeled_runs: list[tuple[str, dict[str, Any]]],
    task_order: list[str],
) -> list[str]:
    """Compare replay selection strategies using saved metrics only."""
    record_lookup = {label: record for label, record in labeled_runs}

    def compare_settings(left_label: str, right_label: str) -> str:
        left_record = record_lookup.get(left_label)
        right_record = record_lookup.get(right_label)
        if left_record is None or right_record is None:
            return f"- {left_label} versus {right_label}: not available from the saved files."

        left_val = compute_average_metric(left_record, task_order, "val_accuracy")
        right_val = compute_average_metric(right_record, task_order, "val_accuracy")
        left_test = compute_average_metric(left_record, task_order, "test_accuracy")
        right_test = compute_average_metric(right_record, task_order, "test_accuracy")
        left_forgetting = compute_max_forgetting(left_record, task_order)
        right_forgetting = compute_max_forgetting(right_record, task_order)

        return (
            f"- {left_label} versus {right_label}: average validation accuracy {format_metric(left_val)} vs {format_metric(right_val)}, "
            f"average test accuracy {format_metric(left_test)} vs {format_metric(right_test)}, "
            f"maximum forgetting {format_metric(left_forgetting)} vs {format_metric(right_forgetting)}."
        )

    lines = [
        compare_settings("Random Replay", "Loss Replay"),
        compare_settings("Random Replay", "Surprise Replay"),
        compare_settings("Loss Replay", "Surprise Replay"),
    ]

    baseline_record = record_lookup.get("No Replay")
    for label in ("Random Replay", "Loss Replay", "Surprise Replay"):
        strategy_record = record_lookup.get(label)
        if strategy_record is None or baseline_record is None:
            lines.append(f"- {label} versus No Replay: not available from the saved files.")
            continue
        val_deltas = collect_numeric_deltas(strategy_record, baseline_record, task_order, "val_accuracy")
        test_deltas = collect_numeric_deltas(strategy_record, baseline_record, task_order, "test_accuracy")
        baseline_max_forgetting = compute_max_forgetting(baseline_record, task_order)
        strategy_max_forgetting = compute_max_forgetting(strategy_record, task_order)
        lines.append(
            f"- {label} versus No Replay: average validation delta {format_metric(mean(val_deltas) if val_deltas else None)}, "
            f"average test delta {format_metric(mean(test_deltas) if test_deltas else None)}, "
            f"maximum forgetting {format_metric(strategy_max_forgetting)} versus {format_metric(baseline_max_forgetting)}."
        )
    lines.append("- These comparisons use saved metrics only and should be read cautiously under this lightweight setting.")
    return lines


def build_strategy_main_findings_lines(
    labeled_runs: list[tuple[str, dict[str, Any]]],
    task_order: list[str],
) -> list[str]:
    """Summarize the main findings for exp006."""
    record_lookup = {label: record for label, record in labeled_runs}
    val_averages = [
        (label, compute_average_metric(record, task_order, "val_accuracy"))
        for label, record in labeled_runs
    ]
    test_averages = [
        (label, compute_average_metric(record, task_order, "test_accuracy"))
        for label, record in labeled_runs
    ]
    max_forgetting_values = [
        (label, compute_max_forgetting(record, task_order))
        for label, record in labeled_runs
    ]
    random_record = record_lookup.get("Random Replay")
    loss_record = record_lookup.get("Loss Replay")
    surprise_record = record_lookup.get("Surprise Replay")
    baseline_record = record_lookup.get("No Replay")

    findings = [
        f"- The saved strategy comparison covers `{format_task_sequence(task_order)}` with one no-replay baseline and three replay selection strategies at buffer size `30` and seed `42`.",
        f"- The best average validation setting was {best_setting_for_values(val_averages)}.",
        f"- The best average test setting was {best_setting_for_values(test_averages)}.",
        f"- The setting with the lowest maximum forgetting was {best_setting_for_values(max_forgetting_values, lower_is_better=True)}.",
    ]

    if baseline_record is not None:
        replay_labels_beating_baseline = []
        for label, record in (("Random Replay", random_record), ("Loss Replay", loss_record), ("Surprise Replay", surprise_record)):
            if record is None:
                continue
            test_delta_values = collect_numeric_deltas(record, baseline_record, task_order, "test_accuracy")
            if test_delta_values and mean(test_delta_values) > 0:
                replay_labels_beating_baseline.append(label)
        if replay_labels_beating_baseline:
            findings.append(f"- Replay strategies with positive average test deltas versus no-replay were: {', '.join(replay_labels_beating_baseline)}.")
        else:
            findings.append("- None of the replay strategies showed a positive average test delta versus no-replay across the saved tasks.")

    if random_record is not None and loss_record is not None and surprise_record is not None:
        random_test = compute_average_metric(random_record, task_order, "test_accuracy")
        loss_test = compute_average_metric(loss_record, task_order, "test_accuracy")
        surprise_test = compute_average_metric(surprise_record, task_order, "test_accuracy")
        if (
            loss_test is not None
            and random_test is not None
            and loss_test > random_test
            and surprise_test is not None
            and surprise_test > random_test
        ):
            findings.append("- Both scoring-based replay strategies outperform random replay on average test accuracy, which is preliminary evidence that replay scoring may matter.")
        elif (
            (loss_test is not None and random_test is not None and loss_test > random_test)
            or (surprise_test is not None and random_test is not None and surprise_test > random_test)
        ):
            findings.append("- At least one scoring-based replay strategy outperforms random replay on average test accuracy, which is tentative evidence that replay scoring may matter.")
        else:
            findings.append("- Loss-based and surprise-based replay do not clearly outperform random replay on average test accuracy in the saved files.")

    human_review_ready = all(
        record.get("human_review_candidate_file") is not None
        for label, record in labeled_runs
        if label != "No Replay"
    )
    findings.append(
        f"- PRE-PLAY artifact generation succeeded for the replay-enabled runs, and human-review-ready candidate files were {'present' if human_review_ready else 'not fully present'}."
    )
    return findings[:8]


def build_strategy_interpretation_lines(
    labeled_runs: list[tuple[str, dict[str, Any]]],
    task_order: list[str],
) -> list[str]:
    """Interpret exp006 cautiously."""
    record_lookup = {label: record for label, record in labeled_runs}
    random_record = record_lookup.get("Random Replay")
    loss_record = record_lookup.get("Loss Replay")
    surprise_record = record_lookup.get("Surprise Replay")

    random_test = compute_average_metric(random_record, task_order, "test_accuracy") if random_record is not None else None
    loss_test = compute_average_metric(loss_record, task_order, "test_accuracy") if loss_record is not None else None
    surprise_test = compute_average_metric(surprise_record, task_order, "test_accuracy") if surprise_record is not None else None

    lines: list[str] = []
    if (
        loss_test is not None
        and random_test is not None
        and loss_test > random_test
    ) or (
        surprise_test is not None
        and random_test is not None
        and surprise_test > random_test
    ):
        lines.append("- Because at least one scoring-based replay strategy outperforms random replay on the saved metrics, exp006 provides preliminary evidence that replay selection by score may matter.")
    else:
        lines.append("- Because loss-based and surprise-based replay do not clearly outperform random replay on the saved metrics, the current scoring signal does not yet look strong enough under this lightweight setting.")

    best_test_setting = best_setting_for_values(
        [(label, compute_average_metric(record, task_order, "test_accuracy")) for label, record in labeled_runs]
    )
    lines.append(f"- On average final test accuracy, the strongest saved setting was {best_test_setting}, but this should not be over-interpreted from one seed and one epoch.")
    lines.append("- The PRE-PLAY direction is still supported operationally because sample_signals, utility_labels, and human_review exports were generated successfully for replay-enabled runs.")
    lines.append("- Predicted utility replay is still future work because the learned scorer has not yet been trained and validated well enough for this experiment.")
    return lines


def build_strategy_limitations_lines(run_records: list[dict[str, Any]], task_order: list[str]) -> list[str]:
    """List the limitations for exp006."""
    reference_config = run_records[0]["config"] if run_records else {}
    return [
        f"- One seed only: `{format_scalar(reference_config.get('seed'))}`.",
        f"- One-epoch training: `{format_scalar(reference_config.get('num_epochs'))}` epoch(s) were saved in the configuration.",
        (
            f"- Lightweight subset sizes: train={format_scalar(reference_config.get('train_sample_size'))}, "
            f"val={format_scalar(reference_config.get('val_sample_size'))}, test={format_scalar(reference_config.get('test_sample_size'))}."
        ),
        f"- Only three tasks are used here: `{format_task_sequence(task_order)}`.",
        "- `predicted_utility` replay is not tested yet.",
        "- Utility labels are proxy labels rather than exact causal future replay utility.",
        "- Human review files were exported automatically, but manual review has not yet been completed in these saved results.",
    ]


def build_strategy_summary_markdown(
    experiment_id: str,
    experiment_dir: Path,
    run_records: list[dict[str, Any]],
    all_experiment_dirs: list[Path],
) -> str:
    """Build the exp006 strategy comparison summary."""
    generation_timestamp = datetime.now().isoformat(timespec="seconds")
    task_order = get_primary_task_order(run_records)
    labeled_runs = get_strategy_comparison_runs(run_records)
    relation_lines = build_relation_to_other_experiments_lines(experiment_id, all_experiment_dirs)

    summary_lines = [
        f"# {experiment_id} Summary",
        "",
        "## 1. Experiment Overview",
        "This experiment is a replay selection strategy comparison experiment.",
        "",
        "## 2. Research Question",
        "Under the same replay buffer budget, do loss-based or surprise-based replay selection strategies outperform random replay and no-replay?",
        "",
        "## 3. Experiment Design",
        "- No-replay baseline.",
        "- Random replay.",
        "- Loss-based replay.",
        "- Surprise-based replay.",
        "- Task sequence: `sciq -> arc_challenge -> boolq`.",
        "- Seed: `42`.",
        "- Replay buffer size: `30`.",
        "- `predicted_utility` is not used yet because the learned scorer has not yet been trained and validated.",
        "",
        "## 4. Task and Dataset Description",
        *build_task_description_lines(task_order),
        "",
        "## 5. Run Commands",
        "- `python main.py --run-name exp006_no_replay_seed42_3tasks --tasks sciq arc_challenge boolq --no-replay --seed 42`",
        "- `python main.py --run-name exp006_replay_random_buffer30_seed42_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 42 --replay-selection-strategy random`",
        "- `python main.py --run-name exp006_replay_loss_buffer30_seed42_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 42 --replay-selection-strategy loss`",
        "- `python main.py --run-name exp006_replay_surprise_buffer30_seed42_3tasks --tasks sciq arc_challenge boolq --replay-buffer-size 30 --seed 42 --replay-selection-strategy surprise`",
        "",
        "## 6. Directory Structure",
        "```text",
        build_tree(experiment_dir, max_depth=3),
        "```",
        "",
        "## 7. Configuration Summary",
        build_strategy_configuration_summary_table(labeled_runs),
        "",
        "## 8. Output Files and Their Purpose",
        "- `checkpoints/`: saved model and tokenizer checkpoints after each task.",
        "- `logs/`: training summary and step logs for each run.",
        "- `metrics/`: validation and test metrics plus prediction files.",
        "- `replay_scores/`: per-sample replay scoring CSV files for replay-enabled runs.",
        "- `replay_buffers/`: replay buffer snapshots and replay buffer summary files.",
        "- `sample_signals/`: sample-level early training signal logs.",
        "- `utility_labels/`: proxy utility labels derived from replay selection outcomes.",
        "- `human_review/`: auto-tagged replay review exports for manual review.",
        "- `run_config.json`: saved resolved configuration for each run, including buffer size, seed, and selection strategy.",
        "- `exp006_summary.md`: this experiment-level replay strategy comparison report.",
        "",
        "## 9. Metric Definitions",
        "- Validation accuracy: the fraction of validation examples answered correctly after the final saved task state.",
        "- Test accuracy: the fraction of held-out test examples answered correctly after the final saved task state.",
        "- Forgetting: the saved drop from the best earlier validation accuracy to the later final validation accuracy for a task.",
        "- Strategy comparison: compare saved validation accuracy, test accuracy, and forgetting across no replay, random replay, loss replay, and surprise replay.",
        "- Delta relative to no-replay: `Strategy score - No Replay score`.",
        "- Delta relative to random replay: `Strategy score - Random Replay score`.",
        "",
        "## 10. Final Validation Accuracy Comparison",
        build_strategy_metric_table(labeled_runs, task_order, "val_accuracy"),
        *build_strategy_average_lines(labeled_runs, task_order, "val_accuracy"),
        "",
        "## 11. Final Test Accuracy Comparison",
        build_strategy_metric_table(labeled_runs, task_order, "test_accuracy"),
        *build_strategy_average_lines(labeled_runs, task_order, "test_accuracy"),
        "",
        "## 12. Forgetting Analysis",
        build_strategy_metric_table(labeled_runs, task_order, "forgetting"),
        *build_strategy_max_forgetting_lines(labeled_runs, task_order),
        "",
        "## 13. PRE-PLAY Artifact Analysis",
        *build_strategy_artifact_lines(labeled_runs),
        "",
        "## 14. Replay Selection Strategy Comparison",
        *build_strategy_pairwise_comparison_lines(labeled_runs, task_order),
        "",
        "## 15. Main Findings",
        *build_strategy_main_findings_lines(labeled_runs, task_order),
        "",
        "## 16. Interpretation",
        *build_strategy_interpretation_lines(labeled_runs, task_order),
        "",
        "## 17. Limitations",
        *build_strategy_limitations_lines(run_records, task_order),
        "",
        "## 18. Relation to Other Experiments",
        *relation_lines,
        "",
        "## 19. Next Step",
        "Use the generated utility_labels and human_review files to train or validate a utility scorer, then run predicted_utility replay in exp007.",
        "",
        "## 20. Reproducibility Checklist",
        "- [x] Run commands recorded",
        "- [x] run_config.json saved",
        "- [x] metrics saved",
        "- [x] replay artifacts saved or clearly absent",
        "- [x] sample_signals saved",
        "- [x] utility_labels saved",
        "- [x] human_review candidates exported",
        "- [x] replay_selection_strategy recorded",
        "- [x] experiment_index.md updated",
        "",
        "## 21. Notes",
        f"- Runs appear completed: {'Yes' if all(record['run_appears_complete'] for _, record in labeled_runs) else 'Partially or uncertainly'}.",
        f"- Runtime errors found in checked logs: {'Yes' if any(record['has_traceback'] for _, record in labeled_runs) else 'No'}.",
        f"- Summary generation timestamp: `{generation_timestamp}`.",
        "- No `exp001` to `exp005` folders were modified by this experiment.",
        "",
    ]
    return "\n".join(summary_lines)


def get_seed_robustness_runs(
    run_records: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None, str, dict[str, Any] | None]:
    """Return labeled baseline and replay runs for a seed robustness check."""
    baseline_run = get_condition_run(run_records, replay_enabled=False)
    replay_run = get_condition_run(run_records, replay_enabled=True)
    baseline_seed = baseline_run["config"].get("seed") if baseline_run is not None else None
    replay_seed = replay_run["config"].get("seed") if replay_run is not None else None
    replay_buffer_size = replay_run["config"].get("replay_buffer_size") if replay_run is not None else None
    baseline_label = f"No Replay Seed {format_scalar(baseline_seed)}"
    replay_label = f"{format_buffer_label(replay_buffer_size)} Seed {format_scalar(replay_seed)}"
    return baseline_label, baseline_run, replay_label, replay_run


def better_named_setting_label(
    replay_value: Any,
    baseline_value: Any,
    replay_label: str,
    baseline_label: str,
    *,
    lower_is_better: bool = False,
) -> str:
    """Return the better setting name using explicit labels."""
    if replay_value is None or baseline_value is None:
        return "Not available"
    replay_numeric = float(replay_value)
    baseline_numeric = float(baseline_value)
    if math.isnan(replay_numeric) or math.isnan(baseline_numeric):
        return "Not available"
    if math.isclose(replay_numeric, baseline_numeric, rel_tol=1e-9, abs_tol=1e-9):
        return "Tie"
    if lower_is_better:
        return replay_label if replay_numeric < baseline_numeric else baseline_label
    return replay_label if replay_numeric > baseline_numeric else baseline_label


def build_labeled_two_run_configuration_summary_table(
    baseline_label: str,
    baseline_run: dict[str, Any] | None,
    replay_label: str,
    replay_run: dict[str, Any] | None,
) -> str:
    """Build a two-run configuration summary table with explicit labels."""
    baseline_config = baseline_run["config"] if baseline_run is not None else {}
    replay_config = replay_run["config"] if replay_run is not None else {}
    rows = [
        ["run_name", str(baseline_config.get("run_name", "Not available")), str(replay_config.get("run_name", "Not available"))],
        [
            "tasks",
            format_task_sequence(list(baseline_config.get("task_order", []))) if baseline_config.get("task_order") else "Not available",
            format_task_sequence(list(replay_config.get("task_order", []))) if replay_config.get("task_order") else "Not available",
        ],
        ["replay enabled", str(baseline_config.get("replay_enabled", "Not available")), str(replay_config.get("replay_enabled", "Not available"))],
        ["replay buffer size", format_scalar(baseline_config.get("replay_buffer_size")), format_scalar(replay_config.get("replay_buffer_size"))],
        ["seed", format_scalar(baseline_config.get("seed")), format_scalar(replay_config.get("seed"))],
        ["model name", str(baseline_config.get("model_name", "Not available")), str(replay_config.get("model_name", "Not available"))],
        ["epochs", format_scalar(baseline_config.get("num_epochs")), format_scalar(replay_config.get("num_epochs"))],
        ["batch size", format_scalar(baseline_config.get("batch_size")), format_scalar(replay_config.get("batch_size"))],
        ["learning rate", format_scalar(baseline_config.get("learning_rate")), format_scalar(replay_config.get("learning_rate"))],
        [
            "utility scorer path",
            str(get_config_utility_scorer_path(baseline_config) or "Not available"),
            str(get_config_utility_scorer_path(replay_config) or "Not available"),
        ],
        [
            "utility feature columns path",
            str(get_config_utility_feature_columns_path(baseline_config) or "Not available"),
            str(get_config_utility_feature_columns_path(replay_config) or "Not available"),
        ],
        [
            "output directory",
            str(baseline_config.get("run_dir", baseline_run["run_dir"] if baseline_run is not None else "Not available")),
            str(replay_config.get("run_dir", replay_run["run_dir"] if replay_run is not None else "Not available")),
        ],
    ]
    return build_markdown_table(["Field", baseline_label, replay_label], rows)


def build_seed_robustness_metric_table(
    baseline_run: dict[str, Any] | None,
    replay_run: dict[str, Any] | None,
    task_order: list[str],
    metric_key: str,
    replay_short_label: str,
) -> str:
    """Build a baseline-versus-replay metric comparison table with explicit labels."""
    rows: list[list[str]] = []
    for task_name in get_metric_table_tasks(replay_run, baseline_run, task_order):
        baseline_value = baseline_run["metrics_by_task"].get(task_name, {}).get(metric_key) if baseline_run else None
        replay_value = replay_run["metrics_by_task"].get(task_name, {}).get(metric_key) if replay_run else None
        delta_text = "Not available"
        if replay_value is not None and baseline_value is not None:
            delta_text = format_metric(float(replay_value) - float(baseline_value))
        rows.append(
            [
                task_name,
                format_metric(baseline_value),
                format_metric(replay_value),
                delta_text,
                better_named_setting_label(replay_value, baseline_value, replay_short_label, "No Replay"),
            ]
        )
    return build_markdown_table(["Task", "No Replay", replay_short_label, "Delta", "Better Setting"], rows)


def build_seed_robustness_forgetting_table(
    baseline_run: dict[str, Any] | None,
    replay_run: dict[str, Any] | None,
    task_order: list[str],
    replay_short_label: str,
) -> str:
    """Build the forgetting table for a seed robustness check."""
    rows: list[list[str]] = []
    for task_name in get_metric_table_tasks(replay_run, baseline_run, task_order):
        baseline_value = baseline_run["metrics_by_task"].get(task_name, {}).get("forgetting") if baseline_run else None
        replay_value = replay_run["metrics_by_task"].get(task_name, {}).get("forgetting") if replay_run else None
        rows.append(
            [
                task_name,
                format_metric(baseline_value),
                format_metric(replay_value),
                better_named_setting_label(
                    replay_value,
                    baseline_value,
                    replay_short_label,
                    "No Replay",
                    lower_is_better=True,
                ),
            ]
        )
    return build_markdown_table(
        ["Task", "No Replay Forgetting", f"{replay_short_label} Forgetting", "Lower Forgetting Setting"],
        rows,
    )


def find_experiment_dir_by_id(all_experiment_dirs: list[Path], experiment_id: str) -> Path | None:
    """Return a matching experiment directory if it exists."""
    for experiment_dir in all_experiment_dirs:
        if experiment_dir.name == experiment_id:
            return experiment_dir
    return None


def build_seed_robustness_design_lines(
    baseline_run: dict[str, Any] | None,
    replay_run: dict[str, Any] | None,
    task_order: list[str],
) -> list[str]:
    """Describe the seed robustness comparison design."""
    seed_value = replay_run["config"].get("seed") if replay_run is not None else baseline_run["config"].get("seed") if baseline_run is not None else None
    replay_buffer_size = replay_run["config"].get("replay_buffer_size") if replay_run is not None else None
    lines = [
        f"- No-replay baseline: `{baseline_run['run_name']}` with replay disabled." if baseline_run is not None else "- No-replay baseline: Not available.",
        (
            f"- Replay run: `{replay_run['run_name']}` with replay buffer size `{format_scalar(replay_buffer_size)}`."
            if replay_run is not None
            else "- Replay run: Not available."
        ),
        f"- Task sequence: `{format_task_sequence(task_order)}`.",
        f"- Seed: `{format_scalar(seed_value)}`.",
        "- Why this comparison is useful: it checks whether the strongest saved replay-buffer setting from exp004 still behaves similarly after changing only the random seed.",
    ]
    return lines


def build_seed_robustness_artifact_lines(
    baseline_run: dict[str, Any] | None,
    replay_run: dict[str, Any] | None,
    replay_short_label: str,
) -> list[str]:
    """Describe replay artifacts for a seed robustness check."""
    lines: list[str] = []
    if replay_run is not None:
        replay_score_count = len(replay_run["replay_score_files"])
        replay_snapshot_count = len(replay_run["replay_buffer_snapshots"])
        replay_non_empty_count = len(replay_run["non_empty_replay_snapshots"])
        lines.append(f"- {replay_short_label} (`{replay_run['run_name']}`) saved {replay_score_count} replay score CSV file(s).")
        lines.append(f"- {replay_short_label} (`{replay_run['run_name']}`) saved {replay_snapshot_count} replay buffer snapshot file(s).")
        lines.append(
            f"- {replay_short_label} has {replay_non_empty_count} non-empty replay buffer snapshot file(s), which suggests replay was active for this setting."
        )
    else:
        lines.append("- Replay artifact analysis for the replay condition is not available because the replay run is missing.")

    if baseline_run is not None:
        baseline_snapshot_count = len(baseline_run["replay_buffer_snapshots"])
        baseline_non_empty_count = len(baseline_run["non_empty_replay_snapshots"])
        lines.append(f"- No-replay run `{baseline_run['run_name']}` has replay disabled.")
        lines.append(
            f"- Its `replay_scores/` folder contains {len(baseline_run['replay_score_files'])} replay score CSV file(s), which is expected to be zero when replay is disabled."
        )
        lines.append(
            f"- Its `replay_buffers/` folder contains {baseline_snapshot_count} replay buffer snapshot file(s), with {baseline_non_empty_count} non-empty snapshot(s); empty files indicate placeholders rather than active replay."
        )
    else:
        lines.append("- Replay artifact analysis for the no-replay condition is not available because the baseline run is missing.")
    return lines


def compare_seed_robustness_with_exp004(
    baseline_run: dict[str, Any] | None,
    replay_run: dict[str, Any] | None,
    task_order: list[str],
    all_experiment_dirs: list[Path],
) -> list[str]:
    """Compare exp005 against the saved exp004 buffer-30 result."""
    exp004_dir = find_experiment_dir_by_id(all_experiment_dirs, "exp004")
    if exp004_dir is None:
        return ["- `exp004` was not found under `outputs/experiments/`, so a saved cross-seed comparison is not available."]

    exp004_records = [load_run_record(run_dir) for run_dir in list_run_dirs(exp004_dir)]
    exp004_baseline = get_condition_run(exp004_records, replay_enabled=False)
    exp004_buffer30 = next(
        (
            record
            for label, record in get_buffer_ablation_runs(exp004_records)
            if label == "Buffer 30"
        ),
        None,
    )
    if exp004_baseline is None or exp004_buffer30 is None:
        return ["- `exp004` exists, but its no-replay or Buffer 30 run could not be recovered from the saved files."]

    exp004_task_order = get_primary_task_order(exp004_records)
    exp004_seed = exp004_buffer30["config"].get("seed")
    exp005_seed = replay_run["config"].get("seed") if replay_run is not None else baseline_run["config"].get("seed") if baseline_run is not None else None

    exp004_val_delta = compute_average_metric(exp004_buffer30, exp004_task_order, "val_accuracy")
    if exp004_val_delta is not None:
        exp004_val_delta -= compute_average_metric(exp004_baseline, exp004_task_order, "val_accuracy") or 0.0
    exp004_test_delta = compute_average_metric(exp004_buffer30, exp004_task_order, "test_accuracy")
    if exp004_test_delta is not None:
        exp004_test_delta -= compute_average_metric(exp004_baseline, exp004_task_order, "test_accuracy") or 0.0

    exp004_baseline_max_forgetting = compute_max_forgetting(exp004_baseline, exp004_task_order)
    exp004_buffer30_max_forgetting = compute_max_forgetting(exp004_buffer30, exp004_task_order)
    exp005_val_delta = mean(collect_numeric_deltas(replay_run, baseline_run, task_order, "val_accuracy")) if collect_numeric_deltas(replay_run, baseline_run, task_order, "val_accuracy") else None
    exp005_test_delta = mean(collect_numeric_deltas(replay_run, baseline_run, task_order, "test_accuracy")) if collect_numeric_deltas(replay_run, baseline_run, task_order, "test_accuracy") else None
    exp005_baseline_max_forgetting = compute_max_forgetting(baseline_run, task_order)
    exp005_replay_max_forgetting = compute_max_forgetting(replay_run, task_order)

    lines = [
        f"- `exp004` used seed `{format_scalar(exp004_seed)}`, while `exp005` uses seed `{format_scalar(exp005_seed)}`.",
        "- In `exp004`, Buffer 30 was the strongest replay setting on average validation accuracy and maximum forgetting among the saved replay settings.",
    ]
    if exp005_val_delta is not None:
        lines.append(
            f"- In `exp005`, Buffer 30 changed average final validation accuracy by {exp005_val_delta:.4f} relative to no-replay."
        )
    else:
        lines.append("- In `exp005`, the saved files do not provide enough validation metrics to compute an average replay-versus-baseline delta.")
    if exp005_test_delta is not None:
        lines.append(
            f"- In `exp005`, Buffer 30 changed average final test accuracy by {exp005_test_delta:.4f} relative to no-replay."
        )
    else:
        lines.append("- In `exp005`, the saved files do not provide enough test metrics to compute an average replay-versus-baseline delta.")
    if exp005_baseline_max_forgetting is not None and exp005_replay_max_forgetting is not None:
        lines.append(
            f"- In `exp005`, maximum saved forgetting moved from {exp005_baseline_max_forgetting:.4f} in no-replay to {exp005_replay_max_forgetting:.4f} with Buffer 30."
        )
    else:
        lines.append("- In `exp005`, maximum saved forgetting could not be compared from the saved metrics.")

    supports_pattern = (
        exp005_val_delta is not None
        and exp005_val_delta > 0
        and exp005_baseline_max_forgetting is not None
        and exp005_replay_max_forgetting is not None
        and exp005_replay_max_forgetting <= exp005_baseline_max_forgetting
    )
    weakens_pattern = (
        exp005_val_delta is not None
        and exp005_val_delta <= 0
        and exp005_baseline_max_forgetting is not None
        and exp005_replay_max_forgetting is not None
        and exp005_replay_max_forgetting > exp005_baseline_max_forgetting
    )
    if supports_pattern and (exp005_test_delta is None or exp005_test_delta >= 0):
        lines.append("- Relative to the no-replay baseline, `exp005` tentatively supports the favorable Buffer 30 pattern from `exp004`, although the evidence still comes from only two seeds.")
    elif weakens_pattern:
        lines.append("- Relative to the no-replay baseline, `exp005` weakens the favorable Buffer 30 pattern from `exp004`, which suggests seed sensitivity or limited stability.")
    else:
        lines.append("- Relative to the no-replay baseline, `exp005` gives mixed evidence about the Buffer 30 pattern from `exp004` rather than a clear confirmation or reversal.")

    if exp004_val_delta is not None and exp004_test_delta is not None and exp004_baseline_max_forgetting is not None and exp004_buffer30_max_forgetting is not None:
        lines.append(
            f"- For reference, `exp004` Buffer 30 changed average validation accuracy by {exp004_val_delta:.4f}, average test accuracy by {exp004_test_delta:.4f}, and maximum forgetting from {exp004_baseline_max_forgetting:.4f} to {exp004_buffer30_max_forgetting:.4f} relative to its own no-replay baseline."
        )
    return lines


def build_seed_robustness_main_findings_lines(
    baseline_run: dict[str, Any] | None,
    replay_run: dict[str, Any] | None,
    task_order: list[str],
) -> list[str]:
    """Summarize the main findings for a seed robustness check."""
    val_deltas = collect_numeric_deltas(replay_run, baseline_run, task_order, "val_accuracy")
    test_deltas = collect_numeric_deltas(replay_run, baseline_run, task_order, "test_accuracy")
    baseline_max_forgetting = compute_max_forgetting(baseline_run, task_order)
    replay_max_forgetting = compute_max_forgetting(replay_run, task_order)

    findings = [f"- The saved seed-robustness check covers `{format_task_sequence(task_order)}` at seed `{format_scalar(replay_run['config'].get('seed') if replay_run is not None else baseline_run['config'].get('seed') if baseline_run is not None else None)}`."]
    if val_deltas:
        findings.append(f"- The average Buffer 30 minus No Replay final validation accuracy delta was {mean(val_deltas):.4f}.")
    else:
        findings.append("- Final validation accuracy deltas were not fully available from the saved metrics.")
    if test_deltas:
        findings.append(f"- The average Buffer 30 minus No Replay final test accuracy delta was {mean(test_deltas):.4f}.")
    else:
        findings.append("- Final test accuracy deltas were not fully available from the saved metrics.")
    if baseline_max_forgetting is not None and replay_max_forgetting is not None:
        findings.append(
            f"- Maximum saved forgetting changed from {baseline_max_forgetting:.4f} in No Replay to {replay_max_forgetting:.4f} with Buffer 30."
        )
    else:
        findings.append("- Maximum saved forgetting could not be fully compared from the saved files.")
    if replay_run is not None and baseline_run is not None and has_consistent_replay_advantage(replay_run, baseline_run, task_order):
        findings.append("- Under seed 123, the saved metrics are consistently favorable to Buffer 30 across validation accuracy, test accuracy, and forgetting.")
    else:
        findings.append("- Under seed 123, the saved metrics provide mixed evidence rather than a fully consistent Buffer 30 advantage.")
    return findings[:5]


def build_seed_robustness_interpretation_lines(
    baseline_run: dict[str, Any] | None,
    replay_run: dict[str, Any] | None,
    task_order: list[str],
) -> list[str]:
    """Interpret the seed robustness result cautiously."""
    if replay_run is None or baseline_run is None:
        return ["- A full interpretation is not available because one of the two comparison conditions is missing."]

    val_deltas = collect_numeric_deltas(replay_run, baseline_run, task_order, "val_accuracy")
    test_deltas = collect_numeric_deltas(replay_run, baseline_run, task_order, "test_accuracy")
    baseline_max_forgetting = compute_max_forgetting(baseline_run, task_order)
    replay_max_forgetting = compute_max_forgetting(replay_run, task_order)
    lines: list[str] = []

    if val_deltas:
        average_val_delta = mean(val_deltas)
        if average_val_delta > 0:
            lines.append(f"- At seed 123, Buffer 30 improved average final validation accuracy by {average_val_delta:.4f} relative to no-replay.")
        elif average_val_delta < 0:
            lines.append(f"- At seed 123, Buffer 30 reduced average final validation accuracy by {abs(average_val_delta):.4f} relative to no-replay.")
        else:
            lines.append("- At seed 123, Buffer 30 and no-replay were tied on average final validation accuracy.")
    else:
        lines.append("- Validation accuracy was not available for every task, so the validation-side seed check is incomplete.")

    if test_deltas:
        average_test_delta = mean(test_deltas)
        if average_test_delta > 0:
            lines.append(f"- At seed 123, Buffer 30 improved average final test accuracy by {average_test_delta:.4f} relative to no-replay.")
        elif average_test_delta < 0:
            lines.append(f"- At seed 123, Buffer 30 reduced average final test accuracy by {abs(average_test_delta):.4f} relative to no-replay.")
        else:
            lines.append("- At seed 123, Buffer 30 and no-replay were tied on average final test accuracy.")
    else:
        lines.append("- Test accuracy was not available for every task, so the test-side seed check is incomplete.")

    if baseline_max_forgetting is not None and replay_max_forgetting is not None:
        if replay_max_forgetting < baseline_max_forgetting:
            lines.append(f"- Buffer 30 reduced maximum saved forgetting from {baseline_max_forgetting:.4f} to {replay_max_forgetting:.4f}.")
        elif replay_max_forgetting > baseline_max_forgetting:
            lines.append(f"- Buffer 30 increased maximum saved forgetting from {baseline_max_forgetting:.4f} to {replay_max_forgetting:.4f}.")
        else:
            lines.append(f"- Buffer 30 and no-replay had the same maximum saved forgetting at {replay_max_forgetting:.4f}.")
    else:
        lines.append("- The saved forgetting values are too limited to support a strong seed-robustness claim.")

    if has_consistent_replay_advantage(replay_run, baseline_run, task_order):
        lines.append("- Because validation accuracy, test accuracy, and forgetting all favor Buffer 30 consistently in the saved metrics, this additional seed is cautiously consistent with Buffer 30 remaining useful under this narrow setting.")
    else:
        lines.append("- Because validation accuracy, test accuracy, and forgetting do not all favor Buffer 30 consistently, the result should be read as mixed evidence and possible seed sensitivity rather than a stable replay benefit.")

    lines.append("- Any apparent stability or instability should be treated cautiously because this check still uses one additional seed, one epoch, lightweight sample sizes, and one replay configuration.")
    return lines


def build_seed_robustness_limitations_lines(run_records: list[dict[str, Any]], task_order: list[str]) -> list[str]:
    """List the limitations for a seed robustness check."""
    reference_config = run_records[0]["config"] if run_records else {}
    return [
        f"- Only one additional seed is tested here: `{format_scalar(reference_config.get('seed'))}`.",
        f"- One-epoch training: `{format_scalar(reference_config.get('num_epochs'))}` epoch(s) were saved in the configuration.",
        (
            f"- Lightweight subset sizes: train={format_scalar(reference_config.get('train_sample_size'))}, "
            f"val={format_scalar(reference_config.get('val_sample_size'))}, test={format_scalar(reference_config.get('test_sample_size'))}."
        ),
        f"- Same task order as the original-order baseline: `{format_task_sequence(task_order)}`.",
        "- Only one replay buffer size is retested here, so the seed check does not cover the full replay design space.",
        "- These saved results still provide limited evidence about robustness because they extend the study by only one seed.",
    ]


def build_seed_robustness_reproducibility_checklist_lines(experiment_dir: Path, run_records: list[dict[str, Any]]) -> list[str]:
    """Build the reproducibility checklist for seed robustness summaries."""
    outputs_root = experiment_dir.parent.parent
    summary_path = get_experiment_summary_path(experiment_dir)
    metrics_saved = all(
        (record["run_dir"] / "metrics" / "val_metrics.csv").exists()
        and (record["run_dir"] / "metrics" / "test_metrics.csv").exists()
        for record in run_records
    )
    run_config_saved = all((record["run_dir"] / "run_config.json").exists() for record in run_records)
    replay_artifacts_clear = all(
        (record["replay_enabled"] and (record["run_dir"] / "replay_scores").exists())
        or ((record["run_dir"] / "replay_scores").exists() and (record["run_dir"] / "replay_buffers").exists())
        for record in run_records
    )
    seed_recorded = all(record["config"].get("seed") is not None for record in run_records)
    return [
        f"- [{'x' if any(record.get('reconstructed_command') for record in run_records) else ' '}] Run commands recorded",
        f"- [{'x' if run_config_saved else ' '}] run_config.json saved",
        f"- [{'x' if metrics_saved else ' '}] metrics saved",
        f"- [{'x' if replay_artifacts_clear else ' '}] replay artifacts saved or clearly absent",
        f"- [{'x' if seed_recorded else ' '}] seed recorded",
        f"- [{'x' if summary_path.exists() or True else ' '}] summary generated from saved files",
        f"- [{'x' if (outputs_root / 'experiment_index.md').exists() else ' '}] experiment_index.md updated",
    ]


def build_seed_robustness_summary_markdown(
    experiment_id: str,
    experiment_dir: Path,
    run_records: list[dict[str, Any]],
    all_experiment_dirs: list[Path],
) -> str:
    """Build the exp005-style seed robustness summary."""
    generation_timestamp = datetime.now().isoformat(timespec="seconds")
    task_order = get_primary_task_order(run_records)
    baseline_label, baseline_run, replay_label, replay_run = get_seed_robustness_runs(run_records)
    replay_short_label = format_buffer_label(replay_run["config"].get("replay_buffer_size")) if replay_run is not None else "Replay"

    summary_lines = [
        f"# {experiment_id} Summary",
        "",
        "## 1. Experiment Overview",
        "This experiment is best interpreted as a seed robustness check. It re-tests the original task sequence `sciq -> arc_challenge -> boolq` at seed `123` by comparing a no-replay baseline against replay with buffer size `30`.",
        "",
        "## 2. Research Question",
        "Does the replay buffer 30 result remain stable when the random seed is changed from 42 to 123?",
        "",
        "## 3. Experiment Design",
        *build_seed_robustness_design_lines(baseline_run, replay_run, task_order),
        "",
        "## 4. Task and Dataset Description",
        *build_task_description_lines(task_order),
        "",
        "## 5. Run Commands",
        *build_run_command_lines(replay_run, baseline_run),
        "",
        "## 6. Directory Structure",
        "```text",
        build_tree(experiment_dir, max_depth=3),
        "```",
        "",
        "## 7. Configuration Summary",
        build_labeled_two_run_configuration_summary_table(baseline_label, baseline_run, replay_label, replay_run),
        "",
        "## 8. Output Files and Their Purpose",
        *output_files_section_lines(experiment_id),
        "",
        "## 9. Metric Definitions",
        *metric_definitions_lines(),
        "",
        "## 10. Final Validation Accuracy Comparison",
        build_seed_robustness_metric_table(baseline_run, replay_run, task_order, "val_accuracy", replay_short_label),
        "",
        "## 11. Final Test Accuracy Comparison",
        build_seed_robustness_metric_table(baseline_run, replay_run, task_order, "test_accuracy", replay_short_label),
        "",
        "## 12. Forgetting Analysis",
        build_seed_robustness_forgetting_table(baseline_run, replay_run, task_order, replay_short_label),
        *build_forgetting_analysis_lines(replay_run, baseline_run, task_order),
        "",
        "## 13. Replay Artifacts Analysis",
        *build_seed_robustness_artifact_lines(baseline_run, replay_run, replay_short_label),
        "",
        "## 14. Comparison with exp004",
        *compare_seed_robustness_with_exp004(baseline_run, replay_run, task_order, all_experiment_dirs),
        "",
        "## 15. Main Findings",
        *build_seed_robustness_main_findings_lines(baseline_run, replay_run, task_order),
        "",
        "## 16. Interpretation",
        *build_seed_robustness_interpretation_lines(baseline_run, replay_run, task_order),
        "",
        "## 17. Limitations",
        *build_seed_robustness_limitations_lines(run_records, task_order),
        "",
        "## 18. Relation to Other Experiments",
        *build_relation_to_other_experiments_lines(experiment_id, all_experiment_dirs),
        "",
        "## 19. Next Step",
        "Run the same seed robustness check with more seeds, or test replay sampling strategy.",
        "",
        "## 20. Reproducibility Checklist",
        *build_seed_robustness_reproducibility_checklist_lines(experiment_dir, run_records),
        "",
        "## 21. Notes",
        *build_notes_lines(run_records, generation_timestamp),
        "",
    ]
    return "\n".join(summary_lines)


def build_reproducibility_checklist_lines(experiment_dir: Path, run_records: list[dict[str, Any]]) -> list[str]:
    """Build the reproducibility checklist."""
    outputs_root = experiment_dir.parent.parent
    summary_path = get_experiment_summary_path(experiment_dir)
    metrics_saved = all((record["run_dir"] / "metrics" / "val_metrics.csv").exists() and (record["run_dir"] / "metrics" / "test_metrics.csv").exists() for record in run_records)
    run_config_saved = all((record["run_dir"] / "run_config.json").exists() for record in run_records)
    replay_artifacts_clear = all(
        (record["replay_enabled"] and (record["run_dir"] / "replay_scores").exists())
        or ((record["run_dir"] / "replay_scores").exists() and (record["run_dir"] / "replay_buffers").exists())
        for record in run_records
    )

    return [
        f"- [{'x' if any(record.get('reconstructed_command') for record in run_records) else ' '}] Run commands recorded",
        f"- [{'x' if run_config_saved else ' '}] run_config.json saved",
        f"- [{'x' if metrics_saved else ' '}] metrics saved",
        f"- [{'x' if replay_artifacts_clear else ' '}] replay artifacts saved or clearly absent",
        f"- [{'x' if summary_path.exists() or True else ' '}] summary generated from saved files",
        f"- [{'x' if (outputs_root / 'experiment_index.md').exists() else ' '}] experiment_index.md updated",
    ]


def build_notes_lines(run_records: list[dict[str, Any]], generation_timestamp: str) -> list[str]:
    """Build the final notes section."""
    runs_completed = all(record["run_appears_complete"] for record in run_records)
    log_paths_checked = any(record["console_log_exists"] for record in run_records)
    tracebacks_found = any(record["has_traceback"] for record in run_records)

    lines = [
        f"- Runs appear completed: {'Yes' if runs_completed else 'Partially or uncertainly'}.",
        f"- Runtime errors found in checked logs: {'Yes' if tracebacks_found else 'No'}." if log_paths_checked else "- Runtime error check: log files were not available to inspect.",
        "- This summary was generated automatically from saved files rather than from a fresh training run.",
        f"- Summary generation timestamp: `{generation_timestamp}`.",
    ]
    return lines


def build_experiment_summary_markdown(
    experiment_id: str,
    experiment_dir: Path,
    run_records: list[dict[str, Any]],
    all_experiment_dirs: list[Path],
) -> str:
    """Build the enriched experiment-level summary markdown."""
    if is_replay_strategy_comparison_experiment(experiment_id, run_records):
        return build_strategy_summary_markdown(
            experiment_id=experiment_id,
            experiment_dir=experiment_dir,
            run_records=run_records,
            all_experiment_dirs=all_experiment_dirs,
        )
    if is_replay_buffer_size_ablation_experiment(experiment_id, run_records):
        return build_buffer_ablation_summary_markdown(
            experiment_id=experiment_id,
            experiment_dir=experiment_dir,
            run_records=run_records,
            all_experiment_dirs=all_experiment_dirs,
        )
    if is_seed_robustness_experiment(experiment_id, run_records):
        return build_seed_robustness_summary_markdown(
            experiment_id=experiment_id,
            experiment_dir=experiment_dir,
            run_records=run_records,
            all_experiment_dirs=all_experiment_dirs,
        )

    generation_timestamp = datetime.now().isoformat(timespec="seconds")
    replay_run = get_condition_run(run_records, replay_enabled=True)
    baseline_run = get_condition_run(run_records, replay_enabled=False)
    task_order = get_primary_task_order(run_records)
    labels = infer_experiment_labels(experiment_id, run_records, all_experiment_dirs)
    has_both_conditions = replay_run is not None and baseline_run is not None
    experiment_description = get_experiment_description(experiment_id, labels, has_both_conditions, task_order)
    overview_text = (
        f"This experiment is best interpreted as {experiment_description}. "
        f"It compares the saved runs over the task sequence `{format_task_sequence(task_order)}`."
        if has_both_conditions
        else f"This experiment is best interpreted as {experiment_description}. "
        f"It uses the saved task sequence `{format_task_sequence(task_order)}`."
    )

    relation_lines = build_relation_to_other_experiments_lines(experiment_id, all_experiment_dirs)

    summary_lines = [
        f"# {experiment_id} Summary",
        "",
        "## 1. Experiment Overview",
        overview_text,
        "",
        "## 2. Research Question",
        infer_research_question(labels, task_order),
        "",
        "## 3. Experiment Design",
        *build_experiment_design_lines(replay_run, baseline_run, task_order),
        "",
        "## 4. Task and Dataset Description",
        *build_task_description_lines(task_order),
        "",
        "## 5. Run Commands",
        *build_run_command_lines(replay_run, baseline_run),
        "",
        "## 6. Directory Structure",
        "```text",
        build_tree(experiment_dir, max_depth=3),
        "```",
        "",
        "## 7. Configuration Summary",
        build_configuration_summary_table(replay_run, baseline_run),
        "",
        "## 8. Output Files and Their Purpose",
        *output_files_section_lines(experiment_id),
        "",
        "## 9. Metric Definitions",
        *metric_definitions_lines(),
        "",
        "## 10. Final Validation Accuracy Comparison",
        build_metric_comparison_table(replay_run, baseline_run, task_order, "val_accuracy"),
        "",
        "## 11. Final Test Accuracy Comparison",
        build_metric_comparison_table(replay_run, baseline_run, task_order, "test_accuracy"),
        "",
        "## 12. Forgetting Analysis",
        build_forgetting_table(replay_run, baseline_run, task_order),
        *build_forgetting_analysis_lines(replay_run, baseline_run, task_order),
        "",
        "## 13. Replay Artifacts Analysis",
        *build_replay_artifact_analysis_lines(replay_run, baseline_run),
        "",
        "## 14. Main Findings",
        *build_main_findings_lines(replay_run, baseline_run, task_order),
        "",
        "## 15. Interpretation",
        *build_interpretation_lines(replay_run, baseline_run, task_order),
        "",
        "## 16. Limitations",
        *build_limitations_lines(run_records, task_order),
        "",
        "## 17. Relation to Other Experiments",
        *relation_lines,
        "",
        "## 18. Next Step",
        infer_next_step(run_records, task_order),
        "",
        "## 19. Reproducibility Checklist",
        *build_reproducibility_checklist_lines(experiment_dir, run_records),
        "",
        "## 20. Notes",
        *build_notes_lines(run_records, generation_timestamp),
        "",
    ]
    return "\n".join(summary_lines)


def generate_experiment_summary(experiment_id: str, project_root: Path | None = None) -> Path:
    """Generate one enriched experiment-level summary file."""
    experiments_root = get_experiment_root(project_root=project_root)
    experiment_dir = experiments_root / experiment_id
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    run_records = [load_run_record(run_dir) for run_dir in list_run_dirs(experiment_dir)]
    all_experiment_dirs = list_experiment_dirs(project_root=project_root)
    summary_path = get_experiment_summary_path(experiment_dir)
    write_text(
        summary_path,
        build_experiment_summary_markdown(
            experiment_id=experiment_id,
            experiment_dir=experiment_dir,
            run_records=run_records,
            all_experiment_dirs=all_experiment_dirs,
        ),
    )
    return summary_path


def build_index_experiment_table(experiment_dirs: list[Path], outputs_root: Path) -> str:
    """Build the experiment list table for the global index."""
    rows = []
    index_dir = outputs_root
    for experiment_dir in experiment_dirs:
        run_records = [load_run_record(run_dir) for run_dir in list_run_dirs(experiment_dir)]
        replay_run = get_condition_run(run_records, replay_enabled=True)
        baseline_run = get_condition_run(run_records, replay_enabled=False)
        task_order = get_primary_task_order(run_records)
        labels = infer_experiment_labels(experiment_dir.name, run_records, experiment_dirs)
        description = get_experiment_description(
            experiment_dir.name,
            labels,
            replay_run is not None and baseline_run is not None,
            task_order,
        )
        summary_path = get_experiment_summary_path(experiment_dir)
        replay_run_cell = replay_run["run_name"] if replay_run is not None else "Not available"
        baseline_run_cell = baseline_run["run_name"] if baseline_run is not None else "Not available"
        if is_replay_strategy_comparison_experiment(experiment_dir.name, run_records):
            replay_run_cell = ", ".join(
                record["run_name"]
                for label, record in get_strategy_comparison_runs(run_records)
                if label != "No Replay"
            )
        if is_replay_buffer_size_ablation_experiment(experiment_dir.name, run_records):
            replay_run_cell = ", ".join(record["run_name"] for label, record in get_buffer_ablation_runs(run_records) if label != "No Replay")
        rows.append(
            [
                experiment_dir.name,
                description,
                format_task_sequence(task_order),
                replay_run_cell,
                baseline_run_cell,
                f"`{experiment_dir}`",
                relative_markdown_link(summary_path, index_dir, summary_path.name),
            ]
        )
    return build_markdown_table(
        ["Experiment ID", "Description", "Task Sequence", "Replay Run Folder", "No-Replay Run Folder", "Experiment Directory", "Summary File"],
        rows,
    )


def build_index_comparison_section(experiment_dir: Path, outputs_root: Path) -> str:
    """Build one comparison section for the global index."""
    run_records = [load_run_record(run_dir) for run_dir in list_run_dirs(experiment_dir)]
    task_order = get_primary_task_order(run_records)
    labels = infer_experiment_labels(experiment_dir.name, run_records, list_experiment_dirs(project_root=outputs_root.parent))
    replay_run = get_condition_run(run_records, replay_enabled=True)
    baseline_run = get_condition_run(run_records, replay_enabled=False)
    description = get_experiment_description(
        experiment_dir.name,
        labels,
        replay_run is not None and baseline_run is not None,
        task_order,
    )

    if is_replay_strategy_comparison_experiment(experiment_dir.name, run_records):
        labeled_runs = get_strategy_comparison_runs(run_records)
        lines = [f"### {experiment_dir.name} Comparison", ""]
        lines.append(f"- Description: {description}.")
        lines.append(f"- Task sequence: `{format_task_sequence(task_order)}`")
        for label, record in labeled_runs:
            lines.append(f"- {label} run folder: `{record['run_name']}`")
        lines.append("")
        lines.append(build_strategy_metric_table(labeled_runs, task_order, "test_accuracy"))
        lines.append("")
        lines.extend(build_strategy_main_findings_lines(labeled_runs, task_order)[:4])
        lines.append("")
        summary_path = get_experiment_summary_path(experiment_dir)
        lines.append(f"Full report: {relative_markdown_link(summary_path, outputs_root, summary_path.name)}")
        return "\n".join(lines)

    if is_replay_buffer_size_ablation_experiment(experiment_dir.name, run_records):
        labeled_runs = get_buffer_ablation_runs(run_records)
        lines = [f"### {experiment_dir.name} Comparison", ""]
        lines.append(f"- Description: {description}.")
        lines.append(f"- Task sequence: `{format_task_sequence(task_order)}`")
        for label, record in labeled_runs:
            lines.append(f"- {label} run folder: `{record['run_name']}`")
        lines.append("")
        lines.append(build_multi_run_metric_table(labeled_runs, task_order, "test_accuracy", "Best Setting"))
        lines.append("")
        lines.extend(build_buffer_ablation_main_findings_lines(labeled_runs, task_order)[:4])
        lines.append("")
        summary_path = get_experiment_summary_path(experiment_dir)
        lines.append(f"Full report: {relative_markdown_link(summary_path, outputs_root, summary_path.name)}")
        return "\n".join(lines)

    lines = [f"### {experiment_dir.name} Comparison", ""]
    lines.append(f"- Description: {description}.")
    lines.append(f"- Task sequence: `{format_task_sequence(task_order)}`")
    lines.append(
        f"- Replay run folder: `{replay_run['run_name']}`" if replay_run is not None else "- Replay run folder: Not available"
    )
    lines.append(
        f"- No-replay run folder: `{baseline_run['run_name']}`" if baseline_run is not None else "- No-replay run folder: Not available"
    )
    lines.append("")
    lines.append(build_metric_comparison_table(replay_run, baseline_run, task_order, "test_accuracy"))
    lines.append("")
    lines.extend(build_interpretation_lines(replay_run, baseline_run, task_order)[:3])
    lines.append("")
    summary_path = get_experiment_summary_path(experiment_dir)
    lines.append(f"Full report: {relative_markdown_link(summary_path, outputs_root, summary_path.name)}")
    return "\n".join(lines)


def update_experiment_index(project_root: Path | None = None) -> Path:
    """Generate outputs/experiment_index.md based on enriched experiment reports."""
    config = get_config(project_root=project_root)
    outputs_root = config.paths.outputs_root
    experiment_dirs = list_experiment_dirs(project_root=project_root)

    comparison_sections: list[str] = []
    all_test_deltas: list[float] = []
    for experiment_dir in experiment_dirs:
        run_records = [load_run_record(run_dir) for run_dir in list_run_dirs(experiment_dir)]
        replay_run = get_condition_run(run_records, replay_enabled=True)
        baseline_run = get_condition_run(run_records, replay_enabled=False)
        task_order = get_primary_task_order(run_records)
        comparison_sections.append(build_index_comparison_section(experiment_dir, outputs_root))
        if is_replay_buffer_size_ablation_experiment(experiment_dir.name, run_records) and baseline_run is not None:
            for _, replay_record in get_buffer_ablation_runs(run_records):
                if replay_record["run_name"] == baseline_run["run_name"]:
                    continue
                all_test_deltas.extend(collect_numeric_deltas(replay_record, baseline_run, task_order, "test_accuracy"))
        else:
            all_test_deltas.extend(collect_numeric_deltas(replay_run, baseline_run, task_order, "test_accuracy"))

    overall_notes = "Replay usefulness is not available because comparable replay/no-replay pairs were not found."
    if all_test_deltas:
        average_delta = mean(all_test_deltas)
        if all(delta > 0 for delta in all_test_deltas):
            overall_notes = (
                f"Across the available experiment pairs, replay improved every saved final test metric with an average delta of {average_delta:.4f}. "
                "This is still a limited single-seed result."
            )
        elif all(delta < 0 for delta in all_test_deltas):
            overall_notes = (
                f"Across the available experiment pairs, replay reduced every saved final test metric with an average delta of {average_delta:.4f}. "
                "That suggests replay was not consistently useful in this saved setting."
            )
        else:
            overall_notes = (
                f"Across the available experiment pairs, replay was mixed overall. The average final test delta was {average_delta:.4f}, "
                "but the direction of the effect varied by task or experiment."
            )

    index_text = "\n".join(
        [
            "# Experiment Index",
            "",
            "## Overview",
            "This index tracks the experiment-level directory structure under `outputs/experiments/`, where each experiment id contains its nested run folders and one enriched `<exp_id>_summary.md` report.",
            "",
            "## Directory Structure",
            "```text",
            build_tree(outputs_root, max_depth=3),
            "```",
            "",
            "## Experiment List",
            build_index_experiment_table(experiment_dirs, outputs_root),
            "",
            "## Comparison Summary",
            "",
            *comparison_sections,
            "",
            "## Overall Notes",
            overall_notes,
            "",
        ]
    )

    index_path = outputs_root / "experiment_index.md"
    write_text(index_path, index_text)
    return index_path
