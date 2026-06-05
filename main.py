from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from config import AppConfig, create_run_paths, get_config
from data_loader import load_task_samples
from evaluator import evaluate_task
from experiment_reports import generate_experiment_summary, update_experiment_index
from model_utils import load_model_and_tokenizer, save_model_and_tokenizer, set_global_seed
from replay_buffer import ReplayBuffer
from sample_signals import enrich_and_filter_sample_signals, write_sample_signals
from scorer import score_samples
from selection_strategies import (
    REPLAY_SELECTION_STRATEGIES,
    attach_replay_selection_scores,
    resolve_predicted_utility_artifact_paths,
    select_samples_for_replay,
    write_replay_selection_comparison,
)
from trainer import train_task
from utility_labels import export_human_review_candidates, generate_proxy_utility_labels


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the PRE-PLAY continual learning experiment.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name used under outputs/experiments/.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional subset of tasks to run in sequence.",
    )
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="Disable replay for a baseline run.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing processed <task>/{train,val,test}.jsonl files.",
    )
    parser.add_argument(
        "--use-audited-review",
        action="store_true",
        help=(
            "Attach the final audited review CSV to this run for replay/utility review "
            "provenance. Train/val/test still come from --data-dir."
        ),
    )
    parser.add_argument(
        "--audited-review-csv",
        type=str,
        default=None,
        help="Path to the final audited replay review CSV.",
    )
    parser.add_argument(
        "--dry-run-config",
        action="store_true",
        help="Validate data inputs, write run_config.json, then exit before model loading.",
    )
    parser.add_argument(
        "--replay-mode",
        type=str,
        default=None,
        choices=("scored", "random", "none"),
        help=(
            "Replay mode: scored uses the configured scoring strategy, random samples "
            "directly from the current task train split, none disables replay."
        ),
    )
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=None,
        help="Optional override for the replay buffer max size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the experiment random seed.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for the number of training epochs.",
    )
    parser.add_argument(
        "--replay-selection-strategy",
        type=str,
        default=None,
        choices=REPLAY_SELECTION_STRATEGIES,
        help="Replay candidate selection strategy. Default: surprise.",
    )
    parser.add_argument(
        "--utility-scorer-path",
        type=str,
        default=None,
        help="Optional explicit scorer model path for predicted_utility replay.",
    )
    parser.add_argument(
        "--utility-feature-columns-path",
        type=str,
        default=None,
        help="Optional explicit feature_columns.json path for predicted_utility replay.",
    )
    return parser.parse_args()


def resolve_project_path(config: AppConfig, path_text: str) -> Path:
    """Resolve a CLI path relative to the project root when it is not absolute."""
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return config.paths.project_root / path


def ensure_local_processed_data(config: AppConfig) -> None:
    """Check that all processed JSONL files already exist locally."""
    missing_paths: list[Path] = []
    for task_name in config.data.task_order:
        task_dir = config.data.data_dir / task_name
        for split_name in ("train", "val", "test"):
            input_path = task_dir / f"{split_name}.jsonl"
            if not input_path.exists():
                missing_paths.append(input_path)

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Processed local data is missing.\n"
            "Run 'python data_download.py' first, then 'python preprocess.py'.\n"
            f"Missing files:\n{missing_text}"
        )


def inspect_audited_review_csv(input_path: Path) -> dict[str, Any]:
    """Return lightweight metadata for an audited replay review CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Audited review CSV not found: {input_path}")

    frame = pd.read_csv(input_path)
    split_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    if "split" in frame.columns:
        split_counts = {
            str(split_name): int(count)
            for split_name, count in frame["split"].value_counts(dropna=False).items()
        }
    if "task_name" in frame.columns:
        task_counts = {
            str(task_name): int(count)
            for task_name, count in frame["task_name"].value_counts(dropna=False).items()
        }

    required_benchmark_splits = {"train", "val", "test"}
    present_splits = set(split_counts)
    contains_full_train_val_test = required_benchmark_splits.issubset(present_splits)
    intended_use = (
        "complete_train_val_test_source"
        if contains_full_train_val_test
        else "audited_replay_review_source_only"
    )
    return {
        "path": str(input_path),
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "split_counts": split_counts,
        "task_counts": task_counts,
        "contains_full_train_val_test": contains_full_train_val_test,
        "intended_use": intended_use,
    }


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def write_text(output_path: Path, text: str) -> None:
    """Write a text file to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def format_project_relative_path(config: AppConfig, input_path: Path) -> str:
    """Format a path relative to the project root when possible."""
    try:
        return str(input_path.relative_to(config.paths.project_root))
    except ValueError:
        return str(input_path)


def build_config_payload(
    config: AppConfig,
    run_paths,
    replay_enabled: bool,
    predicted_utility_artifact_paths: tuple[Path, Path] | None = None,
    audited_review_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable snapshot of the run configuration."""
    predicted_model_path = None
    predicted_feature_columns_path = None
    predicted_source_experiment_dir = None
    if predicted_utility_artifact_paths is not None:
        predicted_model_path = str(predicted_utility_artifact_paths[0])
        predicted_feature_columns_path = str(predicted_utility_artifact_paths[1])
        predicted_source_experiment_dir = str(predicted_utility_artifact_paths[0].parent.parent)

    return {
        "experiment_id": run_paths.experiment_id,
        "run_name": run_paths.run_name,
        "timestamp": run_paths.timestamp,
        "replay_enabled": replay_enabled,
        "replay_mode": config.replay.replay_mode,
        "task_order": list(config.data.task_order),
        "data_dir": format_project_relative_path(config, config.data.data_dir),
        "data_source": config.data.data_source,
        "use_audited_review": config.data.use_audited_review,
        "audited_review_csv": (
            format_project_relative_path(config, config.data.audited_review_csv)
            if config.data.use_audited_review and config.data.audited_review_csv is not None
            else None
        ),
        "train_eval_source_note": config.data.train_eval_source_note,
        "audited_review_source_note": config.data.audited_review_source_note,
        "train_source": config.data.train_eval_source_note,
        "eval_source": config.data.train_eval_source_note,
        "audited_review_source": config.data.audited_review_source_note,
        "audited_review_metadata": audited_review_metadata,
        "model_name": config.model.model_name,
        "seed": config.training.seed,
        "train_sample_size": config.data.train_sample_size,
        "val_sample_size": config.data.val_sample_size,
        "test_sample_size": config.data.test_sample_size,
        "num_epochs": config.training.num_epochs,
        "batch_size": config.training.batch_size,
        "learning_rate": config.training.learning_rate,
        "max_input_length": config.model.max_input_length,
        "max_target_length": config.model.max_target_length,
        "generation_max_new_tokens": config.model.generation_max_new_tokens,
        "replay_buffer_size": config.replay.replay_buffer_size,
        "top_k_per_task": config.replay.top_k_per_task,
        "replay_ratio": config.replay.replay_ratio,
        "replay_selection_strategy": config.replay.replay_selection_strategy,
        "output_root": str(run_paths.output_root),
        "experiment_dir": str(run_paths.experiment_dir),
        "run_dir": str(run_paths.run_dir),
        "checkpoints_dir": str(run_paths.checkpoint_dir),
        "logs_dir": str(run_paths.log_dir),
        "metrics_dir": str(run_paths.metrics_dir),
        "replay_scores_dir": str(run_paths.replay_scores_dir),
        "replay_buffers_dir": str(run_paths.replay_buffers_dir),
        "sample_signals_dir": str(run_paths.sample_signals_dir),
        "utility_labels_dir": str(run_paths.utility_labels_dir),
        "human_review_dir": str(run_paths.human_review_dir),
        "replay_selection_comparison_dir": str(run_paths.replay_selection_comparison_dir),
        "scorer_outputs_dir": str(run_paths.scorer_outputs_dir),
        "scorer_checkpoints_dir": str(run_paths.scorer_checkpoints_dir),
        "utility_scorer_path": predicted_model_path,
        "utility_feature_columns_path": predicted_feature_columns_path,
        "predicted_utility_scorer_model_path": predicted_model_path,
        "predicted_utility_feature_columns_path": predicted_feature_columns_path,
        "predicted_utility_scorer_source_experiment_dir": predicted_source_experiment_dir,
        "experiment_summary_path": str(run_paths.summary_path),
    }


def select_replay_batch(
    replay_buffer: ReplayBuffer,
    current_task_name: str,
    current_train_size: int,
    seed: int,
    replay_ratio: float,
) -> list:
    """Select a deterministic replay subset for the current task."""
    max_replay_samples = int(current_train_size * replay_ratio)
    if max_replay_samples <= 0 or replay_buffer.total_size() == 0:
        return []

    replay_candidates = replay_buffer.get_all_samples()
    if len(replay_candidates) <= max_replay_samples:
        return replay_candidates

    rng = random.Random(f"{seed}:{current_task_name}:replay")
    indices = list(range(len(replay_candidates)))
    rng.shuffle(indices)
    selected_indices = sorted(indices[:max_replay_samples])
    return [replay_candidates[index] for index in selected_indices]


def select_random_samples_for_replay(
    samples: list,
    top_k: int,
    seed: int,
    task_name: str,
) -> tuple[list, list[dict[str, Any]]]:
    """Select replay samples directly from the current task with a deterministic RNG."""
    if top_k <= 0 or not samples:
        return [], []

    rng = random.Random(f"{seed}:{task_name}:random_replay_buffer")
    rows: list[dict[str, Any]] = []
    keyed_samples = []
    for sample in samples:
        random_key = rng.random()
        keyed_samples.append((random_key, sample))

    ranked_samples = sorted(keyed_samples, key=lambda item: (item[0], item[1].sample_id), reverse=True)
    selected_ids = {sample.sample_id for _, sample in ranked_samples[:top_k]}
    for rank, (random_key, sample) in enumerate(ranked_samples, start=1):
        row = asdict(sample)
        row["replay_mode"] = "random"
        row["replay_selection_strategy"] = "random"
        row["random_selection_score"] = random_key
        row["random_selection_rank"] = rank
        row["was_selected_for_replay"] = int(sample.sample_id in selected_ids)
        rows.append(row)

    selected_samples = [sample for _, sample in ranked_samples[:top_k]]
    return selected_samples, rows


def build_random_score_rows(random_selection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build score-like rows so downstream metadata files can mark random selections."""
    rows: list[dict[str, Any]] = []
    for row in random_selection_rows:
        rows.append(
            {
                "sample_id": row.get("sample_id"),
                "task_name": row.get("task_name"),
                "split": row.get("split"),
                "source_dataset": row.get("source_dataset"),
                "input_text": row.get("input_text"),
                "target_text": row.get("target_text"),
                "token_count": None,
                "loss": None,
                "surprise_score": None,
                "replay_selection_score": row.get("random_selection_score"),
            }
        )
    return rows


def main() -> None:
    """Run continual fine-tuning using processed data and optional audited review metadata."""
    args = parse_args()
    config = get_config()

    if args.data_dir is not None:
        config.data.data_dir = resolve_project_path(config, args.data_dir)

    if args.audited_review_csv is not None:
        config.data.audited_review_csv = resolve_project_path(config, args.audited_review_csv)

    if args.use_audited_review:
        config.data.use_audited_review = True
        config.data.data_source = "processed_plus_audited_review"
        config.data.audited_review_source_note = (
            "The audited review CSV is attached for replay selection and utility review "
            "provenance only. Train, validation, and test samples are still loaded from "
            "processed JSONL files unless a future audited train/val/test data source is "
            "provided explicitly."
        )

    if args.seed is not None:
        config.training.seed = args.seed

    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be a positive integer.")
        config.training.num_epochs = args.epochs

    if args.replay_buffer_size is not None:
        if args.replay_buffer_size <= 0:
            raise ValueError("--replay-buffer-size must be a positive integer.")
        config.replay.replay_buffer_size = args.replay_buffer_size

    if args.replay_selection_strategy is not None:
        config.replay.replay_selection_strategy = args.replay_selection_strategy

    if args.replay_mode is None:
        config.replay.replay_mode = (
            "random"
            if config.replay.replay_selection_strategy == "random"
            else "scored"
        )
    else:
        config.replay.replay_mode = args.replay_mode

    if args.no_replay:
        if args.replay_mode in {"scored", "random"}:
            print(
                "Warning: --no-replay was provided with --replay-mode "
                f"{args.replay_mode}; using replay_mode=none."
            )
        config.replay.replay_mode = "none"

    if config.replay.replay_mode == "random":
        config.replay.replay_selection_strategy = "random"
    elif (
        args.replay_mode == "scored"
        and config.replay.replay_selection_strategy == "random"
    ):
        print(
            "Warning: --replay-mode scored cannot use random as a scored strategy; "
            "using replay_selection_strategy=surprise."
        )
        config.replay.replay_selection_strategy = "surprise"

    if args.tasks:
        unknown_tasks = [task_name for task_name in args.tasks if task_name not in config.data.task_order]
        if unknown_tasks:
            raise ValueError(f"Unknown task names: {unknown_tasks}")
        config.data.task_order = args.tasks

    if config.replay.replay_mode == "none":
        config.replay.top_k_per_task = 0
        config.replay.replay_ratio = 0.0

    utility_scorer_path = Path(args.utility_scorer_path).expanduser() if args.utility_scorer_path else None
    utility_feature_columns_path = (
        Path(args.utility_feature_columns_path).expanduser()
        if args.utility_feature_columns_path
        else None
    )
    if (utility_scorer_path is None) != (utility_feature_columns_path is None):
        raise ValueError(
            "Provide both --utility-scorer-path and --utility-feature-columns-path together."
        )
    if (
        utility_scorer_path is not None
        and config.replay.replay_selection_strategy != "predicted_utility"
    ):
        raise ValueError(
            "--utility-scorer-path and --utility-feature-columns-path are only valid when "
            "--replay-selection-strategy predicted_utility is selected."
        )

    ensure_local_processed_data(config)
    audited_review_metadata = None
    if config.data.use_audited_review:
        audited_review_metadata = inspect_audited_review_csv(config.data.audited_review_csv)
        if not audited_review_metadata["contains_full_train_val_test"]:
            print(
                "Warning: audited review CSV does not contain complete train/val/test "
                "splits; keeping train/val/test loading on processed JSONL."
            )

    set_global_seed(config.training.seed)

    run_paths = create_run_paths(config, args.run_name)
    replay_enabled = (
        config.replay.replay_mode != "none"
        and config.replay.replay_buffer_size > 0
        and config.replay.top_k_per_task > 0
        and config.replay.replay_ratio > 0.0
    )
    predicted_utility_artifact_paths: tuple[Path, Path] | None = None
    if config.replay.replay_selection_strategy == "predicted_utility":
        predicted_utility_artifact_paths = resolve_predicted_utility_artifact_paths(
            run_paths.run_dir,
            utility_scorer_path=utility_scorer_path,
            utility_feature_columns_path=utility_feature_columns_path,
        )

    if args.run_name and not args.run_name.startswith("exp"):
        print("Warning: formal experiment run names should start with expXXX_ for easier tracking.")

    write_json(
        run_paths.run_config_path,
        build_config_payload(
            config=config,
            run_paths=run_paths,
            replay_enabled=replay_enabled,
            predicted_utility_artifact_paths=predicted_utility_artifact_paths,
            audited_review_metadata=audited_review_metadata,
        ),
    )

    if args.dry_run_config:
        print(f"Dry run complete. Wrote run_config.json to {run_paths.run_config_path}")
        return

    model, tokenizer = load_model_and_tokenizer(config.model.model_name, config.training.device)
    replay_buffer = ReplayBuffer(max_size=config.replay.replay_buffer_size)

    train_summaries: list[dict[str, Any]] = []
    train_steps: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    final_test_rows: list[dict[str, Any]] = []
    buffer_rows: list[dict[str, Any]] = []
    best_accuracy_by_task: dict[str, float] = {}
    seen_tasks: list[str] = []

    print(f"Run name: {run_paths.run_name}")
    print(f"Device: {config.training.device}")
    print(f"Tasks: {config.data.task_order}")
    print(f"Data dir: {config.data.data_dir}")
    print(f"Data source: {config.data.data_source}")
    if config.data.use_audited_review:
        print(f"Audited review CSV: {config.data.audited_review_csv}")
    print(f"Replay mode: {config.replay.replay_mode}")
    print(f"Replay selection strategy: {config.replay.replay_selection_strategy}")
    if predicted_utility_artifact_paths is not None:
        print(f"Predicted utility scorer model: {predicted_utility_artifact_paths[0]}")
        print(f"Predicted utility features: {predicted_utility_artifact_paths[1]}")
    print()

    utility_label_paths: list[Path] = []
    for task_name in config.data.task_order:
        print(f"[TRAIN] {task_name}")
        current_train_samples = load_task_samples(config, task_name, "train")
        replay_samples = select_replay_batch(
            replay_buffer=replay_buffer,
            current_task_name=task_name,
            current_train_size=len(current_train_samples),
            seed=config.training.seed,
            replay_ratio=config.replay.replay_ratio,
        )
        combined_train_samples = current_train_samples + replay_samples

        train_summary, step_rows, sample_signal_rows = train_task(
            model=model,
            tokenizer=tokenizer,
            train_samples=combined_train_samples,
            config=config,
            task_name=task_name,
        )
        train_summary["current_task_samples"] = len(current_train_samples)
        train_summary["replay_samples_used"] = len(replay_samples)
        train_summaries.append(train_summary)
        train_steps.extend(step_rows)

        checkpoint_dir = run_paths.checkpoint_dir / f"after_{task_name}"
        save_model_and_tokenizer(model, tokenizer, checkpoint_dir)

        seen_tasks.append(task_name)
        for eval_task_name in seen_tasks:
            val_samples = load_task_samples(config, eval_task_name, "val")
            metrics, prediction_rows = evaluate_task(
                model=model,
                tokenizer=tokenizer,
                samples=val_samples,
                config=config,
                task_name=eval_task_name,
            )
            previous_best = best_accuracy_by_task.get(eval_task_name, metrics["accuracy"])
            forgetting = max(previous_best - metrics["accuracy"], 0.0)
            best_accuracy_by_task[eval_task_name] = max(previous_best, metrics["accuracy"])

            metrics["after_task"] = task_name
            metrics["eval_split"] = "val"
            metrics["best_accuracy_so_far"] = best_accuracy_by_task[eval_task_name]
            metrics["forgetting"] = forgetting
            eval_rows.append(metrics)

            pd.DataFrame(prediction_rows).to_csv(
                run_paths.metrics_dir / f"val_predictions_after_{task_name}__{eval_task_name}.csv",
                index=False,
            )

        snapshot_path = run_paths.replay_buffers_dir / f"replay_buffer_after_{task_name}.jsonl"
        selected_samples = []
        selected_sample_ids: set[str] = set()
        if config.replay.replay_mode == "random":
            selected_samples, random_selection_rows = select_random_samples_for_replay(
                samples=current_train_samples,
                top_k=config.replay.top_k_per_task,
                seed=config.training.seed,
                task_name=task_name,
            )
            scored_selection_rows = build_random_score_rows(random_selection_rows)
            pd.DataFrame(random_selection_rows).to_csv(
                run_paths.replay_buffers_dir / f"{task_name}_random_selection.csv",
                index=False,
            )
        else:
            score_rows = score_samples(
                model=model,
                tokenizer=tokenizer,
                samples=current_train_samples,
                config=config,
                description=f"Scoring {task_name}",
            )
            allow_missing_predicted_utility = (
                not replay_enabled and config.replay.replay_selection_strategy == "predicted_utility"
            )
            scored_selection_rows = attach_replay_selection_scores(
                score_rows=score_rows,
                strategy=config.replay.replay_selection_strategy,
                task_name=task_name,
                seed=config.training.seed,
                run_dir=run_paths.run_dir,
                utility_scorer_path=(
                    predicted_utility_artifact_paths[0]
                    if predicted_utility_artifact_paths is not None
                    else None
                ),
                utility_feature_columns_path=(
                    predicted_utility_artifact_paths[1]
                    if predicted_utility_artifact_paths is not None
                    else None
                ),
                allow_missing_predicted_utility=allow_missing_predicted_utility,
            )

        if replay_enabled:
            if config.replay.replay_mode == "scored":
                pd.DataFrame(score_rows).to_csv(
                    run_paths.replay_scores_dir / f"{task_name}_scores.csv",
                    index=False,
                )

                selected_samples = select_samples_for_replay(
                    samples=current_train_samples,
                    scored_rows=scored_selection_rows,
                    top_k=config.replay.top_k_per_task,
                )
            selected_sample_ids = {sample.sample_id for sample in selected_samples}
            replay_buffer.add_samples(task_name, selected_samples)
            replay_buffer.save_snapshot(snapshot_path)

            for sample in selected_samples:
                selection_row = next(
                    (row for row in scored_selection_rows if row.get("sample_id") == sample.sample_id),
                    {},
                )
                buffer_rows.append(
                    {
                        "after_task": task_name,
                        "sample_id": sample.sample_id,
                        "task_name": sample.task_name,
                        "split": sample.split,
                        "source_dataset": sample.source_dataset,
                        "replay_mode": config.replay.replay_mode,
                        "replay_selection_strategy": config.replay.replay_selection_strategy,
                        "replay_selection_score": selection_row.get("replay_selection_score"),
                    }
                )
        else:
            write_text(snapshot_path, "")

        sample_signal_output_path = write_sample_signals(
            output_dir=run_paths.sample_signals_dir,
            task_name=task_name,
            sample_signal_rows=enrich_and_filter_sample_signals(
                sample_signal_rows=sample_signal_rows,
                score_rows=scored_selection_rows,
                current_task_name=task_name,
                replay_selection_strategy=config.replay.replay_selection_strategy,
                selected_sample_ids=selected_sample_ids,
            ),
        )
        utility_label_output_path = generate_proxy_utility_labels(
            sample_signals_path=sample_signal_output_path,
            output_dir=run_paths.utility_labels_dir,
        )
        utility_label_paths.append(utility_label_output_path)
        write_replay_selection_comparison(
            output_dir=run_paths.replay_selection_comparison_dir,
            task_name=task_name,
            scored_rows=scored_selection_rows,
            selected_sample_ids=selected_sample_ids,
            strategy=config.replay.replay_selection_strategy,
        )

        print(
            f"        current_train={len(current_train_samples)} "
            f"replay_used={len(replay_samples)} "
            f"buffer_size={replay_buffer.total_size()}"
        )
        print()

    for task_name in config.data.task_order:
        test_samples = load_task_samples(config, task_name, "test")
        test_metrics, prediction_rows = evaluate_task(
            model=model,
            tokenizer=tokenizer,
            samples=test_samples,
            config=config,
            task_name=task_name,
        )
        test_metrics["after_task"] = config.data.task_order[-1]
        test_metrics["eval_split"] = "test"
        final_test_rows.append(test_metrics)
        pd.DataFrame(prediction_rows).to_csv(
            run_paths.metrics_dir / f"test_predictions_{task_name}.csv",
            index=False,
        )

    pd.DataFrame(train_summaries).to_csv(run_paths.log_dir / "train_summary.csv", index=False)
    pd.DataFrame(train_steps).to_csv(run_paths.log_dir / "train_steps.csv", index=False)
    pd.DataFrame(buffer_rows).to_csv(run_paths.replay_buffers_dir / "replay_buffer_summary.csv", index=False)
    pd.DataFrame(eval_rows).to_csv(run_paths.metrics_dir / "val_metrics.csv", index=False)
    pd.DataFrame(final_test_rows).to_csv(run_paths.metrics_dir / "test_metrics.csv", index=False)
    export_human_review_candidates(utility_label_paths, run_paths.human_review_dir)

    generate_experiment_summary(run_paths.experiment_id)
    update_experiment_index()


if __name__ == "__main__":
    main()
