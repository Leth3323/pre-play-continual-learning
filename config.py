from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from typing import Optional

import torch


def detect_device() -> str:
    """Detect the best available device for PyTorch."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class PathConfig:
    """Filesystem paths used by the project."""

    project_root: Path
    data_raw: Path
    data_processed: Path
    data_audit: Path
    outputs_root: Path
    outputs_experiments: Path


@dataclass
class ModelConfig:
    """Model and tokenization settings."""

    model_name: str = "google/flan-t5-small"
    max_input_length: int = 256
    max_target_length: int = 32
    generation_max_new_tokens: int = 32


@dataclass
class TrainingConfig:
    """Training hyperparameters and runtime settings."""

    seed: int = 42
    batch_size: int = 2
    learning_rate: float = 5e-4
    num_epochs: int = 1
    weight_decay: float = 0.0
    warmup_ratio: float = 0.06
    gradient_clip_norm: float = 1.0
    device: str = field(default_factory=detect_device)


@dataclass
class DataConfig:
    """Dataset order and processed subset sizes."""

    task_order: list[str] = field(
        default_factory=lambda: ["sciq", "arc_challenge", "boolq", "gsm8k"]
    )
    train_sample_size: int = 100
    val_sample_size: int = 30
    test_sample_size: int = 50


@dataclass
class ReplayConfig:
    """Replay buffer settings."""

    replay_buffer_size: int = 100
    replay_ratio: float = 0.2
    top_k_per_task: int = 30
    replay_mode: str = "scored"
    replay_selection_strategy: str = "surprise"


@dataclass
class AppConfig:
    """Top-level PRE-PLAY configuration."""

    paths: PathConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)


@dataclass
class RunPaths:
    """Run-specific output paths."""

    experiment_id: str
    run_name: str
    timestamp: str
    output_root: Path
    experiment_dir: Path
    run_dir: Path
    checkpoint_dir: Path
    log_dir: Path
    replay_scores_dir: Path
    metrics_dir: Path
    replay_buffers_dir: Path
    sample_signals_dir: Path
    utility_labels_dir: Path
    human_review_dir: Path
    replay_selection_comparison_dir: Path
    scorer_outputs_dir: Path
    scorer_checkpoints_dir: Path
    run_config_path: Path
    summary_path: Path


EXPERIMENT_ID_PATTERN = re.compile(r"^(exp\d+)(?:_|$)")


def _build_path_config(project_root: Optional[Path] = None) -> PathConfig:
    """Build the project path configuration."""
    root = project_root or Path(__file__).resolve().parent
    data_dir = root / "data"
    outputs_dir = root / "outputs"
    return PathConfig(
        project_root=root,
        data_raw=data_dir / "raw",
        data_processed=data_dir / "processed",
        data_audit=data_dir / "audit",
        outputs_root=outputs_dir,
        outputs_experiments=outputs_dir / "experiments",
    )


def create_directories(config: AppConfig) -> None:
    """Create all required project directories."""
    directories = [
        config.paths.data_raw,
        config.paths.data_processed,
        config.paths.data_audit,
        config.paths.outputs_root,
        config.paths.outputs_experiments,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_config(project_root: Optional[Path] = None) -> AppConfig:
    """Create and return the application configuration."""
    config = AppConfig(paths=_build_path_config(project_root=project_root))
    create_directories(config)
    return config


def get_raw_dataset_dir(config: AppConfig, task_name: str) -> Path:
    """Return the local raw dataset directory for a task."""
    return config.paths.data_raw / task_name


def get_processed_task_dir(config: AppConfig, task_name: str) -> Path:
    """Return the local processed dataset directory for a task."""
    return config.paths.data_processed / task_name


def extract_experiment_id(run_name: str) -> str:
    """Extract an experiment id like exp004 from a run name."""
    match = EXPERIMENT_ID_PATTERN.match(run_name)
    if match:
        return match.group(1)
    return run_name


def get_experiment_dir(config: AppConfig, run_name: str) -> Path:
    """Return the experiment-level directory for a run name."""
    return config.paths.outputs_experiments / extract_experiment_id(run_name)


def get_run_dir(config: AppConfig, run_name: str) -> Path:
    """Return the nested run directory for a run name."""
    return get_experiment_dir(config, run_name) / run_name


def get_experiment_summary_path(config: AppConfig, run_name: str) -> Path:
    """Return the experiment-level summary path for a run name."""
    experiment_id = extract_experiment_id(run_name)
    return get_experiment_dir(config, run_name) / f"{experiment_id}_summary.md"


def create_run_paths(config: AppConfig, run_name: str | None = None) -> RunPaths:
    """Create run-scoped output directories."""
    resolved_run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().isoformat(timespec="seconds")
    experiment_id = extract_experiment_id(resolved_run_name)
    experiment_dir = get_experiment_dir(config, resolved_run_name)
    run_dir = get_run_dir(config, resolved_run_name)
    run_paths = RunPaths(
        experiment_id=experiment_id,
        run_name=resolved_run_name,
        timestamp=timestamp,
        output_root=config.paths.outputs_root,
        experiment_dir=experiment_dir,
        run_dir=run_dir,
        checkpoint_dir=run_dir / "checkpoints",
        log_dir=run_dir / "logs",
        replay_scores_dir=run_dir / "replay_scores",
        metrics_dir=run_dir / "metrics",
        replay_buffers_dir=run_dir / "replay_buffers",
        sample_signals_dir=run_dir / "sample_signals",
        utility_labels_dir=run_dir / "utility_labels",
        human_review_dir=run_dir / "human_review",
        replay_selection_comparison_dir=run_dir / "replay_selection_comparison",
        scorer_outputs_dir=run_dir / "scorer_outputs",
        scorer_checkpoints_dir=run_dir / "scorer_checkpoints",
        run_config_path=run_dir / "run_config.json",
        summary_path=get_experiment_summary_path(config, resolved_run_name),
    )
    for directory in (
        run_paths.experiment_dir,
        run_paths.run_dir,
        run_paths.checkpoint_dir,
        run_paths.log_dir,
        run_paths.replay_scores_dir,
        run_paths.metrics_dir,
        run_paths.replay_buffers_dir,
        run_paths.sample_signals_dir,
        run_paths.utility_labels_dir,
        run_paths.human_review_dir,
        run_paths.replay_selection_comparison_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return run_paths
