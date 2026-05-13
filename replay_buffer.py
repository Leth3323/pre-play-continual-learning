from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from data_loader import InstructionSample


class ReplayBuffer:
    """A simple replay buffer with a global size limit."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.samples_by_task: dict[str, list[InstructionSample]] = {}

    def add_samples(self, task_name: str, samples: list[InstructionSample]) -> None:
        """Add selected samples for one task and trim the global buffer if needed."""
        self.samples_by_task[task_name] = list(samples)
        self._trim_to_max_size()

    def get_all_samples(self, limit: int | None = None) -> list[InstructionSample]:
        """Return replay samples in task insertion order."""
        replay_samples: list[InstructionSample] = []
        for task_name in self.samples_by_task:
            replay_samples.extend(self.samples_by_task[task_name])
        if limit is None:
            return replay_samples
        return replay_samples[:limit]

    def total_size(self) -> int:
        """Return the total number of stored replay samples."""
        return sum(len(samples) for samples in self.samples_by_task.values())

    def save_snapshot(self, output_path: Path) -> None:
        """Save the current buffer contents to a JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_handle:
            for task_name, samples in self.samples_by_task.items():
                for sample in samples:
                    row = asdict(sample)
                    row["buffer_task_name"] = task_name
                    file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _trim_to_max_size(self) -> None:
        """Trim the oldest task samples first when the buffer is too large."""
        overflow = self.total_size() - self.max_size
        if overflow <= 0:
            return

        task_names = list(self.samples_by_task.keys())
        for task_name in task_names:
            if overflow <= 0:
                break
            task_samples = self.samples_by_task[task_name]
            remove_count = min(len(task_samples), overflow)
            self.samples_by_task[task_name] = task_samples[remove_count:]
            overflow -= remove_count

        empty_tasks = [task_name for task_name, samples in self.samples_by_task.items() if not samples]
        for task_name in empty_tasks:
            del self.samples_by_task[task_name]
