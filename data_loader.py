from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from config import AppConfig, get_processed_task_dir


@dataclass
class InstructionSample:
    sample_id: str
    task_name: str
    split: str
    input_text: str
    target_text: str
    source_dataset: str


class InstructionDataset(Dataset):
    def __init__(self, samples: list[InstructionSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> InstructionSample:
        return self.samples[index]


class Seq2SeqCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_input_length: int,
        max_target_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, batch: list[InstructionSample]) -> dict:
        input_texts = [sample.input_text for sample in batch]
        target_texts = [sample.target_text for sample in batch]

        model_inputs = self.tokenizer(
            input_texts,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        target_tokens = self.tokenizer(
            text_target=target_texts,
            max_length=self.max_target_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels = target_tokens["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels
        model_inputs["sample_ids"] = [sample.sample_id for sample in batch]
        model_inputs["task_names"] = [sample.task_name for sample in batch]
        model_inputs["target_texts"] = target_texts
        return model_inputs


def load_samples_from_jsonl(input_path: Path) -> list[InstructionSample]:
    """Load local processed samples from a JSONL file."""
    samples: list[InstructionSample] = []
    with input_path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            row = json.loads(line)
            samples.append(
                InstructionSample(
                    sample_id=row["sample_id"],
                    task_name=row["task_name"],
                    split=row["split"],
                    input_text=row["input_text"],
                    target_text=row["target_text"],
                    source_dataset=row["source_dataset"],
                )
            )
    return samples


def load_task_samples(config: AppConfig, task_name: str, split_name: str) -> list[InstructionSample]:
    """Load one processed split for a task."""
    task_dir = get_processed_task_dir(config, task_name)
    input_path = task_dir / f"{split_name}.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Processed split not found: {input_path}")
    return load_samples_from_jsonl(input_path)


def build_dataloader(
    samples: list[InstructionSample],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_input_length: int,
    max_target_length: int,
    shuffle: bool,
) -> DataLoader:
    """Build a PyTorch dataloader for processed instruction samples."""
    dataset = InstructionDataset(samples)
    collator = Seq2SeqCollator(
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )
