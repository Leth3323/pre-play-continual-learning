from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from config import AppConfig
from data_loader import InstructionSample, build_dataloader


def score_samples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: list[InstructionSample],
    config: AppConfig,
    description: str,
) -> list[dict[str, Any]]:
    if not samples:
        return []

    sample_lookup = {sample.sample_id: sample for sample in samples}
    dataloader = build_dataloader(
        samples=samples,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_input_length=config.model.max_input_length,
        max_target_length=config.model.max_target_length,
        shuffle=False,
    )
    loss_function = CrossEntropyLoss(ignore_index=-100, reduction="none")
    score_rows: list[dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=description, leave=False):
            sample_ids = batch.pop("sample_ids")
            task_names = batch.pop("task_names")
            target_texts = batch.pop("target_texts")

            tensor_batch = {key: value.to(config.training.device) for key, value in batch.items()}
            outputs = model(**tensor_batch)
            logits = outputs.logits
            labels = tensor_batch["labels"]

            token_losses = loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            token_losses = token_losses.reshape(labels.size(0), labels.size(1))
            token_mask = labels.ne(-100)
            token_counts = token_mask.sum(dim=1).clamp(min=1)
            sequence_losses = (token_losses * token_mask.float()).sum(dim=1) / token_counts

            for index, sample_id in enumerate(sample_ids):
                sample = sample_lookup.get(sample_id)
                score_rows.append(
                    {
                        "sample_id": sample_id,
                        "task_name": sample.task_name if sample is not None else task_names[index],
                        "split": sample.split if sample is not None else None,
                        "source_dataset": sample.source_dataset if sample is not None else None,
                        "input_text": sample.input_text if sample is not None else None,
                        "target_text": target_texts[index],
                        "token_count": int(token_counts[index].item()),
                        "loss": float(sequence_losses[index].item()),
                        "surprise_score": float(sequence_losses[index].item()),
                    }
                )

    return score_rows


def select_top_k_samples(
    samples: list[InstructionSample],
    score_rows: list[dict[str, Any]],
    top_k: int,
    score_field: str = "surprise_score",
) -> list[InstructionSample]:
    if top_k <= 0 or not samples:
        return []

    score_lookup = {row["sample_id"]: row.get(score_field, row.get("surprise_score")) for row in score_rows}
    ranked_samples = sorted(
        samples,
        key=lambda sample: score_lookup.get(sample.sample_id, float("-inf")),
        reverse=True,
    )
    return ranked_samples[:top_k]


def samples_to_rows(samples: list[InstructionSample]) -> list[dict[str, Any]]:
    return [asdict(sample) for sample in samples]
