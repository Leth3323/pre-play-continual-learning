from __future__ import annotations

from typing import Any

import torch
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from config import AppConfig
from data_loader import InstructionSample, build_dataloader


def train_task(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_samples: list[InstructionSample],
    config: AppConfig,
    task_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not train_samples:
        return {
            "task_name": task_name,
            "training_samples": 0,
            "epochs": config.training.num_epochs,
            "steps": 0,
            "mean_loss": 0.0,
        }, [], []

    sample_lookup = {sample.sample_id: sample for sample in train_samples}
    dataloader = build_dataloader(
        samples=train_samples,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_input_length=config.model.max_input_length,
        max_target_length=config.model.max_target_length,
        shuffle=True,
    )
    total_steps = max(len(dataloader) * config.training.num_epochs, 1)
    warmup_steps = int(total_steps * config.training.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    step_rows: list[dict[str, Any]] = []
    sample_signal_rows: list[dict[str, Any]] = []
    loss_values: list[float] = []
    global_step = 0
    loss_function = CrossEntropyLoss(ignore_index=-100, reduction="none")

    optimizer.zero_grad(set_to_none=True)
    model.train()
    for epoch_index in range(config.training.num_epochs):
        progress = tqdm(dataloader, desc=f"Training {task_name} epoch {epoch_index + 1}", leave=False)
        for batch in progress:
            sample_ids = batch.pop("sample_ids")
            batch.pop("task_names")
            batch.pop("target_texts")

            tensor_batch = {key: value.to(config.training.device) for key, value in batch.items()}
            outputs = model(**tensor_batch)
            loss = outputs.loss

            with torch.no_grad():
                logits = outputs.logits.detach()
                labels = tensor_batch["labels"]
                token_losses = loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                token_losses = token_losses.reshape(labels.size(0), labels.size(1))
                token_mask = labels.ne(-100)
                token_counts = token_mask.sum(dim=1).clamp(min=1)
                sequence_losses = (token_losses * token_mask.float()).sum(dim=1) / token_counts

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            loss_value = float(loss.item())
            loss_values.append(loss_value)
            step_rows.append(
                {
                    "task_name": task_name,
                    "epoch": epoch_index + 1,
                    "step": global_step,
                    "loss": loss_value,
                }
            )
            for index, sample_id in enumerate(sample_ids):
                sample = sample_lookup.get(sample_id)
                if sample is None:
                    continue
                sample_signal_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "task_name": sample.task_name,
                        "split": sample.split,
                        "source_dataset": sample.source_dataset,
                        "input_text": sample.input_text,
                        "target_text": sample.target_text,
                        "token_count": int(token_counts[index].item()),
                        "epoch": epoch_index + 1,
                        "step": global_step,
                        "loss": float(sequence_losses[index].item()),
                        "surprise_score": float(sequence_losses[index].item()),
                        "replay_selection_strategy": config.replay.replay_selection_strategy,
                        "was_added_to_replay_buffer": 0,
                        "replay_selection_score": None,
                    }
                )
            progress.set_postfix(loss=f"{loss_value:.4f}")

    summary = {
        "task_name": task_name,
        "training_samples": len(train_samples),
        "epochs": config.training.num_epochs,
        "steps": global_step,
        "mean_loss": sum(loss_values) / len(loss_values),
        "replay_selection_strategy": config.replay.replay_selection_strategy,
    }
    return summary, step_rows, sample_signal_rows
