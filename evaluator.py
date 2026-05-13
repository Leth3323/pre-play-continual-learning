from __future__ import annotations

import re
from typing import Any

import torch
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from config import AppConfig
from data_loader import InstructionSample, build_dataloader


def normalize_answer(text: str) -> str:
    normalized = text.strip().lower().replace(",", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def evaluate_task(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: list[InstructionSample],
    config: AppConfig,
    task_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not samples:
        return {"task_name": task_name, "accuracy": 0.0, "num_samples": 0}, []

    dataloader = build_dataloader(
        samples=samples,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_input_length=config.model.max_input_length,
        max_target_length=config.model.max_target_length,
        shuffle=False,
    )

    references: list[str] = []
    predictions: list[str] = []
    prediction_rows: list[dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {task_name}", leave=False):
            sample_ids = batch.pop("sample_ids")
            task_names = batch.pop("task_names")
            target_texts = batch.pop("target_texts")

            input_ids = batch["input_ids"].to(config.training.device)
            attention_mask = batch["attention_mask"].to(config.training.device)
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.model.generation_max_new_tokens,
            )
            decoded_predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for index, prediction in enumerate(decoded_predictions):
                normalized_prediction = normalize_answer(prediction)
                normalized_reference = normalize_answer(target_texts[index])
                predictions.append(normalized_prediction)
                references.append(normalized_reference)
                prediction_rows.append(
                    {
                        "sample_id": sample_ids[index],
                        "task_name": task_names[index],
                        "prediction": prediction.strip(),
                        "normalized_prediction": normalized_prediction,
                        "target_text": target_texts[index],
                        "normalized_target_text": normalized_reference,
                        "is_correct": normalized_prediction == normalized_reference,
                    }
                )

    accuracy = float(accuracy_score(references, predictions))
    metrics = {
        "task_name": task_name,
        "accuracy": accuracy,
        "num_samples": len(samples),
    }
    return metrics, prediction_rows
