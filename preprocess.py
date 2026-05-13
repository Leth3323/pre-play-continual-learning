from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from datasets import DatasetDict, load_from_disk

from config import AppConfig, get_config, get_processed_task_dir, get_raw_dataset_dir


OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

TASK_SPECS: dict[str, dict[str, Any]] = {
    "sciq": {
        "source_dataset": "allenai/sciq",
        "raw_to_output_splits": {
            "train": ["train"],
            "validation": ["val"],
            "test": ["test"],
        },
    },
    "arc_challenge": {
        "source_dataset": "allenai/ai2_arc",
        "raw_to_output_splits": {
            "train": ["train"],
            "validation": ["val"],
            "test": ["test"],
        },
    },
    "boolq": {
        "source_dataset": "google/boolq",
        "raw_to_output_splits": {
            "train": ["train"],
            "validation": ["val", "test"],
        },
    },
    "gsm8k": {
        "source_dataset": "openai/gsm8k",
        "raw_to_output_splits": {
            "train": ["train"],
            "test": ["val", "test"],
        },
    },
}


def preprocess_all_datasets(config: AppConfig, force_reprocess: bool = False) -> None:
    """Preprocess all configured datasets."""
    for task_name in config.data.task_order:
        try:
            preprocess_task(config, task_name, force_reprocess=force_reprocess)
        except Exception as error:
            print(f"[ERROR] Failed to preprocess {task_name}: {error}")
            print()


def preprocess_task(config: AppConfig, task_name: str, force_reprocess: bool = False) -> None:
    """Preprocess one task into local JSONL files and an audit CSV."""
    processed_dir = get_processed_task_dir(config, task_name)
    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "val.jsonl"
    test_path = processed_dir / "test.jsonl"
    audit_path = config.paths.data_audit / f"{task_name}_audit.csv"

    if not force_reprocess and train_path.exists() and val_path.exists() and test_path.exists() and audit_path.exists():
        print(f"[SKIP] {task_name}: processed files already exist")
        return

    raw_dir = get_raw_dataset_dir(config, task_name)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset not found for task {task_name}: {raw_dir}")

    print(f"[PROCESS] {task_name}")
    print(f"          Raw path: {raw_dir}")

    task_spec = TASK_SPECS[task_name]
    dataset_dict = load_from_disk(str(raw_dir))
    processed_dir.mkdir(parents=True, exist_ok=True)

    records_by_split = _build_records_for_task(
        config=config,
        task_name=task_name,
        dataset_dict=dataset_dict,
        task_spec=task_spec,
    )

    for split_name in ("train", "val", "test"):
        output_path = processed_dir / f"{split_name}.jsonl"
        _write_jsonl(records_by_split[split_name], output_path)
        print(f"          Wrote {len(records_by_split[split_name])} rows to {output_path}")

    audit_rows = _build_audit_rows(
        train_records=records_by_split["train"],
        val_records=records_by_split["val"],
        test_records=records_by_split["test"],
    )
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    print(f"          Wrote audit CSV to {audit_path}")
    print(f"[DONE] {task_name}")
    print()


def _build_records_for_task(
    config: AppConfig,
    task_name: str,
    dataset_dict: DatasetDict,
    task_spec: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    formatter = _get_formatter(task_name)
    split_sizes = {
        "train": config.data.train_sample_size,
        "val": config.data.val_sample_size,
        "test": config.data.test_sample_size,
    }
    payloads_by_split: dict[str, list[dict[str, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for raw_split_name, output_splits in task_spec["raw_to_output_splits"].items():
        if raw_split_name not in dataset_dict:
            raise KeyError(f"Split '{raw_split_name}' not found for task '{task_name}'")

        raw_split = dataset_dict[raw_split_name]
        selected_indices = _allocate_indices(
            total_size=len(raw_split),
            output_splits=output_splits,
            split_sizes=split_sizes,
            seed=config.training.seed,
            task_name=task_name,
            raw_split_name=raw_split_name,
        )

        for output_split in output_splits:
            indices = selected_indices[output_split]
            print(
                f"          Raw split '{raw_split_name}' -> '{output_split}': "
                f"{len(indices)} selected from {len(raw_split)}"
            )
            for index in indices:
                input_text, target_text = formatter(raw_split[index], config.training.seed, task_name, raw_split_name, index)
                payloads_by_split[output_split].append(
                    {
                        "input_text": input_text,
                        "target_text": target_text,
                    }
                )

    records_by_split: dict[str, list[dict[str, str]]] = {}
    source_dataset = task_spec["source_dataset"]
    for split_name in ("train", "val", "test"):
        records_by_split[split_name] = _build_records(
            task_name=task_name,
            split_name=split_name,
            source_dataset=source_dataset,
            payloads=payloads_by_split[split_name],
        )
    return records_by_split


def _get_formatter(task_name: str) -> Callable[[dict[str, Any], int, str, str, int], tuple[str, str]]:
    formatters: dict[str, Callable[[dict[str, Any], int, str, str, int], tuple[str, str]]] = {
        "sciq": _format_sciq,
        "arc_challenge": _format_arc_challenge,
        "boolq": _format_boolq,
        "gsm8k": _format_gsm8k,
    }
    if task_name not in formatters:
        raise KeyError(f"Unsupported task: {task_name}")
    return formatters[task_name]


def _format_sciq(
    example: dict[str, Any],
    seed: int,
    task_name: str,
    raw_split_name: str,
    raw_index: int,
) -> tuple[str, str]:
    question = _clean_text(example["question"])
    correct_answer = _clean_text(example["correct_answer"])
    options = [
        correct_answer,
        _clean_text(example["distractor1"]),
        _clean_text(example["distractor2"]),
        _clean_text(example["distractor3"]),
    ]

    rng = random.Random(_stable_seed(seed, task_name, raw_split_name, raw_index, "options"))
    rng.shuffle(options)

    option_lines = [
        f"{OPTION_LABELS[index]}. {option_text}"
        for index, option_text in enumerate(options)
    ]
    input_text = (
        "Task: Answer the science question.\n"
        f"Question: {question}\n"
        "Options:\n"
        + "\n".join(option_lines)
        + "\nAnswer:"
    )
    return input_text, correct_answer


def _format_arc_challenge(
    example: dict[str, Any],
    seed: int,
    task_name: str,
    raw_split_name: str,
    raw_index: int,
) -> tuple[str, str]:
    del seed, task_name, raw_split_name, raw_index
    question = _clean_text(example["question"])
    choice_texts = [_clean_text(text) for text in example["choices"]["text"]]
    choice_labels = [_clean_text(label).upper() for label in example["choices"]["label"]]
    answer_key = _clean_text(example["answerKey"]).upper()

    option_lines = [
        f"{OPTION_LABELS[index]}. {choice_texts[index]}"
        for index in range(len(choice_texts))
    ]

    if answer_key in choice_labels:
        correct_index = choice_labels.index(answer_key)
        target_text = choice_texts[correct_index]
    elif answer_key.isdigit() and 0 < int(answer_key) <= len(choice_texts):
        target_text = choice_texts[int(answer_key) - 1]
    else:
        target_text = answer_key

    input_text = (
        "Task: Answer the multiple-choice science question.\n"
        f"Question: {question}\n"
        "Options:\n"
        + "\n".join(option_lines)
        + "\nAnswer:"
    )
    return input_text, target_text


def _format_boolq(
    example: dict[str, Any],
    seed: int,
    task_name: str,
    raw_split_name: str,
    raw_index: int,
) -> tuple[str, str]:
    del seed, task_name, raw_split_name, raw_index
    passage = _clean_text(example["passage"])
    question = _clean_text(example["question"])
    target_text = "yes" if bool(example["answer"]) else "no"
    input_text = (
        "Task: Answer the yes/no question based on the passage.\n"
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return input_text, target_text


def _format_gsm8k(
    example: dict[str, Any],
    seed: int,
    task_name: str,
    raw_split_name: str,
    raw_index: int,
) -> tuple[str, str]:
    del seed, task_name, raw_split_name, raw_index
    question = _clean_text(example["question"])
    target_text = _extract_gsm8k_final_answer(str(example["answer"]))
    input_text = (
        "Task: Solve the math word problem.\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return input_text, target_text


def _allocate_indices(
    total_size: int,
    output_splits: list[str],
    split_sizes: dict[str, int],
    seed: int,
    task_name: str,
    raw_split_name: str,
) -> dict[str, list[int]]:
    indices = list(range(total_size))
    rng = random.Random(_stable_seed(seed, task_name, raw_split_name))
    rng.shuffle(indices)

    allocated: dict[str, list[int]] = {}
    cursor = 0
    for output_split in output_splits:
        requested_size = split_sizes[output_split]
        available_size = max(total_size - cursor, 0)
        actual_size = min(requested_size, available_size)
        allocated[output_split] = sorted(indices[cursor: cursor + actual_size])
        cursor += actual_size
    return allocated


def _build_records(
    task_name: str,
    split_name: str,
    source_dataset: str,
    payloads: list[dict[str, str]],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for position, payload in enumerate(payloads, start=1):
        records.append(
            {
                "sample_id": f"{task_name}_{split_name}_{position:06d}",
                "task_name": task_name,
                "split": split_name,
                "input_text": payload["input_text"],
                "target_text": payload["target_text"],
                "source_dataset": source_dataset,
            }
        )
    return records


def _build_audit_rows(
    train_records: list[dict[str, str]],
    val_records: list[dict[str, str]],
    test_records: list[dict[str, str]],
) -> list[dict[str, str]]:
    audit_rows: list[dict[str, str]] = []
    for record in train_records + val_records + test_records:
        audit_rows.append(
            {
                "sample_id": record["sample_id"],
                "task_name": record["task_name"],
                "split": record["split"],
                "input_text": record["input_text"],
                "target_text": record["target_text"],
                "source_dataset": record["source_dataset"],
                "audit_status": "unchecked",
                "audit_note": "",
            }
        )
    return audit_rows


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_gsm8k_final_answer(answer_text: str) -> str:
    match = re.search(r"####\s*(.+)$", answer_text, flags=re.DOTALL)
    if match:
        return _clean_text(match.group(1)).replace(",", "")
    non_empty_lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    if non_empty_lines:
        return _clean_text(non_empty_lines[-1]).replace(",", "")
    return _clean_text(answer_text).replace(",", "")


def _clean_text(value: Any) -> str:
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def _stable_seed(*parts: Any) -> int:
    seed_text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def main() -> None:
    """Run preprocessing for all datasets."""
    config = get_config()
    print("PRE-PLAY preprocessing")
    print("----------------------")
    preprocess_all_datasets(config, force_reprocess=True)
    print("Preprocessing finished.")


if __name__ == "__main__":
    main()
