from __future__ import annotations

from pathlib import Path
from typing import Optional

from datasets import load_dataset

from config import AppConfig, get_config, get_raw_dataset_dir


DATASET_SPECS: dict[str, dict[str, Optional[str]]] = {
    "sciq": {
        "hf_name": "allenai/sciq",
        "hf_config": None,
    },
    "arc_challenge": {
        "hf_name": "allenai/ai2_arc",
        "hf_config": "ARC-Challenge",
    },
    "boolq": {
        "hf_name": "google/boolq",
        "hf_config": None,
    },
    "gsm8k": {
        "hf_name": "openai/gsm8k",
        "hf_config": "main",
    },
}


def is_dataset_saved(dataset_dir: Path) -> bool:
    """Check whether a dataset has already been saved locally."""
    return dataset_dir.exists() and (dataset_dir / "dataset_dict.json").exists()


def download_dataset(
    dataset_key: str,
    hf_name: str,
    hf_config: Optional[str],
    target_dir: Path,
) -> bool:
    """Download one dataset from Hugging Face and save it to disk."""
    if is_dataset_saved(target_dir):
        print(f"[SKIP] {dataset_key}: already exists at {target_dir}")
        return True

    print(f"[START] Downloading {dataset_key}")
    print(f"        Dataset: {hf_name}")
    if hf_config:
        print(f"        Config:  {hf_config}")
    print(f"        Target:  {target_dir}")

    try:
        if hf_config:
            dataset = load_dataset(hf_name, hf_config)
        else:
            dataset = load_dataset(hf_name)

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(target_dir))
        print(f"[DONE] {dataset_key}: saved to {target_dir}")
        return True
    except Exception as error:
        print(f"[ERROR] Failed to download {dataset_key}: {error}")
        return False


def download_raw_datasets(config: AppConfig) -> dict[str, bool]:
    """Download all configured raw datasets."""
    results: dict[str, bool] = {}
    print("PRE-PLAY dataset download")
    print("-------------------------")
    print(f"Raw data root: {config.paths.data_raw}")
    print()

    for dataset_key, spec in DATASET_SPECS.items():
        target_dir = get_raw_dataset_dir(config, dataset_key)
        success = download_dataset(
            dataset_key=dataset_key,
            hf_name=spec["hf_name"],
            hf_config=spec["hf_config"],
            target_dir=target_dir,
        )
        results[dataset_key] = success
        print()

    return results


def print_summary(results: dict[str, bool]) -> None:
    """Print a short summary after all download attempts."""
    print("Download summary:")
    for dataset_key, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  - {dataset_key}: {status}")
    print()


def main() -> None:
    """Run the dataset download script."""
    config = get_config()
    results = download_raw_datasets(config)
    print_summary(results)


if __name__ == "__main__":
    main()
