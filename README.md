# PRE-PLAY Prototype

This project is a small continual fine-tuning prototype inspired by PRE-PLAY style replay for sequential task learning. It uses fixed local subsets of SciQ, ARC-Challenge, BoolQ, and GSM8K, trains `google/flan-t5-small` across tasks, tracks forgetting, scores per-sample surprise with loss, and stores top-scoring samples in a replay buffer.

## Project layout

- `data/raw/`: raw Hugging Face datasets saved locally with `save_to_disk`
- `data/processed/`: processed local JSONL files used by training and evaluation
- `data/audit/`: human-readable audit CSV files
- `outputs/final_review/final_reviewed_dataset.csv`: final audited replay review CSV used as optional review provenance, not as the default train/val/test benchmark source
- `outputs/experiments/<exp_id>/<run_name>/`: unified output folder for each future experiment
- `outputs/experiment_index.md`: index of legacy runs and the current output standard

## Experiment output structure

Each future run is saved under:

```text
outputs/experiments/<exp_id>/<run_name>/
├── checkpoints/
├── logs/
├── metrics/
├── replay_scores/
├── replay_buffers/
├── sample_signals/
├── utility_labels/
├── human_review/
├── replay_selection_comparison/
└── run_config.json
```

- `checkpoints/`: saved model checkpoints after each task
- `logs/`: training summaries and step logs
- `metrics/`: validation and test metrics plus prediction CSV files
- `replay_scores/`: per-sample replay scoring outputs
- `replay_buffers/`: replay buffer snapshots and replay buffer summaries
- `sample_signals/`: sample-level early training signal logs for each task
- `utility_labels/`: proxy replay utility labels derived from replay selection outcomes
- `human_review/`: auto-tagged replay review exports for manual inspection, including original candidate CSVs and reviewed copies
- `replay_selection_comparison/`: per-task replay selection score exports
- `run_config.json`: saved run configuration for reproducibility
- `<exp_id>_summary.md`: experiment-level summary with final metrics and replay artifact checks

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

1. Download the raw datasets once:

```bash
python data_download.py
```

2. Build local processed JSONL files and audit CSV files:

```bash
python preprocess.py
```

3. Run continual training using local processed JSONL files:

```bash
python main.py
```

Recommended formal experiment names start with `expXXX_`, for example:

```bash
python main.py --run-name exp003_replay_3tasks --tasks sciq arc_challenge boolq
python main.py --run-name exp003_no_replay_3tasks --tasks sciq arc_challenge boolq --no-replay
```

By default, experiments read train/validation/test samples from `data/processed/<task>/{train,val,test}.jsonl`.
To attach the final audited replay review CSV to a run without replacing the train/validation/test files, use:

```bash
python main.py \
  --run-name exp011_scored_replay_3tasks \
  --tasks sciq arc_challenge boolq \
  --data-dir data/processed \
  --use-audited-review \
  --audited-review-csv outputs/final_review/final_reviewed_dataset.csv
```

The current `final_reviewed_dataset.csv` contains reviewed train-split replay/utility metadata only. It should be recorded as an audited replay review source unless a future audited train/validation/test dataset is created explicitly.

You can also run a subset of tasks:

```bash
python main.py --tasks sciq boolq
```

## Comparing replay and no-replay runs

Use paired run names and compare the files under:

- `outputs/experiments/<exp_id>/<run_name>/metrics/val_metrics.csv`
- `outputs/experiments/<exp_id>/<run_name>/metrics/test_metrics.csv`
- `outputs/experiments/<exp_id>/<exp_id>_summary.md`

A typical comparison is:

- one replay-enabled run such as `exp003_replay_3tasks`
- one no-replay baseline such as `exp003_no_replay_3tasks`

## Notes

- `main.py` does not download datasets automatically.
- `data_download.py` is the only file that downloads from Hugging Face.
- Training reads local processed JSONL by default. `--data-dir` can point to another processed-style directory with `<task>/{train,val,test}.jsonl`.
- `--use-audited-review` records `outputs/final_review/final_reviewed_dataset.csv` in `run_config.json` as audited replay review provenance. It does not replace validation or test data when the CSV does not contain complete train/val/test splits.
- Experiments `exp001` through `exp010` are preliminary experiments using `data/processed` JSONL as the train/validation/test data source.
- Later final/main experiments should explicitly record the audited review CSV when replay selection or utility review conclusions depend on it.
- Processed splits are written as `train.jsonl`, `val.jsonl`, and `test.jsonl`.
- Replay selection uses per-sample loss as a simple surprise score.
- Legacy runs under `outputs/checkpoints/`, `outputs/logs/`, `outputs/metrics/`, and `outputs/replay_scores/` are preserved and are not moved automatically.

## PRE-PLAY core extensions

- `sample_signals/` stores per-sample early loss and surprise-style logging for future runs.
- `utility_labels/` stores proxy labels where replay selection is treated as a lightweight stand-in for utility.
- `human_review/` writes `replay_review_candidates.csv` as the original auto-generated export and `replay_review_candidates_reviewed.csv` as the manual-edit copy; keep the original unchanged for reproducibility and edit only the reviewed copy.
- `utility_scorer.py` trains a lightweight scikit-learn scorer from proxy labels or, when explicitly requested, human review labels.
- `--replay-selection-strategy` supports `random`, `loss`, `surprise`, and `predicted_utility`.
- Proxy labels are not exact future replay utility; they only approximate usefulness from replay selection or optional manual review.
