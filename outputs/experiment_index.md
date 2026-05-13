# Experiment Index

## Overview
This index tracks the experiment-level directory structure under `outputs/experiments/`, where each experiment id contains its nested run folders and one enriched `<exp_id>_summary.md` report.

## Directory Structure
```text
outputs/
├── experiments/
│   ├── exp001/
│   │   ├── exp001_no_replay_3tasks/
│   │   ├── exp001_replay_3tasks/
│   │   └── exp001_summary.md
│   ├── exp002/
│   │   ├── exp002_no_replay_reverse_3tasks/
│   │   ├── exp002_replay_reverse_3tasks/
│   │   └── exp002_summary.md
│   ├── exp003/
│   │   ├── exp003_no_replay_3tasks/
│   │   ├── exp003_replay_3tasks/
│   │   └── exp003_summary.md
│   ├── exp004/
│   │   ├── exp004_no_replay_3tasks/
│   │   ├── exp004_replay_buffer100_3tasks/
│   │   ├── exp004_replay_buffer30_3tasks/
│   │   ├── exp004_replay_buffer60_3tasks/
│   │   └── exp004_summary.md
│   ├── exp005/
│   │   ├── exp005_no_replay_seed123_3tasks/
│   │   ├── exp005_replay_buffer30_seed123_3tasks/
│   │   └── exp005_summary.md
│   ├── exp006/
│   │   ├── exp006_no_replay_seed42_3tasks/
│   │   ├── exp006_replay_loss_buffer30_seed42_3tasks/
│   │   ├── exp006_replay_random_buffer30_seed42_3tasks/
│   │   ├── exp006_replay_surprise_buffer30_seed42_3tasks/
│   │   ├── human_review/
│   │   ├── scorer_checkpoints/
│   │   ├── scorer_checkpoints_expanded/
│   │   ├── scorer_outputs/
│   │   ├── scorer_outputs_expanded/
│   │   ├── utility_labels/
│   │   └── exp006_summary.md
│   ├── exp007/
│   │   ├── exp007_no_replay_seed42_3tasks/
│   │   ├── exp007_replay_predicted_utility_buffer30_seed42_3tasks/
│   │   ├── exp007_replay_random_buffer30_seed42_3tasks/
│   │   ├── exp007_replay_surprise_buffer30_seed42_3tasks/
│   │   ├── exp007_predicted_utility_diagnostics.md
│   │   └── exp007_summary.md
│   ├── exp008/
│   │   ├── exp008_no_replay_seed42_3tasks/
│   │   ├── exp008_replay_predicted_utility_expanded_scorer_buffer30_seed42_3tasks/
│   │   ├── exp008_replay_predicted_utility_prev_scorer_buffer30_seed42_3tasks/
│   │   ├── exp008_replay_random_buffer30_seed42_3tasks/
│   │   ├── exp008_replay_surprise_buffer30_seed42_3tasks/
│   │   └── exp008_summary.md
│   └── exp009/
│       ├── exp009_no_replay_seed123_3tasks/
│       ├── exp009_replay_predicted_utility_expanded_scorer_buffer30_seed123_3tasks/
│       ├── exp009_replay_predicted_utility_prev_scorer_buffer30_seed123_3tasks/
│       ├── exp009_replay_random_buffer30_seed123_3tasks/
│       ├── exp009_replay_surprise_buffer30_seed123_3tasks/
│       └── exp009_summary.md
├── experiment_index.md
├── overall_experiment_analysis.md
└── preplay_project_status_report.md
```

## Experiment List
| Experiment ID | Description | Task Sequence | Replay Run Folder | No-Replay Run Folder | Experiment Directory | Summary File |
| --- | --- | --- | --- | --- | --- | --- |
| exp001 | a baseline continual learning experiment with a replay versus no-replay comparison | sciq -> arc_challenge -> boolq | exp001_replay_3tasks | exp001_no_replay_3tasks | `/Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp001` | [exp001_summary.md](experiments/exp001/exp001_summary.md) |
| exp002 | a task-order sensitivity experiment with a replay versus no-replay comparison | boolq -> arc_challenge -> sciq | exp002_replay_reverse_3tasks | exp002_no_replay_reverse_3tasks | `/Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp002` | [exp002_summary.md](experiments/exp002/exp002_summary.md) |
| exp003 | an additional baseline-style replay versus no-replay comparison using the original task order | sciq -> arc_challenge -> boolq | exp003_replay_3tasks | exp003_no_replay_3tasks | `/Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp003` | [exp003_summary.md](experiments/exp003/exp003_summary.md) |
| exp004 | a replay buffer size ablation experiment over the task sequence sciq -> arc_challenge -> boolq | sciq -> arc_challenge -> boolq | exp004_replay_buffer30_3tasks, exp004_replay_buffer60_3tasks, exp004_replay_buffer100_3tasks | exp004_no_replay_3tasks | `/Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp004` | [exp004_summary.md](experiments/exp004/exp004_summary.md) |
| exp005 | a seed robustness check comparing no replay with replay buffer size 30 over the original task order | sciq -> arc_challenge -> boolq | exp005_replay_buffer30_seed123_3tasks | exp005_no_replay_seed123_3tasks | `/Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp005` | [exp005_summary.md](experiments/exp005/exp005_summary.md) |
| exp006 | a replay selection strategy comparison experiment over no replay, random replay, loss replay, and surprise replay | sciq -> arc_challenge -> boolq | exp006_replay_random_buffer30_seed42_3tasks, exp006_replay_loss_buffer30_seed42_3tasks, exp006_replay_surprise_buffer30_seed42_3tasks | exp006_no_replay_seed42_3tasks | `/Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp006` | [exp006_summary.md](experiments/exp006/exp006_summary.md) |
| exp007 | a baseline experiment with a replay versus no-replay comparison | sciq -> arc_challenge -> boolq | exp007_replay_predicted_utility_buffer30_seed42_3tasks | exp007_no_replay_seed42_3tasks | `/Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp007` | [exp007_summary.md](experiments/exp007/exp007_summary.md) |
| exp008 | a baseline experiment with a replay versus no-replay comparison | sciq -> arc_challenge -> boolq | exp008_replay_predicted_utility_expanded_scorer_buffer30_seed42_3tasks | exp008_no_replay_seed42_3tasks | `/Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp008` | [exp008_summary.md](experiments/exp008/exp008_summary.md) |
| exp009 | a predicted_utility robustness check comparing previous and expanded scorer sources under seed 123 | sciq -> arc_challenge -> boolq | exp009_replay_random_buffer30_seed123_3tasks, exp009_replay_surprise_buffer30_seed123_3tasks, exp009_replay_predicted_utility_prev_scorer_buffer30_seed123_3tasks, exp009_replay_predicted_utility_expanded_scorer_buffer30_seed123_3tasks | exp009_no_replay_seed123_3tasks | `/Users/chenxiangkai/PycharmProjects/Pre_play/outputs/experiments/exp009` | [exp009_summary.md](experiments/exp009/exp009_summary.md) |

## Comparison Summary

### exp001 Comparison

- Description: a baseline continual learning experiment with a replay versus no-replay comparison.
- Task sequence: `sciq -> arc_challenge -> boolq`
- Replay run folder: `exp001_replay_3tasks`
- No-replay run folder: `exp001_no_replay_3tasks`

| Task | Replay | No Replay | Delta | Better Setting |
| --- | --- | --- | --- | --- |
| sciq | 0.3200 | 0.5000 | -0.1800 | No Replay |
| arc_challenge | 0.2600 | 0.2200 | 0.0400 | Replay |
| boolq | 0.5000 | 0.4600 | 0.0400 | Replay |

- Replay reduced final validation accuracy on average by 0.0111 across the saved tasks.
- Replay reduced final test accuracy on average by 0.0333 across the saved tasks.
- The maximum saved forgetting was identical in both settings at 0.1000.

Full report: [exp001_summary.md](experiments/exp001/exp001_summary.md)
### exp002 Comparison

- Description: a task-order sensitivity experiment with a replay versus no-replay comparison.
- Task sequence: `boolq -> arc_challenge -> sciq`
- Replay run folder: `exp002_replay_reverse_3tasks`
- No-replay run folder: `exp002_no_replay_reverse_3tasks`

| Task | Replay | No Replay | Delta | Better Setting |
| --- | --- | --- | --- | --- |
| boolq | 0.5200 | 0.4800 | 0.0400 | Replay |
| arc_challenge | 0.2200 | 0.1600 | 0.0600 | Replay |
| sciq | 0.4200 | 0.4800 | -0.0600 | No Replay |

- Replay reduced final validation accuracy on average by 0.0111 across the saved tasks.
- Replay improved final test accuracy on average by 0.0133 across the saved tasks.
- The maximum saved forgetting was identical in both settings at 0.0667.

Full report: [exp002_summary.md](experiments/exp002/exp002_summary.md)
### exp003 Comparison

- Description: an additional baseline-style replay versus no-replay comparison using the original task order.
- Task sequence: `sciq -> arc_challenge -> boolq`
- Replay run folder: `exp003_replay_3tasks`
- No-replay run folder: `exp003_no_replay_3tasks`

| Task | Replay | No Replay | Delta | Better Setting |
| --- | --- | --- | --- | --- |
| sciq | 0.3200 | 0.5000 | -0.1800 | No Replay |
| arc_challenge | 0.2800 | 0.2200 | 0.0600 | Replay |
| boolq | 0.5000 | 0.4600 | 0.0400 | Replay |

- Replay reduced final validation accuracy on average by 0.0111 across the saved tasks.
- Replay reduced final test accuracy on average by 0.0267 across the saved tasks.
- The maximum saved forgetting was identical in both settings at 0.1000.

Full report: [exp003_summary.md](experiments/exp003/exp003_summary.md)
### exp004 Comparison

- Description: a replay buffer size ablation experiment over the task sequence sciq -> arc_challenge -> boolq.
- Task sequence: `sciq -> arc_challenge -> boolq`
- No Replay run folder: `exp004_no_replay_3tasks`
- Buffer 30 run folder: `exp004_replay_buffer30_3tasks`
- Buffer 60 run folder: `exp004_replay_buffer60_3tasks`
- Buffer 100 run folder: `exp004_replay_buffer100_3tasks`

| Task | No Replay | Buffer 30 | Buffer 60 | Buffer 100 | Best Setting |
| --- | --- | --- | --- | --- | --- |
| sciq | 0.5000 | 0.3800 | 0.3200 | 0.3200 | No Replay |
| arc_challenge | 0.2200 | 0.2600 | 0.2800 | 0.2800 | Tie (Buffer 60, Buffer 100) |
| boolq | 0.4600 | 0.4600 | 0.5000 | 0.5000 | Tie (Buffer 60, Buffer 100) |

- The saved ablation covers the task sequence `sciq -> arc_challenge -> boolq` across one no-replay baseline and three replay buffer sizes.
- Among replay settings, the best average validation accuracy came from Buffer 30.
- Among replay settings, the best average test accuracy came from Tie (Buffer 30, Buffer 60, Buffer 100).
- Among replay settings, the lowest maximum forgetting came from Buffer 30.

Full report: [exp004_summary.md](experiments/exp004/exp004_summary.md)
### exp005 Comparison

- Description: a seed robustness check comparing no replay with replay buffer size 30 over the original task order.
- Task sequence: `sciq -> arc_challenge -> boolq`
- Replay run folder: `exp005_replay_buffer30_seed123_3tasks`
- No-replay run folder: `exp005_no_replay_seed123_3tasks`

| Task | Replay | No Replay | Delta | Better Setting |
| --- | --- | --- | --- | --- |
| sciq | 0.3200 | 0.2400 | 0.0800 | Replay |
| arc_challenge | 0.2400 | 0.0000 | 0.2400 | Replay |
| boolq | 0.4800 | 0.5000 | -0.0200 | No Replay |

- Replay improved final validation accuracy on average by 0.1222 across the saved tasks.
- Replay improved final test accuracy on average by 0.1000 across the saved tasks.
- Replay forgot less on the worst affected task, reducing maximum saved forgetting from 0.3333 to 0.2333.

Full report: [exp005_summary.md](experiments/exp005/exp005_summary.md)
### exp006 Comparison

- Description: a replay selection strategy comparison experiment over no replay, random replay, loss replay, and surprise replay.
- Task sequence: `sciq -> arc_challenge -> boolq`
- No Replay run folder: `exp006_no_replay_seed42_3tasks`
- Random Replay run folder: `exp006_replay_random_buffer30_seed42_3tasks`
- Loss Replay run folder: `exp006_replay_loss_buffer30_seed42_3tasks`
- Surprise Replay run folder: `exp006_replay_surprise_buffer30_seed42_3tasks`

| Task | No Replay | Random | Loss | Surprise | Best Setting |
| --- | --- | --- | --- | --- | --- |
| sciq | 0.4000 | 0.3400 | 0.3800 | 0.3800 | No Replay |
| arc_challenge | 0.1600 | 0.2800 | 0.2600 | 0.2600 | Random |
| boolq | 0.4800 | 0.4600 | 0.4600 | 0.4600 | No Replay |

- The saved strategy comparison covers `sciq -> arc_challenge -> boolq` with one no-replay baseline and three replay selection strategies at buffer size `30` and seed `42`.
- The best average validation setting was Tie (No Replay, Loss Replay, Surprise Replay).
- The best average test setting was Tie (Loss Replay, Surprise Replay).
- The setting with the lowest maximum forgetting was No Replay.

Full report: [exp006_summary.md](experiments/exp006/exp006_summary.md)
### exp007 Comparison

- Description: a baseline experiment with a replay versus no-replay comparison.
- Task sequence: `sciq -> arc_challenge -> boolq`
- Replay run folder: `exp007_replay_predicted_utility_buffer30_seed42_3tasks`
- No-replay run folder: `exp007_no_replay_seed42_3tasks`

| Task | Replay | No Replay | Delta | Better Setting |
| --- | --- | --- | --- | --- |
| sciq | 0.2800 | 0.4000 | -0.1200 | No Replay |
| arc_challenge | 0.2400 | 0.1600 | 0.0800 | Replay |
| boolq | 0.4800 | 0.4800 | 0.0000 | Tie |

- Replay improved final validation accuracy on average by 0.0222 across the saved tasks.
- Replay reduced final test accuracy on average by 0.0133 across the saved tasks.
- No measurable forgetting is visible in the saved final validation metrics for either setting.

Full report: [exp007_summary.md](experiments/exp007/exp007_summary.md)
### exp008 Comparison

- Description: a baseline experiment with a replay versus no-replay comparison.
- Task sequence: `sciq -> arc_challenge -> boolq`
- Replay run folder: `exp008_replay_predicted_utility_expanded_scorer_buffer30_seed42_3tasks`
- No-replay run folder: `exp008_no_replay_seed42_3tasks`

| Task | Replay | No Replay | Delta | Better Setting |
| --- | --- | --- | --- | --- |
| sciq | 0.4800 | 0.4000 | 0.0800 | Replay |
| arc_challenge | 0.3200 | 0.1800 | 0.1400 | Replay |
| boolq | 0.4800 | 0.4800 | 0.0000 | Tie |

- Replay reduced final validation accuracy on average by 0.0333 across the saved tasks.
- Replay improved final test accuracy on average by 0.0733 across the saved tasks.
- Replay forgot more on the worst affected task, increasing maximum saved forgetting from 0.0000 to 0.1333.

Full report: [exp008_summary.md](experiments/exp008/exp008_summary.md)
### exp009 Comparison

- Description: a predicted_utility robustness check comparing previous and expanded scorer sources under seed 123.
- Task sequence: `sciq -> arc_challenge -> boolq`
- No-replay run folder: `exp009_no_replay_seed123_3tasks`
- Random run folder: `exp009_replay_random_buffer30_seed123_3tasks`
- Surprise run folder: `exp009_replay_surprise_buffer30_seed123_3tasks`
- Predicted Previous run folder: `exp009_replay_predicted_utility_prev_scorer_buffer30_seed123_3tasks`
- Predicted Expanded run folder: `exp009_replay_predicted_utility_expanded_scorer_buffer30_seed123_3tasks`

| Task | No Replay | Random | Surprise | Predicted Previous | Predicted Expanded | Best Setting |
| --- | --- | --- | --- | --- | --- | --- |
| sciq | 0.3000 | 0.4000 | 0.3600 | 0.3400 | 0.4000 | Tie (Random, Predicted Expanded) |
| arc_challenge | 0.1600 | 0.2000 | 0.1800 | 0.2400 | 0.1800 | Predicted Previous |
| boolq | 0.4800 | 0.4800 | 0.4800 | 0.4800 | 0.4800 | Tie (All Settings) |

- The best average validation setting is Predicted Expanded at `0.4111`.
- The best average test setting is Random at `0.3600`.
- The setting with the lowest maximum forgetting is Predicted Expanded at `0.0333`.
- The scorer-source pattern from `exp008` did not hold under seed `123`, so the saved predicted_utility result remains seed-sensitive and exploratory.

Full report: [exp009_summary.md](experiments/exp009/exp009_summary.md)

## Overall Notes
Across the available experiment pairs, replay was mixed overall. The average final test delta was 0.0085, but the direction of the effect varied by task or experiment.
