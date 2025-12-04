# DETR Moved-Object Fine-Tuning

This repo implements Assignment 3 (Option 2) for the VIRAT frame-pair dataset using pixel-difference inputs and DETR.

## Environment Setup

```bash
python3 -m venv .venv          # or use conda
source .venv/bin/activate
pip install -r requirements.txt
```

The listed versions match the ones used to validate the pipeline on Apple Silicon. Adjust if you have an existing PyTorch install.

## Data Preparation Recap

From the project root:

```bash
# 1) Generate motion-only annotations (writes to cv_data_hw2/matched_annotations)
python cv_data_hw2/data_ground_truth_labeller.py

# 2) Build the metadata catalog used by the Dataset class
python cv_data_hw2/build_pairs_metadata.py
```

`cv_data_hw2/metadata/pairs.csv` now contains every VIRAT pair and the corresponding matched annotation path.

## Training DETR Regimes

All fine-tuning experiments share the same entry point:

```bash
python train_detr_moved_objects.py \
  --regime <full|backbone|head|transformer> \
  --run-name <tag_for_outputs> \
  --epochs 15 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --metadata-csv $(pwd)/cv_data_hw2/metadata/pairs.csv \
  --output-dir outputs
```

The `--metadata-csv` flag accepts any path; by default the script already resolves to the copy inside this repo using an absolute path derived from the script directory. Use `$(pwd)` (or your project root) if you run the command from elsewhere. The script automatically splits the metadata 80/20, builds the pixel-diff dataset, logs losses, and saves `best_model.pt`, `latest_model.pt`, and `training_log.json` under `outputs/<run-name>/`.

### Recommended commands

| Regime | Command |
| --- | --- |
| Full fine-tune | `python train_detr_moved_objects.py --regime full --run-name full_finetune --epochs 20 --batch-size 2 --learning-rate 5e-5` |
| Backbone only | `python train_detr_moved_objects.py --regime backbone --run-name backbone_only --epochs 20 --batch-size 2 --learning-rate 1e-4` |
| Transformer block only | `python train_detr_moved_objects.py --regime transformer --run-name transformer_only --epochs 20 --batch-size 2 --learning-rate 1e-4` |
| Head only | `python train_detr_moved_objects.py --regime head --run-name head_only --epochs 20 --batch-size 2 --learning-rate 2e-4` |

Feel free to adjust hyperparameters per experiment; the script records them in the log file for reporting.

### Useful switches

- `--train-only-moved`: filters the dataset to frame pairs that actually have motion annotations.
- `--target-long-side`: change the resize target (default 800px).
- `--device`: force `cpu` if you train locally without GPU.

## Outputs

- `outputs/<run-name>/best_model.pt`: weights with the lowest validation loss.
- `outputs/<run-name>/latest_model.pt`: last-epoch weights.
- `outputs/<run-name>/training_log.json`: regime, hyperparameters, dataset sizes, and epoch-wise losses—use this for SLURM logs/report figures.

## Evaluation & Visuals

After training a regime, run the evaluation helper to compute precision/recall/F1 (IoU ≥ 0.5) on the held-out test split and produce qualitative figures:

```bash
python evaluate_detr_moved_objects.py \
  --checkpoint outputs/full_finetune/best_model.pt \
  --regime full \
  --metadata-csv $(pwd)/cv_data_hw2/metadata/pairs.csv \
  --device cuda \
  --max-plots 12
```

This script writes `outputs/<regime>_metrics.json` with per-class TP/FP/FN, precision, recall, and F1; plots showing frame1 (with GT boxes), frame2, and the pixel-diff image with predictions are saved under `visuals/<regime>/sample_XXXX.png`. Repeat for each trained regime to gather the tables and screenshots required for the report.

