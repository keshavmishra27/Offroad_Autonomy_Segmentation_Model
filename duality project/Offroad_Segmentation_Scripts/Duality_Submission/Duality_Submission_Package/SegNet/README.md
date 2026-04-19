# SegNet

## Project summary

- **Model:** SegNet
- **Best validation IoU:** 0.4370 (epoch 15)
- **Final validation IoU:** 0.4370
- **Checkpoint size (approx.):** 112.5 MB
- **Parameters (approx.):** 29.46M

## Environment and dependencies

- **Python:** 3.10+ recommended.
- **Hardware:** CUDA recommended (mixed precision in training).
- **Install:**

```bash
cd SegNet
pip install -r requirements.txt
```

## Dataset layout and paths

Same pattern as `U-Net` / `FCN` in this package: **`config.py`** points **`TRAIN_DIR`**, **`VAL_DIR`**, and **`TEST_DIR`** under **`duality project`** (four directory levels above `SegNet`):

- `Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/train|val`
- `Offroad_Segmentation_testImages/Offroad_Segmentation_testImages`

`dataset.py` walks folders and pairs images to masks by filename. Customize **`config.py`** if your layout differs.

## Step-by-step: train and test

From `SegNet`:

1. **Train**

```bash
python train.py
```

2. **Test**

```bash
python test.py
```

Artifacts: **`runs/best_model.pth`**, **`runs/log.csv`**, **`runs/predictions/`** colorized outputs.

## How to reproduce the reported results

1. Run **`python train.py`** with default `config.py` (`EPOCHS`, `LR`, etc.).
2. In **`runs/log.csv`**, confirm peak **`val_mean_iou`** ≈ **0.4370** at epoch 15 for the reference run.
3. Run **`python test.py`**; console reports **Test Mean IoU** when test masks exist.

Exact reproduction is not guaranteed across GPUs; trends and peak epoch should match.

## Expected outputs and how to interpret them

| Artifact | Meaning |
|----------|--------|
| `runs/best_model.pth` | Best validation mean IoU checkpoint. |
| `runs/log.csv` | Training and validation metrics by epoch. |
| `runs/predictions/` | Per-image colored predictions. |

Use **`val_mean_iou`** in the CSV to compare against other classical models in this submission package.
