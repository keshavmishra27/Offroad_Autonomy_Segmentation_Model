# U-Net

## Project summary

- **Model:** U-Net
- **Best validation IoU:** 0.5118 (epoch 15)
- **Final validation IoU:** 0.5118
- **Checkpoint size (approx.):** 29.7 MB
- **Parameters (approx.):** 7.77M

## Environment and dependencies

- **Python:** 3.10+ recommended.
- **Hardware:** CUDA recommended.
- **Install:**

```bash
cd U-Net
pip install -r requirements.txt
```

`pandas` is used to write **`runs/log.csv`**.

## Dataset layout and paths

`config.py` sets **`_DATA_ROOT`** to the **`duality project`** directory (four parents above the `U-Net` folder). Expected relative paths:

- **Train:** `Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/train/`
- **Val:** `.../val/`
- **Test:** `Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/`

RGB and mask files are discovered recursively; each RGB image must have a mask with the **same basename**. Edit **`config.py`** if your data is stored elsewhere.

## Step-by-step: train and test

From `U-Net`:

1. **Train**

```bash
python train.py
```

2. **Test**

```bash
python test.py
```

Outputs:

- **`runs/best_model.pth`** — best validation IoU
- **`runs/log.csv`** — epoch history
- **`runs/predictions/*_pred.png`** — test predictions

## How to reproduce the reported results

1. Run **`python train.py`** without changing **`EPOCHS`** (15) and other defaults in **`config.py`** unless you are experimenting.
2. Verify in **`runs/log.csv`** that the best **`val_mean_iou`** matches **~0.5118** (epoch 15 in the reference run).
3. Run **`python test.py`** for qualitative masks and optional **Test Mean IoU** when ground-truth masks are present under the test root.

Minor numerical differences are normal on different hardware.

## Expected outputs and how to interpret them

| Artifact | Meaning |
|----------|--------|
| `runs/best_model.pth` | Model weights at highest validation mean IoU. |
| `runs/log.csv` | `train_loss`, `val_loss`, `train_mean_iou`, `val_mean_iou` per epoch. |
| `runs/predictions/` | Color-coded semantic segmentation for each test RGB input. |

**Val mean IoU** in the CSV is the main metric to compare with the summary at the top of this README.
