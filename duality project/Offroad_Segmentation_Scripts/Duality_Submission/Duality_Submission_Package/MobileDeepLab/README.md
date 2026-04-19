# MobileDeepLab

## Project summary

- **Model:** MobileDeepLab-style segmentation (see `model.py` in this folder)
- **Best validation IoU:** 0.5919 (epoch 48)
- **Final validation IoU:** 0.5916
- **Checkpoint size (approx.):** 18.1 MB
- **Parameters (approx.):** 4.71M

## Environment and dependencies

- **Python:** 3.10+ recommended.
- **Hardware:** CUDA recommended.
- **Install:**

```bash
cd MobileDeepLab
pip install -r requirements.txt
```

## Dataset layout and paths

`config.py` resolves **`duality project`** as **four parents** above `MobileDeepLab`, then:

- **Train:** `Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/train/`
- **Val:** `.../val/`
- **Test:** `Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/`

`dataset.py` discovers RGB / mask files by walking the tree. Adjust **`config.py`** if your dataset root differs.

## Step-by-step: train and test

From `MobileDeepLab`:

1. **Train**

```bash
python train.py
```

2. **Test**

```bash
python test.py
```

Best weights are expected at **`runs/best_model.pth`**. Predictions are written to **`runs/predictions/`**.

## How to reproduce the reported results

1. Run **`python train.py`** with default `config.py` settings.
2. Inspect **`runs/log.csv`**: find the epoch with maximum **`val_mean_iou`** (reference **~0.5919**) and the last epoch’s validation IoU (**~0.5916**).
3. Run **`python test.py`** for test-time outputs; if the test folder includes paired masks, the script prints **Test Mean IoU**.

Small differences vs. this README are acceptable across hardware and library versions.

## Expected outputs and how to interpret them

| Artifact | Meaning |
|----------|--------|
| `runs/best_model.pth` | Weights at the best validation mean IoU. |
| `runs/log.csv` | Epoch-wise train/val loss and mean IoU. |
| `runs/predictions/*.png` | Colorized semantic masks (`*_pred.png`). |

**Mean IoU** averages intersection-over-union across classes; use **`log.csv`** to see whether the model peaked early or kept improving through epoch 48.
