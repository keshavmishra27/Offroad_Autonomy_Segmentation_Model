# FCN

## Project summary

- **Model:** Fully Convolutional Network (**default variant: `fcn8s`** per `config.py`)
- **Best validation IoU:** 0.3119 (epoch 45)
- **Final validation IoU:** 0.3114
- **Checkpoint size (approx.):** 128.5 MB
- **Parameters (approx.):** 33.67M

## Environment and dependencies

- **Python:** 3.10+ recommended.
- **Hardware:** CUDA recommended (`train.py` / `test.py` use AMP on CUDA).
- **Install:**

```bash
cd FCN
pip install -r requirements.txt
```

`pandas` is required for `runs/log_*.csv`.

## Dataset layout and paths

This package uses **`config.py`**: paths are built from **four parent directories** above the `FCN` folder, landing on the **`duality project`** root, then:

- **Train:** `Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/train/`
- **Val:** `.../val/`
- **Test:** `Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/`

`dataset.py` walks these trees and pairs RGB images with masks by **basename** (folders may use naming like `Color_Images` / `Segmentation` as long as pairing works). If your data lives elsewhere, edit **`config.py`** and set `_DATA_ROOT` / `TRAIN_DIR` / `VAL_DIR` / `TEST_DIR` explicitly.

## Step-by-step: train and test

From `FCN`:

1. **Train (default fcn8s)**

```bash
python train.py
```

Other variants:

```bash
python train.py --model fcn16s
python train.py --model fcn32s
```

2. **Test / inference**

```bash
python test.py
```

Match the variant:

```bash
python test.py --model fcn8s
```

Weights are read from **`runs/best_<model>.pth`** (e.g. `runs/best_fcn8s.pth`). Predictions go to **`runs/predictions/<model>/`**.

## How to reproduce the reported results

1. Keep **`config.py`** hyperparameters (`EPOCHS`, `LR`, `BATCH_SIZE`, `MODEL_VARIANT`, etc.) unchanged unless you intentionally ablate.
2. Run **`python train.py`** (or with `--model fcn8s` explicitly).
3. Track validation IoU per epoch in **`runs/log_fcn8s.csv`**; the **maximum `val_mean_iou`** should align with **~0.3119** best and **~0.3114** final row for the original run.
4. Run **`python test.py`** for colored masks and optional test IoU when ground-truth masks exist alongside test RGB.

FCN uses SGD + step LR; different GPUs can still produce minor metric drift.

## Expected outputs and how to interpret them

| Artifact | Meaning |
|----------|--------|
| `runs/best_fcn8s.pth` (or `fcn16s` / `fcn32s`) | Best validation-IoU weights for the chosen variant. |
| `runs/log_<model>.csv` | Per-epoch train/val loss and mean IoU. |
| `runs/predictions/<model>/` | Colorized `*_pred.png` masks for each test image. |

Console prints **mean IoU** on the test set when masks are available; otherwise only inference time and saved predictions are reported.
