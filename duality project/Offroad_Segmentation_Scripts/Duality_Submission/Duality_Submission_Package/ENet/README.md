# ENet

## Project summary

- **Model:** ENet (efficient encoder–decoder)
- **Best validation IoU:** 0.2877 (epoch 11)
- **Final validation IoU:** 0.2877
- **Checkpoint size (approx.):** 4.5 MB
- **Parameters (approx.):** 0.36M

## Environment and dependencies

- **Python:** 3.10+ recommended.
- **Hardware:** CUDA recommended for training.
- **Install:**

```bash
cd ENet
pip install -r requirements.txt
```

## Dataset layout and paths

- **Train / val:** `Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/{train,val}/` with `Color_Images/` and `Segmentation/`.

Set **`OFFROAD_DUALITY_PROJECT`** or **`OFFROAD_SEGMENTATION_DATA_ROOT`** if automatic path search fails.

- **Test:** `Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/` (or env overrides as in `test_segmentation4.py`).

## Step-by-step: train, test, and visualize

From `ENet`:

1. **Train**

```bash
python train_segmentation4.py
```

2. **Test**

```bash
python test_segmentation4.py
```

Default checkpoint: `enet_best.pth`. Override with `--model_path` if needed.

3. **Visualize**

Update `input_folder` in `visualize4.py` for your machine, then:

```bash
python visualize4.py
```

## How to reproduce the reported results

1. Run `python train_segmentation4.py`.
2. Open **`train_stats4/evaluation_metrics.txt`** and compare best/final validation IoU to **0.2877**.
3. Run `python test_segmentation4.py` and read **`test_evaluation_metrics.txt`** in the output directory for test mean / per-class IoU.

Expect small run-to-run variation unless seeds and hardware are aligned with the original experiment.

## Expected outputs and how to interpret them

| Artifact | Meaning |
|----------|--------|
| `enet_best.pth` | Best model by validation IoU. |
| `train_stats4/evaluation_metrics.txt` | Training curves and best-epoch metrics. |
| Test script output | Predictions plus **`test_evaluation_metrics.txt`**. |

**IoU** is the primary segmentation metric here (mean over classes, each class IoU in \[0, 1\]).
