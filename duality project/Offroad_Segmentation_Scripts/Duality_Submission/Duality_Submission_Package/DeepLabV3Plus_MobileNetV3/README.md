# DeepLabV3Plus_MobileNetV3

## Project summary

- **Model:** DeepLabV3+ with MobileNetV3 backbone
- **Best validation IoU:** 0.3858 (epoch 11)
- **Final validation IoU:** 0.3858
- **Checkpoint size (approx.):** 42.3 MB
- **Parameters (approx.):** 11.05M

## Environment and dependencies

- **Python:** 3.10+ recommended.
- **Hardware:** CUDA recommended for training.
- **Install:**

```bash
cd DeepLabV3Plus_MobileNetV3
pip install -r requirements.txt
```

Use a PyTorch build that matches your system from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) if needed.

## Dataset layout and paths

Expect the standard **off-road** dataset under **`duality project`**:

- Train: `Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/train/` (`Color_Images/`, `Segmentation/`).
- Val: `.../val/` with the same subfolders.

Set **`OFFROAD_DUALITY_PROJECT`** (or **`OFFROAD_SEGMENTATION_DATA_ROOT`**) to the `duality project` directory if paths are not found automatically.

**Test images:** `Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/`. Override with **`OFFROAD_SEGMENTATION_TEST_IMAGES`** or **`OFFROAD_DUALITY_PROJECT`** when required.

## Step-by-step: train, test, and visualize

From `DeepLabV3Plus_MobileNetV3`:

1. **Train**

```bash
python train_segmentation2.py
```

2. **Test / inference**

```bash
python test_segmentation2.py
```

Default weights path is `deeplabv3plus_segmentation.pth` in this folder; use `--model_path` to point elsewhere.

3. **Visualize (optional)**

Edit the `input_folder` path at the top of `visualize2.py` to point at your mask directory (same layout as the author’s test `Segmentation` folder), then:

```bash
python visualize2.py
```

## How to reproduce the reported results

1. Run **`python train_segmentation2.py`** on the same dataset layout.
2. Open **`train_stats2/evaluation_metrics.txt`** and compare **Best Val IoU** / **Final Val IoU** to this README (0.3858).
3. Run **`python test_segmentation2.py`** for test-time metrics; read **`evaluation_metrics.txt`** under the output directory printed by the script.

Slight numeric drift across machines and PyTorch versions is normal.

## Expected outputs and how to interpret them

| Artifact | Meaning |
|----------|--------|
| `deeplabv3plus_segmentation.pth` | Saved model weights for deployment / testing. |
| `deeplabv3plus_checkpoint.pth` | Optional full training checkpoint (optimizer state, etc.), if saved. |
| `train_stats2/evaluation_metrics.txt` | Validation-centric training report (IoU, Dice, accuracy, per-class IoU). |
| `test_segmentation2.py` | Writes predictions and **`evaluation_metrics.txt`** in the chosen output folder. |

**IoU:** higher is better. Per-class IoU shows which semantic classes (vegetation, rocks, sky, etc.) are well segmented vs. confused.
