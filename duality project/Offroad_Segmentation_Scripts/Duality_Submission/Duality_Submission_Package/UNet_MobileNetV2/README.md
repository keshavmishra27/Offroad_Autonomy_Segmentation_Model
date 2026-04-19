# UNet_MobileNetV2

## Project summary

- **Model:** U-Net with MobileNetV2 encoder
- **Best validation IoU:** 0.4201 (epoch 9)
- **Final validation IoU:** 0.4112
- **Checkpoint size (approx.):** 76.3 MB
- **Parameters (approx.):** 6.67M

## Environment and dependencies

- **Python:** 3.10+ recommended.
- **Hardware:** CUDA recommended.
- **Install:**

```bash
cd UNet_MobileNetV2
pip install -r requirements.txt
```

## Dataset layout and paths

Same as other **`train_segmentation*.py`** entries:

- **Train / val:** `Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/{train,val}/` with `Color_Images/` and `Segmentation/`.

Set **`OFFROAD_DUALITY_PROJECT`** or **`OFFROAD_SEGMENTATION_DATA_ROOT`** if the script cannot find these folders.

- **Test:** `Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/` (override with **`OFFROAD_SEGMENTATION_TEST_IMAGES`** / **`OFFROAD_DUALITY_PROJECT`** if needed).

## Step-by-step: train, test, and visualize

From `UNet_MobileNetV2`:

1. **Train**

```bash
python train_segmentation3.py
```

2. **Test**

```bash
python test_segmentation3.py
```

Default checkpoint: **`unet_mobilenetv2_best.pth`**.

3. **Visualize**

Set `input_folder` in `visualize3.py` to your mask directory, then:

```bash
python visualize3.py
```

## How to reproduce the reported results

1. Run **`python train_segmentation3.py`**.
2. Read **`train_stats3/evaluation_metrics.txt`**: best validation IoU should be near **0.4201**, final near **0.4112** for the reference configuration.
3. Run **`python test_segmentation3.py`** and open **`test_evaluation_metrics.txt`** for held-out test numbers.

Stochastic training and library versions can shift metrics slightly.

## Expected outputs and how to interpret them

| Artifact | Meaning |
|----------|--------|
| `unet_mobilenetv2_best.pth` | Best-validation checkpoint. |
| `train_stats3/evaluation_metrics.txt` | Full training report (IoU, Dice, accuracy, per-class IoU). |
| Test script output dir | Predictions, plots, **`test_evaluation_metrics.txt`**. |

**Mean IoU** is the headline metric; use per-class IoU to see whether MobileNetV2’s representations favor certain terrain types.
