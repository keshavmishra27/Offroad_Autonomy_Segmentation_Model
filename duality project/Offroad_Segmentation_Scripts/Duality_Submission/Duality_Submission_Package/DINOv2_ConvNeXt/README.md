# DINOv2_ConvNeXt

## Project summary

- **Model:** DINOv2 backbone + ConvNeXt-style segmentation head
- **Best validation IoU:** 0.4475 (epoch 11)
- **Final validation IoU:** 0.4475
- **Checkpoint size (approx.):** 4.5 MB (head weights; backbone loaded separately)
- **Parameters (approx.):** 1.18M (head)

## Environment and dependencies

- **Python:** 3.10+ recommended.
- **Hardware:** CUDA recommended; training uses mixed-precision style workloads typical of segmentation.
- **Install:**

```bash
cd DINOv2_ConvNeXt
pip install -r requirements.txt
```

- **DINOv2 backbone:** `train_segmentation1.py` / `test_segmentation1.py` load the backbone via **`torch.hub.load("facebookresearch/dinov2", ...)`**. The **first run requires internet** to cache the repository and weights; firewall or offline environments must pre-cache the hub directory.

## Dataset layout and paths

Same convention as other `train_segmentation*.py` models:

- **Train / val:** `Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/{train,val}/` with `Color_Images/` and `Segmentation/`.

If discovery fails, set **`OFFROAD_DUALITY_PROJECT`** to the folder that contains `Offroad_Segmentation_Training_Dataset`.

**Test:** `Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/` or override with **`OFFROAD_SEGMENTATION_TEST_IMAGES`** / **`OFFROAD_DUALITY_PROJECT`**.

## Step-by-step: train, test, and visualize

From `DINOv2_ConvNeXt`:

1. **Train** (ensure hub access on first run)

```bash
python train_segmentation1.py
```

2. **Test**

```bash
python test_segmentation1.py
```

Optional: `--model_path` (default `segmentation_head.pth`), `--data_dir`, `--output_dir`, `--batch_size`, `--num_samples`.

3. **Visualize**

Edit `input_folder` in `visualize1.py` to your local masks directory, then:

```bash
python visualize1.py
```

## How to reproduce the reported results

1. Train with `python train_segmentation1.py`.
2. Read **`train_stats/evaluation_metrics.txt`** — align **Best Val IoU** / **Final Val IoU** with 0.4475 in this README.
3. For test evaluation, run `python test_segmentation1.py` and inspect **`predictions/evaluation_metrics.txt`** (default output layout) for mean and per-class IoU.

Backbone and batch effects mean numbers may not match exactly on a different GPU or PyTorch release.

## Expected outputs and how to interpret them

| Artifact | Meaning |
|----------|--------|
| `segmentation_head.pth` | Trained segmentation head (used with the hub DINOv2 backbone). |
| `segmentation_checkpoint.pth` | Full training checkpoint when saved by the script. |
| `train_stats/evaluation_metrics.txt` | Rich training log: final/best metrics, per-class IoU at best epoch, epoch history. |
| `predictions/` (from test) | Comparison PNGs, **`evaluation_metrics.txt`**, **`per_class_metrics.png`**. |

**Mean IoU** summarizes overall segmentation quality; **per-class IoU** explains which terrain categories drive the score (e.g. sky often IoU-high, thin structures lower).
