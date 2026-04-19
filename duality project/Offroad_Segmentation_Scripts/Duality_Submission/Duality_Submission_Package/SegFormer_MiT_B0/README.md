# SegFormer_MiT_B0

## Project summary

- **Model:** SegFormer (MiT-B0) via Hugging Face Transformers
- **Best validation IoU:** 0.3515 (epoch 10)
- **Final validation IoU:** 0.3511
- **Checkpoint size (approx.):** 14.3 MB
- **Parameters (approx.):** 3.72M

## Environment and dependencies

- **Python:** 3.10+ recommended.
- **Hardware:** CUDA recommended.
- **Install:**

```bash
cd SegFormer_MiT_B0
pip install -r requirements.txt
```

`requirements.txt` includes **`transformers`** and **`huggingface-hub`**. The first model download needs **network access** unless you use an offline cache.

## Dataset layout and paths

- **Train / val:** `Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/{train,val}/` with `Color_Images/` and `Segmentation/`.

Use **`OFFROAD_DUALITY_PROJECT`** / **`OFFROAD_SEGMENTATION_DATA_ROOT`** if path auto-discovery fails.

- **Test:** `Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/` (or env overrides in `test_segmentation6.py`).

## Step-by-step: train, test, and visualize

From `SegFormer_MiT_B0`:

1. **Train**

```bash
python train_segmentation6.py
```

2. **Test**

```bash
python test_segmentation6.py
```

By default **`--data_dir`** points at the **validation** split (not the separate test bundle). To evaluate on **`Offroad_Segmentation_testImages/...`**, pass `--data_dir` explicitly to that folder’s root (with `Color_Images` / `Segmentation` as produced by the dataset helpers).

Default weights: `segformer_best.pth`. Use script flags for alternate paths.

3. **Visualize**

Run **`test_segmentation6.py`** first with the default **`--output_dir predictions6`** (or keep masks under `predictions6/masks/`). Then:

```bash
python visualize6.py
```

This reads **`predictions6/masks/`** next to the script and writes colorized masks under **`predictions6/masks/colorized/`**.

## How to reproduce the reported results

1. Run **`python train_segmentation6.py`**.
2. Open **`train_stats6/evaluation_metrics.txt`** — compare best / final validation IoU to **0.3515** / **0.3511**.
3. Run **`python test_segmentation6.py`** and read **`test_evaluation_metrics.txt`** in the output directory.

Transformers and CUDA nondeterminism can change digits slightly between machines.

## Expected outputs and how to interpret them

| Artifact | Meaning |
|----------|--------|
| `segformer_best.pth` | Best validation-IoU weights. |
| `segformer_checkpoint.pth` | Full checkpoint when written by training. |
| `train_stats6/evaluation_metrics.txt` | Training-time validation metrics and history. |
| Test outputs | Predictions + **`test_evaluation_metrics.txt`**. |

**IoU** per class shows which semantic categories SegFormer handles well; mean IoU is the headline comparison number across models in this package.
