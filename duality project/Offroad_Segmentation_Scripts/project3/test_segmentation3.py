"""
Segmentation Test / Inference Script — Project 3 (U-Net + MobileNetV2)
Evaluates a trained U-Net + MobileNetV2 model on validation/test data and saves predictions.

Usage:
  python test_segmentation3.py
  python test_segmentation3.py --model_path unet_mobilenetv2_best.pth --data_dir <path>
  python test_segmentation3.py --num_samples 20 --batch_size 8

Outputs (in predictions3/):
  masks/           — raw class-ID prediction masks (0-9 per pixel)
  masks_color/     — colourised RGB prediction masks
  comparisons/     — side-by-side input | GT | prediction images
  test_evaluation_metrics.txt
  per_class_metrics.png

Requirements:
  pip install segmentation-models-pytorch
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("ERROR: This script requires 'segmentation-models-pytorch'.")
    print("Install with:  pip install segmentation-models-pytorch")
    raise

plt.switch_backend('Agg')


# ============================================================================
# Shared class definitions (same as train_segmentation3.py)
# ============================================================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9,
}
n_classes = len(value_map)

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

color_palette = np.array([
    [0,   0,   0  ],   # Background – black
    [34,  139, 34 ],   # Trees – forest green
    [0,   255, 0  ],   # Lush Bushes – lime
    [210, 180, 140],   # Dry Grass – tan
    [139, 90,  43 ],   # Dry Bushes – brown
    [128, 128, 0  ],   # Ground Clutter – olive
    [139, 69,  19 ],   # Logs – saddle brown
    [128, 128, 128],   # Rocks – gray
    [160, 82,  45 ],   # Landscape – sienna
    [135, 206, 235],   # Sky – sky blue
], dtype=np.uint8)

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


def convert_mask(mask):
    arr     = np.array(mask, dtype=np.int32)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    return Image.fromarray(new_arr)


def mask_to_color(mask_np):
    h, w = mask_np.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_classes):
        rgb[mask_np == c] = color_palette[c]
    return rgb


# ============================================================================
# Dataset
# ============================================================================

class OffRoadTestDataset(Dataset):
    def __init__(self, data_dir, img_size):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.img_size  = img_size   # (h, w)
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.mean      = torch.tensor(IMG_MEAN).view(3, 1, 1)
        self.std       = torch.tensor(IMG_STD ).view(3, 1, 1)

    def __len__(self): return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        h, w    = self.img_size

        image = Image.open(os.path.join(self.image_dir, data_id)).convert('RGB')
        image = image.resize((w, h), Image.BILINEAR)

        mask  = Image.open(os.path.join(self.masks_dir, data_id))
        mask  = convert_mask(mask)
        mask  = mask.resize((w, h), Image.NEAREST)

        image_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_t = (image_t - self.mean) / self.std
        mask_t  = torch.from_numpy(np.array(mask)).long()

        return image_t, mask_t, data_id


# ============================================================================
# Model — must exactly match train_segmentation3.py
# ============================================================================

def build_model(num_classes: int) -> torch.nn.Module:
    """U-Net with MobileNetV2 encoder — no pretrained weights needed for inference."""
    return smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,   # we load our own checkpoint weights
        in_channels=3,
        classes=num_classes,
        activation=None,        # raw logits → argmax for predictions
    )


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    pred = pred.view(-1); target = target.view(-1)
    ious = []
    for c in range(num_classes):
        p, t  = pred == c, target == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious.append(float('nan') if union == 0 else (inter / union).item())
    return float(np.nanmean(ious)), ious


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred = pred.view(-1); target = target.view(-1)
    dices = []
    for c in range(num_classes):
        p, t  = pred == c, target == c
        inter = (p & t).sum().float()
        d = (2. * inter + smooth) / (p.sum().float() + t.sum().float() + smooth)
        dices.append(d.item())
    return float(np.mean(dices)), dices


def compute_pixel_accuracy(pred, target):
    return float((pred == target).float().mean().item())


# ============================================================================
# Visualisation
# ============================================================================

def save_comparison(img_t, gt_mask, pred_mask, output_path, data_id):
    mean = np.array(IMG_MEAN); std = np.array(IMG_STD)
    img  = img_t.cpu().numpy()
    img  = np.moveaxis(img, 0, -1) * std + mean
    img  = np.clip(img, 0, 1)

    gt_c   = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_c = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);    axes[0].set_title('Input Image');                    axes[0].axis('off')
    axes[1].imshow(gt_c);   axes[1].set_title('Ground Truth');                   axes[1].axis('off')
    axes[2].imshow(pred_c); axes[2].set_title('U-Net + MobileNetV2 Prediction'); axes[2].axis('off')
    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_report(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, 'test_evaluation_metrics.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('=' * 65 + '\n')
        f.write('  TEST EVALUATION RESULTS — U-Net + MobileNetV2 (Project 3)\n')
        f.write('=' * 65 + '\n\n')
        f.write(f"  Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write(f"  Mean Dice:         {results['mean_dice']:.4f}\n")
        f.write(f"  Pixel Accuracy:    {results['pixel_acc']:.4f}\n")
        f.write(f"  Samples evaluated: {results['n_samples']}\n")
        f.write('\n' + '=' * 65 + '\n\n')
        f.write('Per-Class IoU:\n')
        f.write('-' * 40 + '\n')
        for name, iou in zip(class_names, results['class_iou']):
            s   = f'{iou:.4f}' if not np.isnan(iou) else ' N/A '
            bar = '#' * int(iou * 20) if not np.isnan(iou) else ''
            f.write(f'  {name:>15s}: {s}  {bar}\n')
    print(f'  Saved {filepath}')

    fig, ax = plt.subplots(figsize=(10, 6))
    valid   = [v if not np.isnan(v) else 0 for v in results['class_iou']]
    colors  = [color_palette[i] / 255.0 for i in range(n_classes)]
    ax.bar(range(n_classes), valid, color=colors, edgecolor='black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f"Per-Class IoU — U-Net+MobileNetV2  (Mean: {results['mean_iou']:.4f})")
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_iou'], color='red', linestyle='--', label='Mean IoU')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved per_class_metrics.png')


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description='U-Net + MobileNetV2 inference / evaluation — Project 3'
    )
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(script_dir, 'unet_mobilenetv2_best.pth'))
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(script_dir, '..', '..',
                                             'Offroad_Segmentation_Training_Dataset',
                                             'Offroad_Segmentation_Training_Dataset', 'val'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(script_dir, 'predictions3'))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of side-by-side comparison images to save')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Must match training image size (default: 256)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*65}")
    print(f'  U-Net + MobileNetV2 Inference — Project 3')
    print(f"{'='*65}")
    print(f'  Device:     {device}')
    print(f'  Model:      {args.model_path}')
    print(f'  Data dir:   {args.data_dir}')
    print(f'  Output dir: {args.output_dir}')
    print(f'  Image size: {args.img_size}x{args.img_size}')

    # ── Load model ────────────────────────────────────────────────────────
    print('\nLoading U-Net + MobileNetV2 checkpoint...')
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {args.model_path}\n'
            f'Train the model first with train_segmentation3.py'
        )

    model = build_model(num_classes=n_classes).to(device)

    # weights_only=False required for torch 2.5.x with dict checkpoints
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Restored from epoch {ckpt.get('epoch', '?')}  "
              f"(best Val IoU: {ckpt.get('best_val_iou', float('nan')):.4f})")
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f'  Total params: {sum(p.numel() for p in model.parameters()):,}')
    print('  Model loaded successfully!')

    # ── Dataset ───────────────────────────────────────────────────────────
    img_size = (args.img_size, args.img_size)
    dataset  = OffRoadTestDataset(data_dir=args.data_dir, img_size=img_size)
    loader   = DataLoader(dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=0)
    print(f'\n  Loaded {len(dataset)} test samples')

    # ── Output directories ────────────────────────────────────────────────
    masks_dir       = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    for d in [masks_dir, masks_color_dir, comparisons_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Inference ─────────────────────────────────────────────────────────
    print(f'\nRunning inference on {len(dataset)} images...')
    iou_scores, dice_scores, pixel_accs = [], [], []
    all_class_iou = []
    sample_count  = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc='Inference', unit='batch')
        for pixel_values, gt_masks, data_ids in pbar:
            pixel_values = pixel_values.to(device)
            gt_masks     = gt_masks.to(device)

            logits = model(pixel_values)   # (B, C, H, W) — raw logits
            preds  = logits.argmax(dim=1)  # (B, H, W)

            for i in range(preds.shape[0]):
                pred = preds[i]; gt = gt_masks[i]; did = data_ids[i]
                base = os.path.splitext(did)[0]

                iou,  ci = compute_iou(pred, gt, n_classes)
                dice, _  = compute_dice(pred, gt, n_classes)
                pacc     = compute_pixel_accuracy(pred, gt)

                iou_scores.append(iou)
                dice_scores.append(dice)
                pixel_accs.append(pacc)
                all_class_iou.append(ci)

                # Save raw class-ID mask
                pred_np = pred.cpu().numpy().astype(np.uint8)
                Image.fromarray(pred_np).save(
                    os.path.join(masks_dir, f'{base}_pred.png')
                )

                # Save colourised mask
                pred_rgb = mask_to_color(pred_np)
                cv2.imwrite(
                    os.path.join(masks_color_dir, f'{base}_pred_color.png'),
                    cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
                )

                # Save side-by-side comparison (first N samples only)
                if sample_count < args.num_samples:
                    save_comparison(
                        pixel_values[i], gt, pred,
                        os.path.join(comparisons_dir,
                                     f'sample_{sample_count:04d}_cmp.png'),
                        did
                    )
                sample_count += 1

            pbar.set_postfix(iou=f'{iou:.3f}')

    # ── Aggregate & report ────────────────────────────────────────────────
    mean_iou  = float(np.nanmean(iou_scores))
    mean_dice = float(np.nanmean(dice_scores))
    mean_pacc = float(np.mean(pixel_accs))
    avg_ci    = np.nanmean(all_class_iou, axis=0)

    results = {
        'mean_iou':  mean_iou,
        'mean_dice': mean_dice,
        'pixel_acc': mean_pacc,
        'class_iou': avg_ci,
        'n_samples': len(dataset),
    }

    print('\n' + '=' * 65)
    print('  TEST RESULTS — U-Net + MobileNetV2 (Project 3)')
    print('=' * 65)
    print(f'  Mean IoU:       {mean_iou:.4f}')
    print(f'  Mean Dice:      {mean_dice:.4f}')
    print(f'  Pixel Accuracy: {mean_pacc:.4f}')
    print('\n  Per-Class IoU:')
    for name, iou in zip(class_names, avg_ci):
        s = f'{iou:.4f}' if not np.isnan(iou) else ' N/A '
        print(f'    {name:>15s}: {s}')
    print('=' * 65)

    save_metrics_report(results, args.output_dir)

    print(f"\nDone! {sample_count} images processed. Results saved to '{args.output_dir}/'")
    print(f'  - masks/                     : Raw class-ID prediction masks')
    print(f'  - masks_color/               : Colourised RGB prediction masks')
    print(f'  - comparisons/               : Side-by-side comparisons ({args.num_samples} samples)')
    print(f'  - test_evaluation_metrics.txt')
    print(f'  - per_class_metrics.png')


if __name__ == '__main__':
    main()
