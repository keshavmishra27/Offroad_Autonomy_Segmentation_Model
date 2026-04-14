"""
Segmentation Test / Inference Script — Project 5 (Custom CNN)
Evaluates a trained Custom CNN model on test data and saves prediction visualisations.

Usage:
  python test_segmentation5.py
  python test_segmentation5.py --model_path custom_cnn_best.pth --data_dir <path>

No extra libraries required — uses PyTorch + PIL + OpenCV only.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

plt.switch_backend('Agg')


# ============================================================================
# Shared definitions (mask mapping, class names, colours)
# ============================================================================

value_map = {
    0:     0,   100:   1,   200:   2,   300:   3,   500:   4,
    550:   5,   700:   6,   800:   7,   7100:  8,   10000: 9,
}
n_classes = len(value_map)

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

color_palette = np.array([
    [0,   0,   0  ],  [34,  139, 34 ],  [0,   255, 0  ],  [210, 180, 140],
    [139, 90,  43 ],  [128, 128, 0  ],  [139, 69,  19 ],  [128, 128, 128],
    [160, 82,  45 ],  [135, 206, 235],
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
# Custom CNN Model (must match train_segmentation5.py exactly)
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout=dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderStage(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes, base_ch=32):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16

        self.enc1 = EncoderStage(3,  c1, dropout=0.0)
        self.enc2 = EncoderStage(c1, c2, dropout=0.0)
        self.enc3 = EncoderStage(c2, c3, dropout=0.1)
        self.enc4 = EncoderStage(c3, c4, dropout=0.1)

        self.bottleneck = ConvBlock(c4, c5, dropout=0.2)

        self.dec4 = DecoderStage(c5, c4, c4, dropout=0.1)
        self.dec3 = DecoderStage(c4, c3, c3, dropout=0.1)
        self.dec2 = DecoderStage(c3, c2, c2, dropout=0.0)
        self.dec1 = DecoderStage(c2, c1, c1, dropout=0.0)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.head(x)


# ============================================================================
# Dataset
# ============================================================================

class OffRoadTestDataset(Dataset):
    def __init__(self, data_dir, img_size):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.img_size  = img_size
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.mean      = torch.tensor(IMG_MEAN).view(3, 1, 1)
        self.std       = torch.tensor(IMG_STD ).view(3, 1, 1)

    def __len__(self): return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        h, w    = self.img_size

        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        image = image.resize((w, h), Image.BILINEAR)

        mask  = Image.open(os.path.join(self.masks_dir, data_id))
        mask  = convert_mask(mask)
        mask  = mask.resize((w, h), Image.NEAREST)

        image_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_t = (image_t - self.mean) / self.std
        mask_t  = torch.from_numpy(np.array(mask)).long()

        return image_t, mask_t, data_id


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    pred = pred.view(-1); target = target.view(-1)
    ious, class_ious = [], []
    for c in range(num_classes):
        p, t  = pred == c, target == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        v     = float('nan') if union == 0 else (inter / union).item()
        ious.append(v); class_ious.append(v)
    return float(np.nanmean(ious)), class_ious


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred = pred.view(-1); target = target.view(-1)
    dices = []
    for c in range(num_classes):
        p, t  = pred == c, target == c
        inter = (p & t).sum().float()
        d     = (2. * inter + smooth) / (p.sum().float() + t.sum().float() + smooth)
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
    axes[0].imshow(img);    axes[0].set_title('Input Image');        axes[0].axis('off')
    axes[1].imshow(gt_c);   axes[1].set_title('Ground Truth');       axes[1].axis('off')
    axes[2].imshow(pred_c); axes[2].set_title('Custom CNN Prediction'); axes[2].axis('off')
    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_report(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, 'test_evaluation_metrics.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  TEST EVALUATION RESULTS — Custom CNN (Project 5)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write(f"  Mean Dice:         {results['mean_dice']:.4f}\n")
        f.write(f"  Pixel Accuracy:    {results['pixel_acc']:.4f}\n")
        f.write(f"  Samples evaluated: {results['n_samples']}\n")
        f.write("\n" + "=" * 60 + "\n\n")
        f.write("Per-Class IoU:\n")
        f.write("-" * 40 + "\n")
        for name, iou in zip(class_names, results['class_iou']):
            s = f"{iou:.4f}" if not np.isnan(iou) else " N/A "
            bar = "#" * int(iou * 20) if not np.isnan(iou) else ""
            f.write(f"  {name:>15s}: {s}  {bar}\n")

    print(f"  Saved {filepath}")

    fig, ax = plt.subplots(figsize=(10, 6))
    valid   = [v if not np.isnan(v) else 0 for v in results['class_iou']]
    colors  = [color_palette[i] / 255 for i in range(n_classes)]
    ax.bar(range(n_classes), valid, color=colors, edgecolor='black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f"Per-Class IoU — Custom CNN  (Mean: {results['mean_iou']:.4f})")
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_iou'], color='red', linestyle='--', label='Mean IoU')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved per_class_metrics.png")


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Custom CNN inference / evaluation — Project 5')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(script_dir, 'custom_cnn_best.pth'))
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(script_dir, '..', '..',
                                             'Offroad_Segmentation_Training_Dataset',
                                             'Offroad_Segmentation_Training_Dataset', 'val'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(script_dir, 'predictions5'))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  Custom CNN Inference — Project 5")
    print(f"{'='*60}")
    print(f"  Device:     {device}")
    print(f"  Model:      {args.model_path}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Image size: {args.img_size}×{args.img_size}")

    # ── Load model ────────────────────────────────────────────────────────
    print("\nLoading Custom CNN checkpoint…")
    model = CustomCNN(num_classes=n_classes, base_ch=32).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.model_path}\n"
            f"Train the model first with train_segmentation5.py"
        )

    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Restored from epoch {ckpt.get('epoch', '?')}")
        best_iou = ckpt.get('best_val_iou', None)
        if best_iou is not None:
            print(f"  Reported best Val IoU: {best_iou:.4f}")
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print("  Custom CNN loaded successfully!")

    # ── Dataset ───────────────────────────────────────────────────────────
    img_size = (args.img_size, args.img_size)
    dataset  = OffRoadTestDataset(data_dir=args.data_dir, img_size=img_size)
    loader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"\n  Loaded {len(dataset)} test samples")

    # ── Output directories ────────────────────────────────────────────────
    masks_dir       = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    for d in [masks_dir, masks_color_dir, comparisons_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Inference ─────────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(dataset)} images…")
    iou_scores, dice_scores, pixel_accs = [], [], []
    all_class_iou = []
    sample_count  = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Inference", unit="batch")
        for pixel_values, gt_masks, data_ids in pbar:
            pixel_values = pixel_values.to(device)
            gt_masks     = gt_masks.to(device)

            logits = model(pixel_values)
            preds  = logits.argmax(dim=1)

            for i in range(preds.shape[0]):
                pred = preds[i]; gt = gt_masks[i]; did = data_ids[i]
                base = os.path.splitext(did)[0]

                iou, ci   = compute_iou(pred, gt, n_classes)
                dice, _   = compute_dice(pred, gt, n_classes)
                pacc      = compute_pixel_accuracy(pred, gt)

                iou_scores.append(iou)
                dice_scores.append(dice)
                pixel_accs.append(pacc)
                all_class_iou.append(ci)

                # Save raw mask
                pred_np = pred.cpu().numpy().astype(np.uint8)
                Image.fromarray(pred_np).save(os.path.join(masks_dir, f'{base}_pred.png'))

                # Save colour mask
                pred_rgb = mask_to_color(pred_np)
                cv2.imwrite(os.path.join(masks_color_dir, f'{base}_pred_color.png'),
                            cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

                # Save comparison
                if sample_count < args.num_samples:
                    save_comparison(
                        pixel_values[i], gt, pred,
                        os.path.join(comparisons_dir, f'sample_{sample_count:04d}_cmp.png'),
                        did
                    )
                sample_count += 1

            pbar.set_postfix(iou=f"{iou:.3f}")

    # ── Results ───────────────────────────────────────────────────────────
    mean_iou  = float(np.nanmean(iou_scores))
    mean_dice = float(np.nanmean(dice_scores))
    mean_pacc = float(np.mean(pixel_accs))
    avg_ci    = np.nanmean(all_class_iou, axis=0)

    results = {
        'mean_iou': mean_iou, 'mean_dice': mean_dice,
        'pixel_acc': mean_pacc, 'class_iou': avg_ci,
        'n_samples': len(dataset),
    }

    print("\n" + "=" * 60)
    print("  TEST RESULTS — Custom CNN (Project 5)")
    print("=" * 60)
    print(f"  Mean IoU:       {mean_iou:.4f}")
    print(f"  Mean Dice:      {mean_dice:.4f}")
    print(f"  Pixel Accuracy: {mean_pacc:.4f}")
    print("\n  Per-Class IoU:")
    for name, iou in zip(class_names, avg_ci):
        s = f"{iou:.4f}" if not np.isnan(iou) else " N/A "
        print(f"    {name:>15s}: {s}")
    print("=" * 60)

    save_metrics_report(results, args.output_dir)

    print(f"\nDone! Results saved to '{args.output_dir}/'")
    print(f"  - masks/           : Raw class-ID prediction masks")
    print(f"  - masks_color/     : Coloured RGB prediction masks")
    print(f"  - comparisons/     : Side-by-side comparisons ({args.num_samples} samples)")
    print(f"  - test_evaluation_metrics.txt")
    print(f"  - per_class_metrics.png")


if __name__ == "__main__":
    main()
