"""
Segmentation Test / Inference Script — Project 4 (ENet)
Evaluates a trained ENet model on test data and saves prediction visualisations.

Usage:
  python test_segmentation4.py
  python test_segmentation4.py --model_path enet_best.pth --data_dir <path> --output_dir predictions

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
    0:     0,   # Background
    100:   1,   # Trees
    200:   2,   # Lush Bushes
    300:   3,   # Dry Grass
    500:   4,   # Dry Bushes
    550:   5,   # Ground Clutter
    700:   6,   # Logs
    800:   7,   # Rocks
    7100:  8,   # Landscape
    10000: 9,   # Sky
}
n_classes = len(value_map)

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

color_palette = np.array([
    [0,   0,   0  ],
    [34,  139, 34 ],
    [0,   255, 0  ],
    [210, 180, 140],
    [139, 90,  43 ],
    [128, 128, 0  ],
    [139, 69,  19 ],
    [128, 128, 128],
    [160, 82,  45 ],
    [135, 206, 235],
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
# ENet (must match train_segmentation4.py exactly)
# ============================================================================

class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels,
                              kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.PReLU(out_channels)

    def forward(self, x):
        return self.act(self.bn(torch.cat([self.conv(x), self.pool(x)], dim=1)))


class RegularBottleneck(nn.Module):
    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=1,
                 dilation=1, asymmetric=False, dropout_prob=0.1):
        super().__init__()
        internal = channels // internal_ratio
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, internal, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal), nn.PReLU(internal),
        )
        if asymmetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(internal, internal, kernel_size=(kernel_size, 1),
                          padding=(padding, 0), bias=False),
                nn.BatchNorm2d(internal), nn.PReLU(internal),
                nn.Conv2d(internal, internal, kernel_size=(1, kernel_size),
                          padding=(0, padding), bias=False),
                nn.BatchNorm2d(internal), nn.PReLU(internal),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(internal, internal, kernel_size=kernel_size,
                          padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(internal), nn.PReLU(internal),
            )
        self.conv3  = nn.Sequential(
            nn.Conv2d(internal, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out_act = nn.PReLU(channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        return self.out_act(self.conv3(self.dropout(self.conv2(self.conv1(x)))) + x)


class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0.1):
        super().__init__()
        internal     = in_channels // internal_ratio
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv1   = nn.Sequential(
            nn.Conv2d(in_channels, internal, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(internal), nn.PReLU(internal),
        )
        self.conv2   = nn.Sequential(
            nn.Conv2d(internal, internal, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(internal), nn.PReLU(internal),
        )
        self.conv3   = nn.Sequential(
            nn.Conv2d(internal, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.expand  = out_channels - in_channels
        self.out_act = nn.PReLU(out_channels)

    def forward(self, x):
        pooled, indices = self.pool(x)
        if self.expand > 0:
            n, c, h, w = pooled.size()
            pad = torch.zeros(n, self.expand, h, w, dtype=pooled.dtype, device=pooled.device)
            pooled = torch.cat([pooled, pad], dim=1)
        out = self.dropout(self.conv3(self.conv2(self.conv1(x))))
        return self.out_act(out + pooled), indices


class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0.1):
        super().__init__()
        internal       = in_channels // internal_ratio
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.unpool    = nn.MaxUnpool2d(kernel_size=2)
        self.conv1     = nn.Sequential(
            nn.Conv2d(in_channels, internal, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal), nn.PReLU(internal),
        )
        self.conv2     = nn.Sequential(
            nn.ConvTranspose2d(internal, internal, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(internal), nn.PReLU(internal),
        )
        self.conv3     = nn.Sequential(
            nn.Conv2d(internal, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout   = nn.Dropout2d(p=dropout_prob)
        self.out_act   = nn.PReLU(out_channels)

    def forward(self, x, indices, output_size):
        main = self.unpool(self.main_conv(x), indices, output_size=output_size)
        out  = self.dropout(self.conv3(self.conv2(self.conv1(x))))
        return self.out_act(out + main)


class ENet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial      = InitialBlock(3, 16)
        self.downsample1_0 = DownsamplingBottleneck(16, 64, dropout_prob=0.01)
        self.regular1     = nn.Sequential(
            RegularBottleneck(64, dropout_prob=0.01),
            RegularBottleneck(64, dropout_prob=0.01),
            RegularBottleneck(64, dropout_prob=0.01),
            RegularBottleneck(64, dropout_prob=0.01),
        )
        self.downsample2_0 = DownsamplingBottleneck(64, 128, dropout_prob=0.1)
        self.stage2 = nn.ModuleList([
            RegularBottleneck(128, dropout_prob=0.1),
            RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1),
            RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1),
            RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1),
            RegularBottleneck(128, dropout_prob=0.1),
            RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1),
            RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1),
            RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1),
        ])
        self.stage3 = nn.ModuleList([
            RegularBottleneck(128, dropout_prob=0.1),
            RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1),
            RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1),
            RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1),
            RegularBottleneck(128, dropout_prob=0.1),
            RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1),
            RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1),
            RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1),
        ])
        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1)
        self.stage4 = nn.Sequential(
            RegularBottleneck(64, dropout_prob=0.1),
            RegularBottleneck(64, dropout_prob=0.1),
        )
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1)
        self.stage5      = nn.Sequential(RegularBottleneck(16, dropout_prob=0.1))
        self.fullconv    = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2, bias=True)

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.initial(x)
        x, idx1 = self.downsample1_0(x);  size1 = x.shape[-2:]
        x = self.regular1(x)
        x, idx2 = self.downsample2_0(x);  size2 = x.shape[-2:]
        for blk in self.stage2: x = blk(x)
        for blk in self.stage3: x = blk(x)
        x = self.upsample4_0(x, idx2, output_size=size1)
        x = self.stage4(x)
        x = self.upsample5_0(x, idx1, output_size=(input_size[0] // 2, input_size[1] // 2))
        x = self.stage5(x)
        x = self.fullconv(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x


# ============================================================================
# Dataset
# ============================================================================

class OffRoadTestDataset(Dataset):
    """
    Loads image/mask pairs from a test directory.
    Returns (image_tensor, gt_mask_tensor, filename).
    """

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
    pred   = pred.view(-1); target = target.view(-1)
    ious   = []
    class_ious = []
    for c in range(num_classes):
        p, t  = pred == c, target == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        if union == 0:
            ious.append(float('nan')); class_ious.append(float('nan'))
        else:
            v = (inter / union).item()
            ious.append(v); class_ious.append(v)
    return float(np.nanmean(ious)), class_ious


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred   = pred.view(-1); target = target.view(-1)
    dices  = []
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
    """Save 3-panel image: input | ground truth | prediction."""
    mean = np.array(IMG_MEAN); std = np.array(IMG_STD)
    img  = img_t.cpu().numpy()
    img  = np.moveaxis(img, 0, -1) * std + mean
    img  = np.clip(img, 0, 1)

    gt_color   = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);        axes[0].set_title('Input Image');     axes[0].axis('off')
    axes[1].imshow(gt_color);   axes[1].set_title('Ground Truth');    axes[1].axis('off')
    axes[2].imshow(pred_color); axes[2].set_title('ENet Prediction'); axes[2].axis('off')
    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_report(results, output_dir):
    """Save text report + per-class IoU bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, 'test_evaluation_metrics.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  TEST EVALUATION RESULTS — ENet (Project 4)\n")
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

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    valid   = [v if not np.isnan(v) else 0 for v in results['class_iou']]
    colors  = [color_palette[i] / 255 for i in range(n_classes)]
    ax.bar(range(n_classes), valid, color=colors, edgecolor='black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f"Per-Class IoU — ENet  (Mean: {results['mean_iou']:.4f})")
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

    parser = argparse.ArgumentParser(description='ENet inference / evaluation script — Project 4')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(script_dir, 'enet_best.pth'),
                        help='Path to trained ENet checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(script_dir, '..', '..',
                                             'Offroad_Segmentation_Training_Dataset',
                                             'Offroad_Segmentation_Training_Dataset', 'val'),
                        help='Path to dataset directory (must contain Color_Images/ and Segmentation/)')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(script_dir, 'predictions4'),
                        help='Directory to save predictions')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of side-by-side comparison images to save')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Input image size (square). Must match training.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  ENet Inference — Project 4")
    print(f"{'='*60}")
    print(f"  Device:     {device}")
    print(f"  Model:      {args.model_path}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Image size: {args.img_size}×{args.img_size}")

    # ── Load model ────────────────────────────────────────────────────────
    print("\nLoading ENet checkpoint…")
    model = ENet(num_classes=n_classes).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.model_path}\n"
            f"Train the model first with train_segmentation4.py"
        )

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Restored from training epoch {ckpt.get('epoch', '?')}")
        print(f"  Reported best Val IoU: {ckpt.get('best_val_iou', 'N/A'):.4f}")
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print("  ENet loaded successfully!")

    # ── Dataset ───────────────────────────────────────────────────────────
    img_size = (args.img_size, args.img_size)
    dataset  = OffRoadTestDataset(data_dir=args.data_dir, img_size=img_size)
    loader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"\n  Loaded {len(dataset)} test samples")

    # ── Output directories ────────────────────────────────────────────────
    masks_dir      = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    for d in [masks_dir, masks_color_dir, comparisons_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Inference ─────────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(dataset)} images…")
    iou_scores, dice_scores, pixel_accs = [], [], []
    all_class_iou, all_class_dice       = [], []
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Inference", unit="batch")
        for pixel_values, gt_masks, data_ids in pbar:
            pixel_values = pixel_values.to(device)
            gt_masks     = gt_masks.to(device)

            logits = model(pixel_values)
            preds  = logits.argmax(dim=1)          # (B, H, W)

            for i in range(preds.shape[0]):
                pred   = preds[i]
                gt     = gt_masks[i]
                did    = data_ids[i]
                base   = os.path.splitext(did)[0]

                iou,  ci  = compute_iou(pred, gt, n_classes)
                dice, cd  = compute_dice(pred, gt, n_classes)
                pacc      = compute_pixel_accuracy(pred, gt)

                iou_scores.append(iou)
                dice_scores.append(dice)
                pixel_accs.append(pacc)
                all_class_iou.append(ci)
                all_class_dice.append(cd)

                # Save raw class-ID mask
                pred_np = pred.cpu().numpy().astype(np.uint8)
                Image.fromarray(pred_np).save(os.path.join(masks_dir, f'{base}_pred.png'))

                # Save colour mask
                pred_rgb = mask_to_color(pred_np)
                cv2.imwrite(os.path.join(masks_color_dir, f'{base}_pred_color.png'),
                            cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

                # Save comparison for first N samples
                if sample_count < args.num_samples:
                    save_comparison(
                        pixel_values[i], gt, pred,
                        os.path.join(comparisons_dir, f'sample_{sample_count:04d}_cmp.png'),
                        did
                    )
                sample_count += 1

            pbar.set_postfix(iou=f"{iou:.3f}")

    # ── Aggregate & report ────────────────────────────────────────────────
    mean_iou   = float(np.nanmean(iou_scores))
    mean_dice  = float(np.nanmean(dice_scores))
    mean_pacc  = float(np.mean(pixel_accs))
    avg_ci     = np.nanmean(all_class_iou, axis=0)

    results = {
        'mean_iou':  mean_iou,
        'mean_dice': mean_dice,
        'pixel_acc': mean_pacc,
        'class_iou': avg_ci,
        'n_samples': len(dataset),
    }

    print("\n" + "=" * 60)
    print("  TEST RESULTS — ENet (Project 4)")
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
