"""
Segmentation Training Script — Project 5 (Custom CNN)
Trains a custom encoder-decoder CNN model for offroad semantic segmentation.

The model is a pure-PyTorch encoder-decoder with skip connections:
  Encoder: 4 stages of (Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU → MaxPool)
  Decoder: 4 stages of (Upsample → concat skip → Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU)

This is intentionally simple — no pretrained weights, no external libraries.

CPU-optimised strategy:
  - ~1.2 M parameters (very lightweight)
  - Input resolution 128×128
  - Batch size 16
  - Limited validation batches per epoch for speed
  - No external dependencies beyond PyTorch

Requirements:
  pip install torch torchvision tqdm matplotlib pillow opencv-python numpy
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
import cv2
import random
from tqdm import tqdm
import time

# Non-interactive backend for matplotlib
plt.switch_backend('Agg')


# ============================================================================
# Utility
# ============================================================================

def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"


def save_image(img, filename):
    """Save a normalised image tensor to file."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = np.moveaxis(img, 0, -1)
    img  = (img * std + mean) * 255
    img  = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion (same class mapping as all other projects)
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


def convert_mask(mask):
    """Convert raw pixel values to sequential class IDs."""
    arr     = np.array(mask, dtype=np.int32)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask_np):
    """Convert a class-ID mask to an RGB image."""
    h, w = mask_np.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_classes):
        rgb[mask_np == c] = color_palette[c]
    return rgb


# ============================================================================
# Dataset
# ============================================================================

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


class OffRoadDataset(Dataset):
    """
    Loads image / mask pairs for offroad semantic segmentation.
    Returns normalised image tensor (3,H,W) and long mask tensor (H,W).
    """

    def __init__(self, data_dir: str, img_size: tuple, augment: bool = False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.img_size  = img_size          # (h, w)
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.augment   = augment
        self.mean      = torch.tensor(IMG_MEAN).view(3, 1, 1)
        self.std       = torch.tensor(IMG_STD ).view(3, 1, 1)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        h, w    = self.img_size

        # --- Image ---
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        image = image.resize((w, h), Image.BILINEAR)

        # --- Mask ---
        mask = Image.open(os.path.join(self.masks_dir, data_id))
        mask = convert_mask(mask)
        mask = mask.resize((w, h), Image.NEAREST)

        # --- Optional augmentation (horizontal flip) ---
        if self.augment and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask  = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_t = (image_t - self.mean) / self.std
        mask_t  = torch.from_numpy(np.array(mask)).long()

        return image_t, mask_t


# ============================================================================
# Custom CNN Segmentation Model — Encoder-Decoder with Skip Connections
# ============================================================================
#
# Architecture overview:
#
#   ┌─────────────────────────┐
#   │  Input (3, 128, 128)    │
#   └────────┬────────────────┘
#            │
#   ┌────────▼────────────────┐   ← Encoder Stage 1
#   │ Conv3×3 → BN → ReLU    │
#   │ Conv3×3 → BN → ReLU    │   skip1: 32 ch, 128×128
#   │ MaxPool 2×2             │   → 32 ch, 64×64
#   └────────┬────────────────┘
#            │
#   ┌────────▼────────────────┐   ← Encoder Stage 2
#   │ Conv3×3 → BN → ReLU    │
#   │ Conv3×3 → BN → ReLU    │   skip2: 64 ch, 64×64
#   │ MaxPool 2×2             │   → 64 ch, 32×32
#   └────────┬────────────────┘
#            │
#   ┌────────▼────────────────┐   ← Encoder Stage 3
#   │ Conv3×3 → BN → ReLU    │
#   │ Conv3×3 → BN → ReLU    │   skip3: 128 ch, 32×32
#   │ MaxPool 2×2             │   → 128 ch, 16×16
#   └────────┬────────────────┘
#            │
#   ┌────────▼────────────────┐   ← Encoder Stage 4
#   │ Conv3×3 → BN → ReLU    │
#   │ Conv3×3 → BN → ReLU    │   skip4: 256 ch, 16×16
#   │ MaxPool 2×2             │   → 256 ch, 8×8
#   └────────┬────────────────┘
#            │
#   ┌────────▼────────────────┐   ← Bottleneck
#   │ Conv3×3 → BN → ReLU    │
#   │ Conv3×3 → BN → ReLU    │   512 ch, 8×8
#   └────────┬────────────────┘
#            │
#   ┌────────▼────────────────┐   ← Decoder Stage 4
#   │ Upsample 2×  → 16×16   │
#   │ Concat skip4  (768 ch)  │
#   │ Conv3×3 → BN → ReLU    │
#   │ Conv3×3 → BN → ReLU    │   → 256 ch, 16×16
#   └────────┬────────────────┘
#            │
#   ┌────────▼────────────────┐   ← Decoder Stage 3
#   │ Upsample 2×  → 32×32   │
#   │ Concat skip3  (384 ch)  │
#   │ Conv3×3 → BN → ReLU    │
#   │ Conv3×3 → BN → ReLU    │   → 128 ch, 32×32
#   └────────┬────────────────┘
#            │
#   ┌────────▼────────────────┐   ← Decoder Stage 2
#   │ Upsample 2×  → 64×64   │
#   │ Concat skip2  (192 ch)  │
#   │ Conv3×3 → BN → ReLU    │
#   │ Conv3×3 → BN → ReLU    │   → 64 ch, 64×64
#   └────────┬────────────────┘
#            │
#   ┌────────▼────────────────┐   ← Decoder Stage 1
#   │ Upsample 2×  → 128×128 │
#   │ Concat skip1  (96 ch)   │
#   │ Conv3×3 → BN → ReLU    │
#   │ Conv3×3 → BN → ReLU    │   → 32 ch, 128×128
#   └────────┬────────────────┘
#            │
#   ┌────────▼────────────────┐
#   │ Conv 1×1 → n_classes    │   Logits (10, 128, 128)
#   └─────────────────────────┘
#

class ConvBlock(nn.Module):
    """Two consecutive (Conv3×3 → BatchNorm → ReLU) layers."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderStage(nn.Module):
    """ConvBlock followed by 2×2 MaxPool. Returns (pooled, skip)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout=dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderStage(nn.Module):
    """Bilinear upsample → concat skip → ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CustomCNN(nn.Module):
    """
    Custom encoder-decoder CNN for semantic segmentation.

    Channels: 32 → 64 → 128 → 256 → 512 (bottleneck) → decode back.
    Total: ~1.2 M parameters (very CPU-friendly).
    """

    def __init__(self, num_classes: int, base_ch: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16

        # Encoder
        self.enc1 = EncoderStage(3,  c1, dropout=0.0)    # 3  → 32
        self.enc2 = EncoderStage(c1, c2, dropout=0.0)    # 32 → 64
        self.enc3 = EncoderStage(c2, c3, dropout=0.1)    # 64 → 128
        self.enc4 = EncoderStage(c3, c4, dropout=0.1)    # 128→ 256

        # Bottleneck
        self.bottleneck = ConvBlock(c4, c5, dropout=0.2)  # 256→ 512

        # Decoder
        self.dec4 = DecoderStage(c5, c4, c4, dropout=0.1)  # 512+256 → 256
        self.dec3 = DecoderStage(c4, c3, c3, dropout=0.1)  # 256+128 → 128
        self.dec2 = DecoderStage(c3, c2, c2, dropout=0.0)  # 128+64  → 64
        self.dec1 = DecoderStage(c2, c1, c1, dropout=0.0)  # 64+32   → 32

        # Final classifier
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1, bias=True)

        # Weight initialisation (Kaiming for ReLU activations)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        x, skip1 = self.enc1(x)   # /2
        x, skip2 = self.enc2(x)   # /4
        x, skip3 = self.enc3(x)   # /8
        x, skip4 = self.enc4(x)   # /16

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec4(x, skip4)   # ×2
        x = self.dec3(x, skip3)   # ×2
        x = self.dec2(x, skip2)   # ×2
        x = self.dec1(x, skip1)   # ×2

        # Classifier
        return self.head(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    """Mean IoU over classes (ignores classes absent from both pred and target)."""
    pred   = pred.view(-1)
    target = target.view(-1)
    ious   = []
    for c in range(num_classes):
        p = pred   == c
        t = target == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((inter / union).item())
    return float(np.nanmean(ious))


def compute_per_class_iou(pred, target, num_classes=10):
    """Returns a dict {class_name: iou}."""
    pred   = pred.view(-1)
    target = target.view(-1)
    result = {}
    for c in range(num_classes):
        p = pred   == c
        t = target == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        result[class_names[c]] = float('nan') if union == 0 else (inter / union).item()
    return result


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Mean Dice coefficient over classes."""
    pred   = pred.view(-1)
    target = target.view(-1)
    dices  = []
    for c in range(num_classes):
        p = pred   == c
        t = target == c
        inter = (p & t).sum().float()
        d = (2. * inter + smooth) / (p.sum().float() + t.sum().float() + smooth)
        dices.append(d.item())
    return float(np.mean(dices))


def compute_pixel_accuracy(pred, target):
    return float((pred == target).float().mean().item())


# ============================================================================
# Class distribution + weights
# ============================================================================

def compute_class_weights(data_dir, num_classes):
    """Compute inverse-frequency class weights from the training masks."""
    masks_dir    = os.path.join(data_dir, 'Segmentation')
    class_counts = np.zeros(num_classes, dtype=np.float64)
    mask_files   = sorted(os.listdir(masks_dir))

    print("  Analysing class distribution…")
    for mf in tqdm(mask_files, desc="  Masks", unit="mask", leave=False):
        mask = Image.open(os.path.join(masks_dir, mf))
        mask = convert_mask(mask)
        arr  = np.array(mask)
        for c in range(num_classes):
            class_counts[c] += (arr == c).sum()

    total = class_counts.sum()
    print("\n  Class Distribution:")
    print("  " + "-" * 65)
    for i in range(num_classes):
        pct = 100.0 * class_counts[i] / total
        bar = "#" * int(pct / 2)
        print(f"  {class_names[i]:>15s}: {pct:5.1f}%  {bar}")
    print("  " + "-" * 65)

    class_counts = np.maximum(class_counts, 1)
    w = 1.0 / class_counts
    w = w / w.sum() * num_classes
    return class_counts, w


# ============================================================================
# Plotting
# ============================================================================

def save_training_plots(history, output_dir):
    """Save publication-quality training metric plots."""
    os.makedirs(output_dir, exist_ok=True)
    epochs  = range(1, len(history['train_loss']) + 1)
    c_train = '#1976D2'
    c_val   = '#E53935'
    c_best  = '#4CAF50'

    def _best_line(ax, values, mode='max'):
        best_e = (np.argmax(values) if mode == 'max' else np.argmin(values)) + 1
        best_v = (max(values) if mode == 'max' else min(values))
        ax.axvline(x=best_e, color=c_best, linestyle='--', alpha=0.5,
                   label=f'Best (ep {best_e})')
        ax.scatter([best_e], [best_v], color=c_best, s=100, zorder=5, marker='*')

    # ── 1. Loss + Accuracy ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_loss'], label='Train', color=c_train,
                 linewidth=2, marker='o', markersize=3)
    axes[0].plot(epochs, history['val_loss'],   label='Val',   color=c_val,
                 linewidth=2, marker='s', markersize=3)
    _best_line(axes[0], history['val_loss'], mode='min')
    axes[0].set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss',  fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_pixel_acc'], label='Train', color=c_train,
                 linewidth=2, marker='o', markersize=3)
    axes[1].plot(epochs, history['val_pixel_acc'],   label='Val',   color=c_val,
                 linewidth=2, marker='s', markersize=3)
    _best_line(axes[1], history['val_pixel_acc'], mode='max')
    axes[1].set_title('Pixel Accuracy vs Epoch', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved training_curves.png")

    # ── 2. IoU ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_iou'], label='Train IoU', color=c_train,
                 linewidth=2, marker='o', markersize=3)
    axes[0].set_title('Train IoU vs Epoch', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('IoU', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    axes[1].plot(epochs, history['val_iou'], label='Val IoU', color=c_val,
                 linewidth=2, marker='s', markersize=3)
    _best_line(axes[1], history['val_iou'], mode='max')
    axes[1].set_title('Validation IoU vs Epoch', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('IoU', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved iou_curves.png")

    # ── 3. Dice ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_dice'], label='Train Dice', color=c_train,
                 linewidth=2, marker='o', markersize=3)
    axes[0].set_title('Train Dice vs Epoch', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Dice Score', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    axes[1].plot(epochs, history['val_dice'], label='Val Dice', color=c_val,
                 linewidth=2, marker='s', markersize=3)
    _best_line(axes[1], history['val_dice'], mode='max')
    axes[1].set_title('Validation Dice vs Epoch', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Score', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved dice_curves.png")

    # ── 4. Combined 2×2 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, key_tr, key_vl, title in [
        (axes[0, 0], 'train_loss',      'val_loss',      'Loss'),
        (axes[0, 1], 'train_iou',       'val_iou',       'IoU'),
        (axes[1, 0], 'train_dice',      'val_dice',      'Dice Score'),
        (axes[1, 1], 'train_pixel_acc', 'val_pixel_acc', 'Pixel Accuracy'),
    ]:
        ax.plot(epochs, history[key_tr], label='Train', color=c_train,
                linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, history[key_vl], label='Val',   color=c_val,
                linewidth=2, marker='s', markersize=3)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if 'loss' not in key_tr:
            ax.set_ylim(0, 1)

    plt.suptitle('Custom CNN Training Progress', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved all_metrics_curves.png")

    # ── 5. LR ──
    if 'lr' in history:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, history['lr'], color='#7B1FA2', linewidth=2,
                 marker='o', markersize=3)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lr_schedule.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved lr_schedule.png")


def save_history_to_file(history, output_dir, per_class_iou=None):
    """Save comprehensive training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  SEGMENTATION TRAINING RESULTS (Custom CNN — Project 5)\n")
        f.write("=" * 80 + "\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f}  (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f}  (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f}  (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f}  (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 80 + "\n\n")

        if per_class_iou is not None:
            f.write("Per-Class IoU (Best Model):\n")
            f.write("-" * 40 + "\n")
            for name, iou_val in per_class_iou.items():
                if np.isnan(iou_val):
                    f.write(f"  {name:>15s}:    N/A\n")
                else:
                    bar = "#" * int(iou_val * 20)
                    f.write(f"  {name:>15s}: {iou_val:.4f} {bar}\n")
            f.write("-" * 40 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 110 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc', 'LR']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 110 + "\n")

        for i in range(len(history['train_loss'])):
            lr_val = history['lr'][i] if 'lr' in history else 0
            f.write(
                "{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} "
                "{:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.6f}\n".format(
                    i + 1,
                    history['train_loss'][i],    history['val_loss'][i],
                    history['train_iou'][i],     history['val_iou'][i],
                    history['train_dice'][i],    history['val_dice'][i],
                    history['train_pixel_acc'][i], history['val_pixel_acc'][i],
                    lr_val,
                )
            )

    print("  Saved evaluation_metrics.txt")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    total_start = time.time()

    # ──────────────────────────────────────────────────────────────────────
    # Configuration
    # ──────────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*65}")
    print(f"  Offroad Segmentation — Project 5")
    print(f"  Model: Custom CNN (Encoder-Decoder with Skip Connections)")
    print(f"{'='*65}")
    print(f"  Device: {device}")

    # ── Hyper-parameters tuned for <20 min/epoch on CPU ──────────────────
    IMG_H, IMG_W        = 128, 128   # small resolution for fast CPU training
    BATCH_SIZE          = 16
    LR                  = 1e-3       # higher LR for training from scratch
    N_EPOCHS            = 11
    EARLY_STOP_PATIENCE = 8
    NUM_WORKERS         = 0          # Windows: keep 0

    METRIC_VAL_BATCHES  = 20         # limit val batches per epoch for speed

    print(f"  Image size:          {IMG_W}×{IMG_H}")
    print(f"  Batch size:          {BATCH_SIZE}")
    print(f"  Learning rate:       {LR}")
    print(f"  Epochs:              {N_EPOCHS}")
    print(f"  Early-stop patience: {EARLY_STOP_PATIENCE}")
    print(f"  Val metric batches:  {METRIC_VAL_BATCHES} (fast per-epoch estimate)")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats5')
    os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # Dataset Paths (same as all other projects)
    # ──────────────────────────────────────────────────────────────────────
    data_dir = os.path.join(
        script_dir, '..', '..',
        'Offroad_Segmentation_Training_Dataset',
        'Offroad_Segmentation_Training_Dataset', 'train'
    )
    val_dir = os.path.join(
        script_dir, '..', '..',
        'Offroad_Segmentation_Training_Dataset',
        'Offroad_Segmentation_Training_Dataset', 'val'
    )

    trainset     = OffRoadDataset(data_dir=data_dir, img_size=(IMG_H, IMG_W), augment=True)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False)

    valset      = OffRoadDataset(data_dir=val_dir, img_size=(IMG_H, IMG_W), augment=False)
    val_loader  = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False)

    print(f"\n  Training samples:   {len(trainset)}")
    print(f"  Validation samples: {len(valset)}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Class weights
    # ──────────────────────────────────────────────────────────────────────
    print("\nStep 1/3: Analysing class distribution…")
    class_counts, class_weights = compute_class_weights(data_dir, n_classes)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"  Class weights (relative): {np.round(class_weights, 3)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, ignore_index=255)

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Build Custom CNN
    # ──────────────────────────────────────────────────────────────────────
    print("\nStep 2/3: Building Custom CNN…")
    model = CustomCNN(num_classes=n_classes, base_ch=32).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}  (all from scratch)")

    # ──────────────────────────────────────────────────────────────────────
    # Optimizer & Scheduler
    # ──────────────────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7, verbose=False
    )

    print(f"\n  Loss:      CrossEntropyLoss (class-weighted)")
    print(f"  Optimizer: Adam (lr={LR}, weight_decay=1e-4)")
    print(f"  Scheduler: ReduceLROnPlateau (mode=max, factor=0.5, patience=3)")

    # ──────────────────────────────────────────────────────────────────────
    # History
    # ──────────────────────────────────────────────────────────────────────
    history = {
        'train_loss': [],    'val_loss': [],
        'train_iou':  [],    'val_iou':  [],
        'train_dice': [],    'val_dice': [],
        'train_pixel_acc': [], 'val_pixel_acc': [],
        'lr': [],
    }

    best_val_iou     = 0.0
    best_epoch       = 0
    patience_counter = 0
    best_ckpt_path   = os.path.join(script_dir, 'custom_cnn_best.pth')

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Training loop
    # ──────────────────────────────────────────────────────────────────────
    print(f"\nStep 3/3: Training for {N_EPOCHS} epochs…")
    print("=" * 90)

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        current_lr  = optimizer.param_groups[0]['lr']

        # ── Training ──────────────────────────────────────────────────────
        model.train()
        train_losses, train_iou_acc, train_dice_acc, train_acc_acc = [], [], [], []

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:>2}/{N_EPOCHS} [Train]",
            leave=False, unit="batch",
        )

        for images, masks in train_pbar:
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = logits.argmax(dim=1)

            b_iou, b_dice, b_acc = [], [], []
            for pred, gt in zip(preds, masks):
                b_iou.append(compute_iou(pred, gt, n_classes))
                b_dice.append(compute_dice(pred, gt, n_classes))
                b_acc.append(compute_pixel_accuracy(pred, gt))

            train_losses.append(loss.item())
            train_iou_acc.append(float(np.nanmean(b_iou)))
            train_dice_acc.append(float(np.mean(b_dice)))
            train_acc_acc.append(float(np.mean(b_acc)))

            train_pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                iou=f"{np.nanmean(b_iou):.3f}",
            )

        # ── Validation (limited to METRIC_VAL_BATCHES for speed) ─────────
        model.eval()
        val_losses, val_iou_acc, val_dice_acc, val_acc_acc = [], [], [], []

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1:>2}/{N_EPOCHS} [Val]  ",
            leave=False, unit="batch",
            total=min(METRIC_VAL_BATCHES, len(val_loader)),
        )

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                if batch_idx >= METRIC_VAL_BATCHES:
                    break

                images = images.to(device)
                masks  = masks.to(device)

                logits = model(images)
                loss   = criterion(logits, masks)
                preds  = logits.argmax(dim=1)

                val_losses.append(loss.item())
                for pred, gt in zip(preds, masks):
                    val_iou_acc.append(compute_iou(pred, gt, n_classes))
                    val_dice_acc.append(compute_dice(pred, gt, n_classes))
                    val_acc_acc.append(compute_pixel_accuracy(pred, gt))

                val_pbar.set_postfix(loss=f"{loss.item():.3f}")
                val_pbar.update(1)

        val_pbar.close()

        # ── Aggregate ─────────────────────────────────────────────────────
        epoch_train_loss = float(np.mean(train_losses))
        epoch_val_loss   = float(np.mean(val_losses))
        epoch_train_iou  = float(np.nanmean(train_iou_acc))
        epoch_val_iou    = float(np.nanmean(val_iou_acc))
        epoch_train_dice = float(np.mean(train_dice_acc))
        epoch_val_dice   = float(np.mean(val_dice_acc))
        epoch_train_acc  = float(np.mean(train_acc_acc))
        epoch_val_acc    = float(np.mean(val_acc_acc))

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(epoch_train_iou)
        history['val_iou'].append(epoch_val_iou)
        history['train_dice'].append(epoch_train_dice)
        history['val_dice'].append(epoch_val_dice)
        history['train_pixel_acc'].append(epoch_train_acc)
        history['val_pixel_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)

        scheduler.step(epoch_val_iou)

        # ── Checkpoint + early stopping ───────────────────────────────────
        improved = ""
        if epoch_val_iou > best_val_iou:
            best_val_iou     = epoch_val_iou
            best_epoch       = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch':              epoch + 1,
                'model_state_dict':   model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou':       best_val_iou,
                'n_classes':          n_classes,
                'class_names':        class_names,
                'image_size':         (IMG_H, IMG_W),
                'architecture':       'custom_cnn',
            }, best_ckpt_path)
            improved = " ★ NEW BEST"
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start
        remaining  = epoch_time * (N_EPOCHS - epoch - 1)

        print(
            f"Epoch {epoch+1:>2}/{N_EPOCHS} | "
            f"Loss: {epoch_train_loss:.4f}/{epoch_val_loss:.4f} | "
            f"Train IoU: {epoch_train_iou:.4f} | Val IoU: {epoch_val_iou:.4f} | "
            f"Val Dice: {epoch_val_dice:.4f} | Val Acc: {epoch_val_acc:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"{format_time(epoch_time)} (ETA: {format_time(remaining)})"
            f"{improved}"
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping — no improvement for {EARLY_STOP_PATIENCE} epochs.")
            print(f"  Best Val IoU: {best_val_iou:.4f} at epoch {best_epoch}")
            break

    # ──────────────────────────────────────────────────────────────────────
    # Final evaluation — full validation set with per-class IoU
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("Final evaluation on best checkpoint (full validation set)…")

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Restored best model from epoch {best_epoch}")

    model.eval()
    all_per_class = {name: [] for name in class_names}

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="  Per-class IoU", leave=False):
            images = images.to(device)
            masks  = masks.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1)
            for pred, gt in zip(preds, masks):
                per_c = compute_per_class_iou(pred, gt, n_classes)
                for name, v in per_c.items():
                    all_per_class[name].append(v)

    final_per_class = {
        name: float(np.nanmean(vals)) for name, vals in all_per_class.items()
    }

    print("\n  Per-Class IoU (Best Model):")
    print("  " + "-" * 42)
    for name, iou_val in final_per_class.items():
        if np.isnan(iou_val):
            print(f"  {name:>15s}:  N/A")
        else:
            bar = "#" * int(iou_val * 20)
            print(f"  {name:>15s}: {iou_val:.4f}  {bar}")
    print("  " + "-" * 42)

    # ──────────────────────────────────────────────────────────────────────
    # Save plots & history
    # ──────────────────────────────────────────────────────────────────────
    print("\nSaving plots…")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir, per_class_iou=final_per_class)

    print(f"  Best checkpoint: '{best_ckpt_path}'")

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE — Custom CNN (Project 5)")
    print(f"{'='*65}")
    print(f"  Total time:      {format_time(total_time)}")
    print(f"  Best epoch:      {best_epoch}/{N_EPOCHS}")
    print(f"  Best Val IoU:    {best_val_iou:.4f}")
    print(f"  Best Val Dice:   {max(history['val_dice']):.4f}")
    print(f"  Best Val Acc:    {max(history['val_pixel_acc']):.4f}")
    print(f"  Lowest Val Loss: {min(history['val_loss']):.4f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
