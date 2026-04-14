"""
Segmentation Training Script
Converted from train_mask.ipynb
Trains a segmentation head on top of DINOv2 backbone

Key improvements over original:
  - Feature caching: backbone features computed once (huge CPU speedup)
  - Fixed mask interpolation: NEAREST instead of bilinear (critical bug fix)
  - Deeper segmentation head with BatchNorm + residual connections
  - AdamW optimizer with cosine LR schedule
  - Class-weighted CrossEntropyLoss for imbalanced classes
  - Early stopping with best model checkpointing
  - 20 epochs of training for proper convergence
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import cv2
import os
import torchvision
from tqdm import tqdm
import time

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"


# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0,        # Background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = len(value_map)

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255

        return image, mask


# ============================================================================
# Model: Segmentation Head (Improved ConvNeXt-style)
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    """
    Improved segmentation head with:
    - Wider hidden dimension (256 channels)
    - BatchNorm for training stability
    - Two ConvNeXt blocks with residual connections
    - Dropout for regularization
    """

    def __init__(self, in_channels, out_channels, tokenW, tokenH, dropout=0.1):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        hidden_dim = 256

        # Projection from backbone embedding to hidden dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # Block 1: Depthwise separable convolution with residual
        self.block1_dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7,
                                   padding=3, groups=hidden_dim)
        self.block1_norm1 = nn.BatchNorm2d(hidden_dim)
        self.block1_pw = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1),
        )
        self.block1_norm2 = nn.BatchNorm2d(hidden_dim)

        # Block 2: Depthwise separable convolution with residual
        self.block2_dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7,
                                   padding=3, groups=hidden_dim)
        self.block2_norm1 = nn.BatchNorm2d(hidden_dim)
        self.block2_pw = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1),
        )
        self.block2_norm2 = nn.BatchNorm2d(hidden_dim)

        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(hidden_dim, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)

        # Block 1 with residual
        residual = x
        x = self.block1_dw(x)
        x = self.block1_norm1(x)
        x = self.block1_pw(x)
        x = self.block1_norm2(x + residual)

        # Block 2 with residual
        residual = x
        x = self.block2_dw(x)
        x = self.block2_norm1(x)
        x = self.block2_pw(x)
        x = self.block2_norm2(x + residual)

        x = self.dropout(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute mean IoU across all classes."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class)


def compute_per_class_iou(pred, target, num_classes=10):
    """Compute IoU for each class separately."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = {}
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class[class_names[class_id]] = float('nan')
        else:
            iou_per_class[class_names[class_id]] = (intersection / union).cpu().item()

    return iou_per_class


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute mean Dice coefficient across all classes."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (
            pred_inds.sum().float() + target_inds.sum().float() + smooth
        )
        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


# ============================================================================
# Feature Caching (critical for CPU training speed)
# ============================================================================

def precompute_features(backbone, data_loader, device):
    """
    Pre-compute and cache backbone features for all images.

    Since the backbone is frozen, features are identical every epoch.
    Caching avoids redundant backbone forward passes, reducing training
    time from ~54 hours to ~4 hours on CPU.
    """
    all_features = []
    all_labels = []

    backbone.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="  Caching features", unit="batch"):
            imgs = imgs.to(device)
            features = backbone.forward_features(imgs)["x_norm_patchtokens"]
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    feat_mb = features_tensor.element_size() * features_tensor.nelement() / 1e6
    label_mb = labels_tensor.element_size() * labels_tensor.nelement() / 1e6
    print(f"  Features: {features_tensor.shape} ({feat_mb:.1f} MB)")
    print(f"  Labels:   {labels_tensor.shape} ({label_mb:.1f} MB)")

    return features_tensor, labels_tensor


def compute_class_weights(data_dir, num_classes):
    """Compute inverse-frequency class weights from training masks."""
    masks_dir = os.path.join(data_dir, 'Segmentation')
    class_counts = np.zeros(num_classes, dtype=np.float64)
    mask_files = os.listdir(masks_dir)

    for mask_file in tqdm(mask_files, desc="Computing class weights", unit="mask"):
        mask_path = os.path.join(masks_dir, mask_file)
        mask = Image.open(mask_path)
        mask = convert_mask(mask)
        mask_np = np.array(mask)
        for c in range(num_classes):
            class_counts[c] += (mask_np == c).sum()

    # Inverse frequency weighting
    total_pixels = class_counts.sum()
    weights = total_pixels / (num_classes * class_counts + 1e-6)

    # Clip to prevent extreme weights causing instability
    weights = np.clip(weights, 0.5, 10.0)

    # Print class distribution
    print("\n  Class Distribution:")
    print("  " + "-" * 65)
    for i in range(num_classes):
        pct = 100.0 * class_counts[i] / total_pixels
        bar = "█" * int(pct / 2)
        print(f"  {class_names[i]:>15s}: {pct:5.1f}% {bar:25s} (w={weights[i]:.2f})")
    print("  " + "-" * 65)

    return torch.FloatTensor(weights)


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save publication-quality training metric plots."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # Color palette
    c_train = '#1976D2'   # Blue
    c_val = '#E53935'     # Red
    c_best = '#4CAF50'    # Green

    # ── Plot 1: Loss + Accuracy ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_loss'], label='Train', color=c_train,
                 linewidth=2, marker='o', markersize=3)
    axes[0].plot(epochs, history['val_loss'], label='Val', color=c_val,
                 linewidth=2, marker='s', markersize=3)
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val = min(history['val_loss'])
    axes[0].axvline(x=best_epoch, color=c_best, linestyle='--', alpha=0.5,
                    label=f'Best (epoch {best_epoch})')
    axes[0].scatter([best_epoch], [best_val], color=c_best, s=100, zorder=5,
                    marker='*')
    axes[0].set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_pixel_acc'], label='Train', color=c_train,
                 linewidth=2, marker='o', markersize=3)
    axes[1].plot(epochs, history['val_pixel_acc'], label='Val', color=c_val,
                 linewidth=2, marker='s', markersize=3)
    best_epoch = np.argmax(history['val_pixel_acc']) + 1
    best_val = max(history['val_pixel_acc'])
    axes[1].axvline(x=best_epoch, color=c_best, linestyle='--', alpha=0.5,
                    label=f'Best (epoch {best_epoch})')
    axes[1].scatter([best_epoch], [best_val], color=c_best, s=100, zorder=5,
                    marker='*')
    axes[1].set_title('Pixel Accuracy vs Epoch', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training_curves.png")

    # ── Plot 2: IoU curves ───────────────────────────────────────────────
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
    best_epoch = np.argmax(history['val_iou']) + 1
    best_val = max(history['val_iou'])
    axes[1].axvline(x=best_epoch, color=c_best, linestyle='--', alpha=0.5,
                    label=f'Best (epoch {best_epoch})')
    axes[1].scatter([best_epoch], [best_val], color=c_best, s=100, zorder=5,
                    marker='*')
    axes[1].set_title('Validation IoU vs Epoch', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('IoU', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved iou_curves.png")

    # ── Plot 3: Dice curves ──────────────────────────────────────────────
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
    best_epoch = np.argmax(history['val_dice']) + 1
    best_val = max(history['val_dice'])
    axes[1].axvline(x=best_epoch, color=c_best, linestyle='--', alpha=0.5,
                    label=f'Best (epoch {best_epoch})')
    axes[1].scatter([best_epoch], [best_val], color=c_best, s=100, zorder=5,
                    marker='*')
    axes[1].set_title('Validation Dice vs Epoch', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Score', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved dice_curves.png")

    # ── Plot 4: Combined 2x2 metrics ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', color=c_train,
                    linewidth=2, marker='o', markersize=3)
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', color=c_val,
                    linewidth=2, marker='s', markersize=3)
    axes[0, 0].set_title('Loss vs Epoch', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # IoU
    axes[0, 1].plot(epochs, history['train_iou'], label='Train', color=c_train,
                    linewidth=2, marker='o', markersize=3)
    axes[0, 1].plot(epochs, history['val_iou'], label='Val', color=c_val,
                    linewidth=2, marker='s', markersize=3)
    axes[0, 1].set_title('IoU vs Epoch', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    # Dice
    axes[1, 0].plot(epochs, history['train_dice'], label='Train', color=c_train,
                    linewidth=2, marker='o', markersize=3)
    axes[1, 0].plot(epochs, history['val_dice'], label='Val', color=c_val,
                    linewidth=2, marker='s', markersize=3)
    axes[1, 0].set_title('Dice Score vs Epoch', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    # Pixel Accuracy
    axes[1, 1].plot(epochs, history['train_pixel_acc'], label='Train', color=c_train,
                    linewidth=2, marker='o', markersize=3)
    axes[1, 1].plot(epochs, history['val_pixel_acc'], label='Val', color=c_val,
                    linewidth=2, marker='s', markersize=3)
    axes[1, 1].set_title('Pixel Accuracy vs Epoch', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Pixel Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)

    plt.suptitle('Training Progress - All Metrics', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved all_metrics_curves.png")

    # ── Plot 5: Learning Rate curve ──────────────────────────────────────
    if 'lr' in history:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, history['lr'], color='#7B1FA2', linewidth=2,
                 marker='o', markersize=3)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lr_schedule.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved lr_schedule.png")


def save_history_to_file(history, output_dir, per_class_iou=None):
    """Save comprehensive training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  SEGMENTATION TRAINING RESULTS\n")
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
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 80 + "\n\n")

        # Per-class IoU breakdown
        if per_class_iou is not None:
            f.write("Per-Class IoU (Best Model):\n")
            f.write("-" * 40 + "\n")
            for name, iou_val in per_class_iou.items():
                if np.isnan(iou_val):
                    f.write(f"  {name:>15s}:    N/A\n")
                else:
                    bar = "█" * int(iou_val * 20)
                    f.write(f"  {name:>15s}: {iou_val:.4f} {bar}\n")
            f.write("-" * 40 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 110 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc', 'LR']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 110 + "\n")

        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            lr_val = history['lr'][i] if 'lr' in history else 0
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.6f}\n".format(
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_iou'][i],
                history['val_iou'][i],
                history['train_dice'][i],
                history['val_dice'][i],
                history['train_pixel_acc'][i],
                history['val_pixel_acc'][i],
                lr_val
            ))

    print(f"  Saved evaluation_metrics.txt")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    total_start_time = time.time()

    # ── Configuration ────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Offroad Segmentation Training (DINOv2 + ConvNeXt Head)")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # Hyperparameters
    batch_size = 2
    w = int(((960 / 2) // 14) * 14)   # 476
    h = int(((540 / 2) // 14) * 14)   # 266
    lr = 1e-3
    n_epochs = 11
    early_stop_patience = 7

    print(f"  Image size: {w}x{h}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Early stopping patience: {early_stop_patience}")

    # Output directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # ── Transforms ───────────────────────────────────────────────────────
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Mask transform — CRITICAL: use NEAREST interpolation to preserve class IDs
    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # ── Dataset Paths ────────────────────────────────────────────────────
    data_dir = os.path.join(script_dir, '..', '..', 'Offroad_Segmentation_Training_Dataset',
                            'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, '..', '..', 'Offroad_Segmentation_Training_Dataset',
                           'Offroad_Segmentation_Training_Dataset', 'val')

    # Create datasets
    trainset = MaskDataset(data_dir=data_dir, transform=transform,
                           mask_transform=mask_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False,
                              num_workers=0)

    valset = MaskDataset(data_dir=val_dir, transform=transform,
                         mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    print(f"\n  Training samples: {len(trainset)}")
    print(f"  Validation samples: {len(valset)}")

    # ── Compute class weights ────────────────────────────────────────────
    print("\nStep 1/4: Computing class weights...")
    class_weights = compute_class_weights(data_dir, n_classes)

    # ── Load DINOv2 backbone ─────────────────────────────────────────────
    print("\nStep 2/4: Loading DINOv2 backbone...")
    BACKBONE_SIZE = "small"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2",
                                    model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)
    print(f"  Backbone: {backbone_name} (frozen)")

    # Get embedding dimension
    with torch.no_grad():
        sample_imgs, _ = next(iter(train_loader))
        sample_output = backbone_model.forward_features(
            sample_imgs.to(device)
        )["x_norm_patchtokens"]
    n_embedding = sample_output.shape[2]
    tokenH, tokenW = h // 14, w // 14
    print(f"  Embedding dim: {n_embedding}")
    print(f"  Patch grid: {tokenH}x{tokenW} = {tokenH * tokenW} tokens")

    # ── Pre-compute features (one-time cost, huge CPU speedup) ───────────
    print("\nStep 3/4: Pre-computing backbone features (one-time cost)...")
    cache_start = time.time()

    print("  Training set:")
    train_features, train_labels = precompute_features(
        backbone_model, train_loader, device
    )
    print("  Validation set:")
    val_features, val_labels = precompute_features(
        backbone_model, val_loader, device
    )

    cache_time = time.time() - cache_start
    print(f"  Feature caching completed in {format_time(cache_time)}")

    # Free backbone from memory (no longer needed)
    del backbone_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  Backbone model freed from memory")

    # Create cached data loaders
    cached_train_dataset = TensorDataset(train_features, train_labels)
    cached_val_dataset = TensorDataset(val_features, val_labels)

    cached_train_loader = DataLoader(cached_train_dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=0)
    cached_val_loader = DataLoader(cached_val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=0)

    # ── Create segmentation head ─────────────────────────────────────────
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=tokenW,
        tokenH=tokenH,
        dropout=0.1
    )
    classifier = classifier.to(device)

    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"\n  Segmentation head: {total_params:,} params ({trainable_params:,} trainable)")

    # ── Loss, Optimizer, Scheduler ───────────────────────────────────────
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )

    print(f"  Loss: CrossEntropyLoss (class-weighted)")
    print(f"  Optimizer: AdamW (lr={lr}, weight_decay=0.01)")
    print(f"  Scheduler: CosineAnnealingLR (T_max={n_epochs})")

    # Target size for interpolation (image resolution)
    target_size = (h, w)

    # ── Training history ─────────────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': [],
        'train_pixel_acc': [], 'val_pixel_acc': [],
        'lr': []
    }

    # Early stopping state
    best_val_iou = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\nStep 4/4: Training for {n_epochs} epochs...")
    print("=" * 80)

    for epoch in range(n_epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # ── Training phase ───────────────────────────────────────────
        classifier.train()
        train_losses = []
        train_iou_accum = []
        train_dice_accum = []
        train_acc_accum = []

        train_pbar = tqdm(cached_train_loader,
                          desc=f"Epoch {epoch+1:>2}/{n_epochs} [Train]",
                          leave=False, unit="batch")

        for features, labels in train_pbar:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = classifier(features)
            outputs = F.interpolate(logits, size=target_size,
                                    mode="bilinear", align_corners=False)
            labels_sq = labels.squeeze(dim=1).long()

            loss = loss_fct(outputs, labels_sq)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Compute training metrics inline (avoid separate evaluation pass)
            with torch.no_grad():
                iou = compute_iou(outputs, labels_sq, num_classes=n_classes)
                dice = compute_dice(outputs, labels_sq, num_classes=n_classes)
                acc = compute_pixel_accuracy(outputs, labels_sq)
                train_iou_accum.append(iou)
                train_dice_accum.append(dice)
                train_acc_accum.append(acc)

            train_pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                iou=f"{iou:.3f}"
            )

        # ── Validation phase ─────────────────────────────────────────
        classifier.eval()
        val_losses = []
        val_iou_accum = []
        val_dice_accum = []
        val_acc_accum = []

        val_pbar = tqdm(cached_val_loader,
                        desc=f"Epoch {epoch+1:>2}/{n_epochs} [Val]  ",
                        leave=False, unit="batch")

        with torch.no_grad():
            for features, labels in val_pbar:
                features, labels = features.to(device), labels.to(device)

                logits = classifier(features)
                outputs = F.interpolate(logits, size=target_size,
                                        mode="bilinear", align_corners=False)
                labels_sq = labels.squeeze(dim=1).long()

                loss = loss_fct(outputs, labels_sq)
                val_losses.append(loss.item())

                iou = compute_iou(outputs, labels_sq, num_classes=n_classes)
                dice = compute_dice(outputs, labels_sq, num_classes=n_classes)
                acc = compute_pixel_accuracy(outputs, labels_sq)
                val_iou_accum.append(iou)
                val_dice_accum.append(dice)
                val_acc_accum.append(acc)

                val_pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    iou=f"{iou:.3f}"
                )

        # ── Aggregate epoch metrics ──────────────────────────────────
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_iou = np.nanmean(train_iou_accum)
        epoch_val_iou = np.nanmean(val_iou_accum)
        epoch_train_dice = np.mean(train_dice_accum)
        epoch_val_dice = np.mean(val_dice_accum)
        epoch_train_acc = np.mean(train_acc_accum)
        epoch_val_acc = np.mean(val_acc_accum)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(epoch_train_iou)
        history['val_iou'].append(epoch_val_iou)
        history['train_dice'].append(epoch_train_dice)
        history['val_dice'].append(epoch_val_dice)
        history['train_pixel_acc'].append(epoch_train_acc)
        history['val_pixel_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)

        # Step scheduler
        scheduler.step()

        # ── Early stopping check ─────────────────────────────────────
        improved = ""
        if epoch_val_iou > best_val_iou:
            best_val_iou = epoch_val_iou
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = classifier.state_dict().copy()
            improved = " ★ NEW BEST"
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start
        remaining = epoch_time * (n_epochs - epoch - 1)

        # Print epoch summary
        print(f"Epoch {epoch+1:>2}/{n_epochs} │ "
              f"Loss: {epoch_train_loss:.4f}/{epoch_val_loss:.4f} │ "
              f"IoU: {epoch_train_iou:.4f}/{epoch_val_iou:.4f} │ "
              f"Dice: {epoch_train_dice:.4f}/{epoch_val_dice:.4f} │ "
              f"Acc: {epoch_train_acc:.4f}/{epoch_val_acc:.4f} │ "
              f"LR: {current_lr:.6f} │ "
              f"{format_time(epoch_time)} (ETA: {format_time(remaining)})"
              f"{improved}")

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n  ⚠ Early stopping triggered! No improvement for "
                  f"{early_stop_patience} epochs.")
            print(f"  Best Val IoU: {best_val_iou:.4f} at epoch {best_epoch}")
            break

    # ── Save best model ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Saving results...")

    # Restore best model
    if best_model_state is not None:
        classifier.load_state_dict(best_model_state)
        print(f"  Restored best model from epoch {best_epoch}")

    # Compute per-class IoU with best model
    classifier.eval()
    all_per_class_iou = {name: [] for name in class_names}
    with torch.no_grad():
        for features, labels in cached_val_loader:
            features, labels = features.to(device), labels.to(device)
            logits = classifier(features)
            outputs = F.interpolate(logits, size=target_size,
                                    mode="bilinear", align_corners=False)
            labels_sq = labels.squeeze(dim=1).long()
            batch_per_class = compute_per_class_iou(outputs, labels_sq,
                                                     num_classes=n_classes)
            for name, val in batch_per_class.items():
                all_per_class_iou[name].append(val)

    final_per_class_iou = {
        name: np.nanmean(vals) for name, vals in all_per_class_iou.items()
    }

    # Print per-class IoU
    print("\n  Per-Class IoU (Best Model):")
    print("  " + "-" * 40)
    for name, iou_val in final_per_class_iou.items():
        if np.isnan(iou_val):
            print(f"  {name:>15s}:  N/A")
        else:
            bar = "█" * int(iou_val * 20)
            print(f"  {name:>15s}: {iou_val:.4f} {bar}")
    print("  " + "-" * 40)

    # Save plots
    print("\nSaving plots...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir, per_class_iou=final_per_class_iou)

    # Save model
    model_path = os.path.join(script_dir, "segmentation_head.pth")
    torch.save(classifier.state_dict(), model_path)
    print(f"  Saved model to '{model_path}'")

    # Also save with metadata
    checkpoint_path = os.path.join(script_dir, "segmentation_checkpoint.pth")
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'best_val_iou': best_val_iou,
        'best_epoch': best_epoch,
        'n_classes': n_classes,
        'n_embedding': n_embedding,
        'tokenW': tokenW,
        'tokenH': tokenH,
        'history': history,
        'class_names': class_names,
    }, checkpoint_path)
    print(f"  Saved checkpoint to '{checkpoint_path}'")

    # ── Final summary ────────────────────────────────────────────────────
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time:     {format_time(total_time)}")
    print(f"  Best epoch:     {best_epoch}/{n_epochs}")
    print(f"  Best Val IoU:   {best_val_iou:.4f}")
    print(f"  Best Val Dice:  {max(history['val_dice']):.4f}")
    print(f"  Best Val Acc:   {max(history['val_pixel_acc']):.4f}")
    print(f"  Lowest Val Loss: {min(history['val_loss']):.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
