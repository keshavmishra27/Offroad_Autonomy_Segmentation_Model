"""
Segmentation Training Script — Project 2 (DeepLabV3+)
Trains a DeepLabV3+ model (torchvision) for offroad segmentation.

CPU-optimised strategy (same idea as Project 1):
  - Uses DeepLabV3 with MobileNetV3-Large backbone (lightweight, fast)
  - Backbone is FROZEN — features are pre-computed once and cached in RAM
  - Only the DeepLabV3 classifier head (ASPP) is trained on cached features
  - This reduces each epoch from ~3 hours to ~5-10 minutes on CPU
  - Same dataset, metrics, plotting, and class-weighted loss as Project 1
  - AdamW optimizer with cosine LR schedule
  - Early stopping with best model checkpointing
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import OrderedDict
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)
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
# Model: DeepLabV3+ (MobileNetV3-Large — CPU-friendly)
# ============================================================================

def build_deeplabv3plus(num_classes, pretrained=True):
    """
    Build a DeepLabV3 model with MobileNetV3-Large backbone from torchvision.
    MobileNetV3 is ~10x faster than ResNet-101 on CPU.
    Replaces the final classifier head to output the desired number of classes.
    """
    if pretrained:
        model = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )
    else:
        model = deeplabv3_mobilenet_v3_large(weights=None)

    # Replace the classifier head for our number of classes
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead
    # MobileNetV3-Large outputs 960 channels from the backbone
    model.classifier = DeepLabHead(960, num_classes)

    # Also replace the auxiliary classifier if present
    if model.aux_classifier is not None:
        from torchvision.models.segmentation.fcn import FCNHead
        model.aux_classifier = FCNHead(40, num_classes)

    return model


# ============================================================================
# Feature Caching (critical for CPU training speed)
# ============================================================================

def precompute_backbone_features(backbone, data_loader, device):
    """
    Pre-compute and cache backbone features for all images.

    Since the backbone is frozen, features are identical every epoch.
    Caching avoids redundant backbone forward passes, reducing training
    time from hours to minutes on CPU.

    The backbone returns an OrderedDict. We cache:
      - 'out': main feature map (for classifier head)
      - 'low' / other keys: if present, for auxiliary head
    """
    all_main_features = []
    all_aux_features = []
    all_labels = []
    has_aux = False

    backbone.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="  Caching features", unit="batch"):
            imgs = imgs.to(device)
            features = backbone(imgs)

            # features is an OrderedDict, typically with keys
            # for MobileNetV3: key '0' -> low-level, key '1' -> high-level (main)
            keys = list(features.keys())
            # The last key is the main output
            main_key = keys[-1]
            all_main_features.append(features[main_key].cpu())

            # Cache aux features if there's more than one key
            if len(keys) > 1:
                has_aux = True
                aux_key = keys[0]
                all_aux_features.append(features[aux_key].cpu())

            all_labels.append(labels.cpu())

    main_tensor = torch.cat(all_main_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    feat_mb = main_tensor.element_size() * main_tensor.nelement() / 1e6
    label_mb = labels_tensor.element_size() * labels_tensor.nelement() / 1e6
    print(f"  Main features: {main_tensor.shape} ({feat_mb:.1f} MB)")
    print(f"  Labels:        {labels_tensor.shape} ({label_mb:.1f} MB)")

    if has_aux:
        aux_tensor = torch.cat(all_aux_features, dim=0)
        aux_mb = aux_tensor.element_size() * aux_tensor.nelement() / 1e6
        print(f"  Aux features:  {aux_tensor.shape} ({aux_mb:.1f} MB)")
        return main_tensor, aux_tensor, labels_tensor
    else:
        return main_tensor, None, labels_tensor


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
# Class Weight Computation
# ============================================================================

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
        bar = "#" * int(pct / 2)
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

    plt.suptitle('Training Progress - All Metrics (DeepLabV3+)', fontsize=15, fontweight='bold', y=1.01)
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
        f.write("  SEGMENTATION TRAINING RESULTS (DeepLabV3+ MobileNetV3)\n")
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
                    bar = "#" * int(iou_val * 20)
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
    print(f"  Offroad Segmentation Training")
    print(f"  DeepLabV3+ (MobileNetV3-Large, frozen backbone + caching)")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # Hyperparameters
    batch_size = 2
    # DeepLabV3+ works with arbitrary sizes, keeping consistent with project 1
    w = int(((960 / 2) // 14) * 14)   # 476
    h = int(((540 / 2) // 14) * 14)   # 266
    lr = 1e-3   # Higher LR since we're only training the head
    n_epochs = 11
    early_stop_patience = 7

    print(f"  Image size: {w}x{h}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Early stopping patience: {early_stop_patience}")

    # Output directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats2')
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

    # ── Load DeepLabV3+ model ────────────────────────────────────────────
    print("\nStep 2/4: Loading DeepLabV3+ (MobileNetV3-Large, pre-trained)...")
    model = build_deeplabv3plus(num_classes=n_classes, pretrained=True)
    model.to(device)

    # Freeze the entire backbone — only train classifier & aux_classifier
    for param in model.backbone.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} (backbone frozen)")

    # ── Pre-compute backbone features (one-time cost) ────────────────────
    print("\nStep 3/4: Pre-computing backbone features (one-time cost)...")
    cache_start = time.time()

    # Extract just the backbone for caching
    backbone = model.backbone
    backbone.eval()

    print("  Training set:")
    train_main, train_aux, train_labels = precompute_backbone_features(
        backbone, train_loader, device
    )
    print("  Validation set:")
    val_main, val_aux, val_labels = precompute_backbone_features(
        backbone, val_loader, device
    )

    cache_time = time.time() - cache_start
    print(f"  Feature caching completed in {format_time(cache_time)}")

    # Create cached data loaders
    # We need to pass both main and aux features if aux exists
    has_aux = train_aux is not None
    if has_aux:
        cached_train_dataset = TensorDataset(train_main, train_aux, train_labels)
        cached_val_dataset = TensorDataset(val_main, val_aux, val_labels)
    else:
        cached_train_dataset = TensorDataset(train_main, train_labels)
        cached_val_dataset = TensorDataset(val_main, val_labels)

    cached_train_loader = DataLoader(cached_train_dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=0,
                                     drop_last=True)
    cached_val_loader = DataLoader(cached_val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=0)

    # ── Get classifier and aux_classifier heads ──────────────────────────
    classifier_head = model.classifier
    aux_head = model.aux_classifier if has_aux and model.aux_classifier is not None else None

    # Target size for upsampling predictions to match labels
    target_size = (h, w)

    # ── Loss, Optimizer, Scheduler ───────────────────────────────────────
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Only optimise heads (backbone is frozen)
    trainable_modules = list(classifier_head.parameters())
    if aux_head is not None:
        trainable_modules += list(aux_head.parameters())

    optimizer = optim.AdamW(trainable_modules, lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )

    print(f"  Loss: CrossEntropyLoss (class-weighted)")
    print(f"  Optimizer: AdamW (lr={lr}, weight_decay=0.01)")
    print(f"  Scheduler: CosineAnnealingLR (T_max={n_epochs})")

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
    best_classifier_state = None
    best_aux_state = None

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\nStep 4/4: Training for {n_epochs} epochs...")
    print("=" * 80)

    for epoch in range(n_epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # ── Training phase ───────────────────────────────────────────
        classifier_head.train()
        if aux_head is not None:
            aux_head.train()

        train_losses = []
        train_iou_accum = []
        train_dice_accum = []
        train_acc_accum = []

        train_pbar = tqdm(cached_train_loader,
                          desc=f"Epoch {epoch+1:>2}/{n_epochs} [Train]",
                          leave=False, unit="batch")

        for batch_data in train_pbar:
            if has_aux:
                main_feat, aux_feat, labels = batch_data
                main_feat = main_feat.to(device)
                aux_feat = aux_feat.to(device)
            else:
                main_feat, labels = batch_data
                main_feat = main_feat.to(device)

            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward through classifier head only
            logits = classifier_head(main_feat)

            # Upsample to target size
            outputs = F.interpolate(logits, size=target_size,
                                    mode="bilinear", align_corners=False)

            labels_sq = labels.squeeze(dim=1).long()

            # Main loss
            loss = loss_fct(outputs, labels_sq)

            # Auxiliary loss (if present)
            if has_aux and aux_head is not None:
                aux_logits = aux_head(aux_feat)
                aux_outputs = F.interpolate(aux_logits, size=target_size,
                                            mode="bilinear", align_corners=False)
                loss = loss + 0.4 * loss_fct(aux_outputs, labels_sq)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Compute training metrics inline
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
        classifier_head.eval()
        if aux_head is not None:
            aux_head.eval()

        val_losses = []
        val_iou_accum = []
        val_dice_accum = []
        val_acc_accum = []

        val_pbar = tqdm(cached_val_loader,
                        desc=f"Epoch {epoch+1:>2}/{n_epochs} [Val]  ",
                        leave=False, unit="batch")

        with torch.no_grad():
            for batch_data in val_pbar:
                if has_aux:
                    main_feat, aux_feat, labels = batch_data
                    main_feat = main_feat.to(device)
                    aux_feat = aux_feat.to(device)
                else:
                    main_feat, labels = batch_data
                    main_feat = main_feat.to(device)

                labels = labels.to(device)

                logits = classifier_head(main_feat)
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
            best_classifier_state = {k: v.cpu().clone()
                                     for k, v in classifier_head.state_dict().items()}
            if aux_head is not None:
                best_aux_state = {k: v.cpu().clone()
                                  for k, v in aux_head.state_dict().items()}
            improved = " * NEW BEST"
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start
        remaining = epoch_time * (n_epochs - epoch - 1)

        # Print epoch summary
        print(f"Epoch {epoch+1:>2}/{n_epochs} | "
              f"Loss: {epoch_train_loss:.4f}/{epoch_val_loss:.4f} | "
              f"IoU: {epoch_train_iou:.4f}/{epoch_val_iou:.4f} | "
              f"Dice: {epoch_train_dice:.4f}/{epoch_val_dice:.4f} | "
              f"Acc: {epoch_train_acc:.4f}/{epoch_val_acc:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"{format_time(epoch_time)} (ETA: {format_time(remaining)})"
              f"{improved}")

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping triggered! No improvement for "
                  f"{early_stop_patience} epochs.")
            print(f"  Best Val IoU: {best_val_iou:.4f} at epoch {best_epoch}")
            break

    # ── Save best model ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Saving results...")

    # Restore best classifier head
    if best_classifier_state is not None:
        classifier_head.load_state_dict(best_classifier_state)
        if aux_head is not None and best_aux_state is not None:
            aux_head.load_state_dict(best_aux_state)
        print(f"  Restored best model from epoch {best_epoch}")

    # Compute per-class IoU with best model
    classifier_head.eval()
    all_per_class_iou = {name: [] for name in class_names}
    with torch.no_grad():
        for batch_data in cached_val_loader:
            if has_aux:
                main_feat, aux_feat, labels = batch_data
            else:
                main_feat, labels = batch_data
            main_feat = main_feat.to(device)
            labels = labels.to(device)

            logits = classifier_head(main_feat)
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
            bar = "#" * int(iou_val * 20)
            print(f"  {name:>15s}: {iou_val:.4f} {bar}")
    print("  " + "-" * 40)

    # Save plots
    print("\nSaving plots...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir, per_class_iou=final_per_class_iou)

    # Save the full model (backbone + trained heads) for inference
    # Reassemble the model with the best head weights
    model.classifier.load_state_dict(best_classifier_state)
    if aux_head is not None and best_aux_state is not None:
        model.aux_classifier.load_state_dict(best_aux_state)

    model_path = os.path.join(script_dir, "deeplabv3plus_segmentation.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  Saved model to '{model_path}'")

    # Also save with metadata
    checkpoint_path = os.path.join(script_dir, "deeplabv3plus_checkpoint.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_val_iou': best_val_iou,
        'best_epoch': best_epoch,
        'n_classes': n_classes,
        'history': history,
        'class_names': class_names,
        'image_size': (h, w),
        'backbone': 'mobilenet_v3_large',
    }, checkpoint_path)
    print(f"  Saved checkpoint to '{checkpoint_path}'")

    # ── Final summary ────────────────────────────────────────────────────
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE (DeepLabV3+ MobileNetV3)")
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
