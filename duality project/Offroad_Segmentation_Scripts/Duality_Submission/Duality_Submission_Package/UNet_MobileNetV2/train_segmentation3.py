import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
import cv2
import random
from tqdm import tqdm
import time
from pathlib import Path

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("=" * 60)
    print("ERROR: U-Net + MobileNetV2 requires 'segmentation-models-pytorch'.")
    print("Install with:  pip install segmentation-models-pytorch")
    print("=" * 60)
    raise

plt.switch_backend('Agg')

def resolve_offroad_train_val_dirs(script_path: str) -> tuple[str, str]:
    """
    Find train/val split folders that contain Color_Images/ and Segmentation/.

    Layout: <root>/Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/{train,val}

    Override: set OFFROAD_DUALITY_PROJECT to the folder that contains Offroad_Segmentation_Training_Dataset/.
    """
    script = Path(script_path).resolve()
    inner = Path("Offroad_Segmentation_Training_Dataset") / "Offroad_Segmentation_Training_Dataset"

    roots: list[Path] = []
    for key in ("OFFROAD_DUALITY_PROJECT", "OFFROAD_SEGMENTATION_DATA_ROOT"):
        raw = os.environ.get(key)
        if raw:
            roots.append(Path(raw).expanduser().resolve())

    for idx in (4, 3, 5, 2, 6, 1):
        if idx < len(script.parents):
            roots.append(script.parents[idx])

    seen: set[str] = set()
    for base in roots:
        b = str(base)
        if b in seen:
            continue
        seen.add(b)
        train_dir = base / inner / "train"
        val_dir = base / inner / "val"
        if (train_dir / "Color_Images").is_dir():
            return str(train_dir), str(val_dir)

    raise FileNotFoundError(
        "Could not find training images at "
        f"**/<root>/{inner.as_posix()}/train/Color_Images. "
        f"Script location: {script}. "
        "Set environment variable OFFROAD_DUALITY_PROJECT to your "
        "'duality project' directory (the folder that contains Offroad_Segmentation_Training_Dataset)."
    )

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = np.moveaxis(img, 0, -1)
    img  = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])

def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"

value_map = {
    0:     0,
    100:   1,
    200:   2,
    300:   3,
    500:   4,
    550:   5,
    700:   6,
    800:   7,
    7100:  8,
    10000: 9
}
n_classes = len(value_map)

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr     = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

class HRNetSegDataset(Dataset):
    """
    Dataset for semantic segmentation (image / mask pairs).
    Returns normalised image tensor (C,H,W) and long class-ID mask (H,W).
    """

    def __init__(self, data_dir, img_size):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.img_size  = img_size
        self.data_ids  = os.listdir(self.image_dir)
        self.mean      = torch.tensor(IMG_MEAN).view(3, 1, 1)
        self.std       = torch.tensor(IMG_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        h, w    = self.img_size

        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        image = image.resize((w, h), Image.BILINEAR)
        image_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_t = (image_t - self.mean) / self.std

        mask   = Image.open(os.path.join(self.masks_dir, data_id))
        mask   = convert_mask(mask)
        mask   = mask.resize((w, h), Image.NEAREST)
        mask_t = torch.from_numpy(np.array(mask)).long()

        return image_t, mask_t

def build_unet_mobilenetv2(num_classes: int, pretrained_encoder: bool = True) -> nn.Module:
    """
    U-Net segmentation model with a MobileNetV2 backbone (ImageNet weights on encoder).
    Returns logits (no activation) for CrossEntropyLoss.
    """
    encoder_weights = "imagenet" if pretrained_encoder else None
    return smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )

def compute_iou(pred_map, target_map, num_classes=10):
    """Compute mean IoU from per-pixel class ID maps (tensors)."""
    pred   = pred_map.view(-1)
    target = target_map.view(-1)
    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds   = pred   == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union        = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().item())
    return float(np.nanmean(iou_per_class))

def compute_per_class_iou(pred_map, target_map, num_classes=10):
    """Compute IoU for each class separately."""
    pred   = pred_map.view(-1)
    target = target_map.view(-1)
    iou_per_class = {}
    for class_id in range(num_classes):
        pred_inds   = pred   == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union        = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class[class_names[class_id]] = float('nan')
        else:
            iou_per_class[class_names[class_id]] = (intersection / union).cpu().item()
    return iou_per_class

def compute_dice(pred_map, target_map, num_classes=10, smooth=1e-6):
    """Compute mean Dice coefficient from per-pixel class ID maps."""
    pred   = pred_map.view(-1)
    target = target_map.view(-1)
    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds   = pred   == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        dice_score   = (2. * intersection + smooth) / (
            pred_inds.sum().float() + target_inds.sum().float() + smooth
        )
        dice_per_class.append(dice_score.cpu().item())
    return float(np.mean(dice_per_class))

def compute_pixel_accuracy(pred_map, target_map):
    """Compute pixel accuracy from per-pixel class ID maps."""
    return float((pred_map == target_map).float().mean().cpu().item())

def print_class_distribution(data_dir, num_classes):
    """Display class distribution."""
    masks_dir   = os.path.join(data_dir, 'Segmentation')
    class_counts = np.zeros(num_classes, dtype=np.float64)
    mask_files   = os.listdir(masks_dir)

    for mask_file in tqdm(mask_files, desc="  Analyzing classes", unit="mask"):
        mask_path = os.path.join(masks_dir, mask_file)
        mask      = Image.open(mask_path)
        mask      = convert_mask(mask)
        mask_np   = np.array(mask)
        for c in range(num_classes):
            class_counts[c] += (mask_np == c).sum()

    total_pixels = class_counts.sum()
    print("\n  Class Distribution:")
    print("  " + "-" * 65)
    for i in range(num_classes):
        pct = 100.0 * class_counts[i] / total_pixels
        bar = "#" * int(pct / 2)
        print(f"  {class_names[i]:>15s}: {pct:5.1f}% {bar}")
    print("  " + "-" * 65)
    return class_counts

def save_training_plots(history, output_dir):
    """Save publication-quality training metric plots."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    c_train = '#1976D2'
    c_val   = '#E53935'
    c_best  = '#4CAF50'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_loss'], label='Train', color=c_train,
                 linewidth=2, marker='o', markersize=3)
    axes[0].plot(epochs, history['val_loss'],   label='Val',   color=c_val,
                 linewidth=2, marker='s', markersize=3)
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val   = min(history['val_loss'])
    axes[0].axvline(x=best_epoch, color=c_best, linestyle='--', alpha=0.5,
                    label=f'Best (epoch {best_epoch})')
    axes[0].scatter([best_epoch], [best_val], color=c_best, s=100, zorder=5, marker='*')
    axes[0].set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_pixel_acc'], label='Train', color=c_train,
                 linewidth=2, marker='o', markersize=3)
    axes[1].plot(epochs, history['val_pixel_acc'],   label='Val',   color=c_val,
                 linewidth=2, marker='s', markersize=3)
    best_epoch = np.argmax(history['val_pixel_acc']) + 1
    best_val   = max(history['val_pixel_acc'])
    axes[1].axvline(x=best_epoch, color=c_best, linestyle='--', alpha=0.5,
                    label=f'Best (epoch {best_epoch})')
    axes[1].scatter([best_epoch], [best_val], color=c_best, s=100, zorder=5, marker='*')
    axes[1].set_title('Pixel Accuracy vs Epoch', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved training_curves.png")

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
    best_val   = max(history['val_iou'])
    axes[1].axvline(x=best_epoch, color=c_best, linestyle='--', alpha=0.5,
                    label=f'Best (epoch {best_epoch})')
    axes[1].scatter([best_epoch], [best_val], color=c_best, s=100, zorder=5, marker='*')
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
    best_val   = max(history['val_dice'])
    axes[1].axvline(x=best_epoch, color=c_best, linestyle='--', alpha=0.5,
                    label=f'Best (epoch {best_epoch})')
    axes[1].scatter([best_epoch], [best_val], color=c_best, s=100, zorder=5, marker='*')
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, history['train_loss'], label='Train', color=c_train,
                    linewidth=2, marker='o', markersize=3)
    axes[0, 0].plot(epochs, history['val_loss'],   label='Val',   color=c_val,
                    linewidth=2, marker='s', markersize=3)
    axes[0, 0].set_title('Loss', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['train_iou'], label='Train', color=c_train,
                    linewidth=2, marker='o', markersize=3)
    axes[0, 1].plot(epochs, history['val_iou'],   label='Val',   color=c_val,
                    linewidth=2, marker='s', markersize=3)
    axes[0, 1].set_title('IoU', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].plot(epochs, history['train_dice'], label='Train', color=c_train,
                    linewidth=2, marker='o', markersize=3)
    axes[1, 0].plot(epochs, history['val_dice'],   label='Val',   color=c_val,
                    linewidth=2, marker='s', markersize=3)
    axes[1, 0].set_title('Dice Score', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].plot(epochs, history['train_pixel_acc'], label='Train', color=c_train,
                    linewidth=2, marker='o', markersize=3)
    axes[1, 1].plot(epochs, history['val_pixel_acc'],   label='Val',   color=c_val,
                    linewidth=2, marker='s', markersize=3)
    axes[1, 1].set_title('Pixel Accuracy', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)

    plt.suptitle('U-Net (MobileNetV2) Training Progress', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved all_metrics_curves.png")

    if 'lr' in history:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, history['lr'], color='#7B1FA2', linewidth=2, marker='o', markersize=3)
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
        f.write("  SEGMENTATION TRAINING RESULTS (U-Net + MobileNetV2)\n")
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

def main():
    total_start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Offroad Segmentation Training")
    print(f"  U-Net + MobileNetV2  (segmentation_models_pytorch)")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    batch_size          = 8
    w, h                = 256, 256
    lr                  = 1e-4
    n_epochs            = 11
    early_stop_patience = 7
    num_workers         = 0

    print(f"  Image size:  {w}x{h}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LR:          {lr}")
    print(f"  Epochs:      {n_epochs}")
    print(f"  Patience:    {early_stop_patience}")
    print(f"  Model:       U-Net + MobileNetV2 encoder (ImageNet pretrain)")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats3')
    os.makedirs(output_dir, exist_ok=True)

    data_dir, val_dir = resolve_offroad_train_val_dirs(__file__)
    print(f"  Train split: {data_dir}")
    print(f"  Val split:   {val_dir}")

    trainset    = HRNetSegDataset(data_dir=data_dir, img_size=(h, w))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)

    valset      = HRNetSegDataset(data_dir=val_dir, img_size=(h, w))
    val_loader  = DataLoader(valset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)

    print(f"\n  Training samples:   {len(trainset)}")
    print(f"  Validation samples: {len(valset)}")

    print("\nStep 1/3: Analyzing class distribution...")
    class_counts = print_class_distribution(data_dir, n_classes)

    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"  Class weights (relative): {np.round(class_weights, 3)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, ignore_index=255)

    print("\nStep 2/3: Building U-Net (MobileNetV2 encoder)...")
    model = build_unet_mobilenetv2(num_classes=n_classes, pretrained_encoder=True)
    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} (all layers trainable)")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)

    print(f"\n  Loss:      CrossEntropyLoss (class-weighted)")
    print(f"  Optimizer: AdamW (lr={lr}, weight_decay=1e-4)")
    print(f"  Scheduler: CosineAnnealingLR (T_max={n_epochs})")

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
    best_ckpt_path   = os.path.join(script_dir, 'unet_mobilenetv2_best.pth')

    print(f"\nStep 3/3: Training for {n_epochs} epochs...")
    print("=" * 80)

    for epoch in range(n_epochs):
        epoch_start = time.time()
        current_lr  = optimizer.param_groups[0]['lr']

        model.train()
        train_losses, train_iou_list, train_dice_list, train_acc_list = [], [], [], []

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:>2}/{n_epochs} [Train]",
            leave=False, unit="batch",
        )

        for images, masks in train_pbar:
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)

            batch_iou, batch_dice, batch_acc = [], [], []
            for pred, gt in zip(preds, masks):
                batch_iou.append(compute_iou(pred, gt, n_classes))
                batch_dice.append(compute_dice(pred, gt, n_classes))
                batch_acc.append(compute_pixel_accuracy(pred, gt))

            train_losses.append(loss.item())
            train_iou_list.append(float(np.nanmean(batch_iou)))
            train_dice_list.append(float(np.mean(batch_dice)))
            train_acc_list.append(float(np.mean(batch_acc)))

            train_pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                iou=f"{np.nanmean(batch_iou):.3f}",
            )

        model.eval()
        val_losses, val_iou_list, val_dice_list, val_acc_list = [], [], [], []

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1:>2}/{n_epochs} [Val]  ",
            leave=False, unit="batch",
        )

        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device)
                masks  = masks.to(device)

                logits = model(images)
                loss   = criterion(logits, masks)
                preds  = logits.argmax(dim=1)

                val_losses.append(loss.item())

                for pred, gt in zip(preds, masks):
                    val_iou_list.append(compute_iou(pred, gt, n_classes))
                    val_dice_list.append(compute_dice(pred, gt, n_classes))
                    val_acc_list.append(compute_pixel_accuracy(pred, gt))

                val_pbar.set_postfix(loss=f"{loss.item():.3f}")

        epoch_train_loss = float(np.mean(train_losses))
        epoch_val_loss   = float(np.mean(val_losses))
        epoch_train_iou  = float(np.nanmean(train_iou_list))
        epoch_val_iou    = float(np.nanmean(val_iou_list))
        epoch_train_dice = float(np.mean(train_dice_list))
        epoch_val_dice   = float(np.mean(val_dice_list))
        epoch_train_acc  = float(np.mean(train_acc_list))
        epoch_val_acc    = float(np.mean(val_acc_list))

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(epoch_train_iou)
        history['val_iou'].append(epoch_val_iou)
        history['train_dice'].append(epoch_train_dice)
        history['val_dice'].append(epoch_val_dice)
        history['train_pixel_acc'].append(epoch_train_acc)
        history['val_pixel_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)

        scheduler.step()

        improved = ""
        if epoch_val_iou > best_val_iou:
            best_val_iou     = epoch_val_iou
            best_epoch       = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch':           epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou':    best_val_iou,
                'n_classes':       n_classes,
                'class_names':     class_names,
                'image_size':      (h, w),
                'backbone':        'unet_mobilenet_v2',
            }, best_ckpt_path)
            improved = " * NEW BEST"
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start
        remaining  = epoch_time * (n_epochs - epoch - 1)

        print(
            f"Epoch {epoch+1:>2}/{n_epochs} | "
            f"Loss: {epoch_train_loss:.4f}/{epoch_val_loss:.4f} | "
            f"Train IoU: {epoch_train_iou:.4f} | Val IoU: {epoch_val_iou:.4f} | "
            f"Val Dice: {epoch_val_dice:.4f} | Val Acc: {epoch_val_acc:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"{format_time(epoch_time)} (ETA: {format_time(remaining)})"
            f"{improved}"
        )

        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping! No improvement for {early_stop_patience} epochs.")
            print(f"  Best Val IoU: {best_val_iou:.4f} at epoch {best_epoch}")
            break

    print("\n" + "=" * 80)
    print("Evaluating best model & saving results...")

    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Restored best model from epoch {best_epoch}")

    model.eval()
    all_per_class_iou = {name: [] for name in class_names}

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="  Per-class IoU", leave=False):
            images = images.to(device)
            masks  = masks.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1)
            for pred, gt in zip(preds, masks):
                per_class = compute_per_class_iou(pred, gt, n_classes)
                for name, val in per_class.items():
                    all_per_class_iou[name].append(val)

    final_per_class_iou = {
        name: float(np.nanmean(vals)) for name, vals in all_per_class_iou.items()
    }

    print("\n  Per-Class IoU (Best Model):")
    print("  " + "-" * 40)
    for name, iou_val in final_per_class_iou.items():
        if np.isnan(iou_val):
            print(f"  {name:>15s}:  N/A")
        else:
            bar = "#" * int(iou_val * 20)
            print(f"  {name:>15s}: {iou_val:.4f} {bar}")
    print("  " + "-" * 40)

    print("\nSaving plots...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir, per_class_iou=final_per_class_iou)

    print(f"  Best checkpoint: '{best_ckpt_path}'")

    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE (U-Net + MobileNetV2)")
    print(f"{'='*60}")
    print(f"  Total time:      {format_time(total_time)}")
    print(f"  Best epoch:      {best_epoch}/{n_epochs}")
    print(f"  Best Val IoU:    {best_val_iou:.4f}")
    print(f"  Best Val Dice:   {max(history['val_dice']):.4f}")
    print(f"  Best Val Acc:    {max(history['val_pixel_acc']):.4f}")
    print(f"  Lowest Val Loss: {min(history['val_loss']):.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
