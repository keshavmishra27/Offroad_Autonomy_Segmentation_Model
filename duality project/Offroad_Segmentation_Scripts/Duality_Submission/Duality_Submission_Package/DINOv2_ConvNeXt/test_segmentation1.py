import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm

plt.switch_backend('Agg')


def resolve_offroad_test_images_dir(script_path: str) -> str:
    """Default test bundle: <root>/Offroad_Segmentation_testImages/Offroad_Segmentation_testImages."""
    script = Path(script_path).resolve()
    inner = Path("Offroad_Segmentation_testImages") / "Offroad_Segmentation_testImages"

    roots: list[Path] = []
    for key in ("OFFROAD_DUALITY_PROJECT", "OFFROAD_SEGMENTATION_TEST_IMAGES"):
        raw = os.environ.get(key)
        if raw:
            roots.append(Path(raw).expanduser().resolve())

    for idx in (4, 3, 5, 2, 6, 1):
        if idx < len(script.parents):
            roots.append(script.parents[idx])

    seen: set[str] = set()
    for base in roots:
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        cand = base / inner
        if cand.is_dir():
            return str(cand)

    raise FileNotFoundError(
        f"Could not find {inner.as_posix()} under ancestors of {script}. "
        "Set OFFROAD_DUALITY_PROJECT to your 'duality project' folder."
    )


def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, img[:, :, ::-1])

value_map = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

n_classes = len(value_map)

color_palette = np.array([
    [0, 0, 0],
    [34, 139, 34],
    [0, 255, 0],
    [210, 180, 140],
    [139, 90, 43],
    [128, 128, 0],
    [139, 69, 19],
    [128, 128, 128],
    [160, 82, 45],
    [135, 206, 235],
], dtype=np.uint8)

def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

def mask_to_color(mask):
    """Convert a class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask

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

        return image, mask, data_id

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH, dropout=0.1):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        hidden_dim = 256

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        self.block1_dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7,
                                   padding=3, groups=hidden_dim)
        self.block1_norm1 = nn.BatchNorm2d(hidden_dim)
        self.block1_pw = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1),
        )
        self.block1_norm2 = nn.BatchNorm2d(hidden_dim)

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

        residual = x
        x = self.block1_dw(x)
        x = self.block1_norm1(x)
        x = self.block1_pw(x)
        x = self.block1_norm2(x + residual)

        residual = x
        x = self.block2_dw(x)
        x = self.block2_norm1(x)
        x = self.block2_pw(x)
        x = self.block2_norm2(x + residual)

        x = self.dropout(x)
        return self.classifier(x)

class SegmentationHeadLegacy(nn.Module):
    """
    Backwards-compatible segmentation head used by older checkpoints.

    Expected parameter names include:
    - stem.0.weight / stem.0.bias  (Conv2d with 7x7 kernel)
    - block.0.* and block.2.*      (Conv2d -> GELU -> Conv2d)
    - classifier.*                 (1x1 Conv2d to logits)
    """
    def __init__(self, in_channels, out_channels, tokenW, tokenH, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
        )

        self.block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(hidden_dim, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        x = self.dropout(x)
        return self.classifier(x)

def _unwrap_state_dict(loaded_obj):
    if isinstance(loaded_obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in loaded_obj and isinstance(loaded_obj[k], dict):
                return loaded_obj[k]
    return loaded_obj

def build_classifier_for_checkpoint(state_dict, *, in_channels, out_channels, tokenW, tokenH):
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint did not contain a state_dict dict.")

    is_legacy = any(k.startswith("block.") for k in state_dict.keys())
    if is_legacy:
        stem_w = state_dict.get("stem.0.weight", None)
        if stem_w is None or not hasattr(stem_w, "shape") or len(stem_w.shape) < 1:
            hidden_dim = 128
        else:
            hidden_dim = int(stem_w.shape[0])
        return SegmentationHeadLegacy(
            in_channels=in_channels,
            out_channels=out_channels,
            tokenW=tokenW,
            tokenH=tokenH,
            hidden_dim=hidden_dim,
        )

    return SegmentationHeadConvNeXt(
        in_channels=in_channels,
        out_channels=out_channels,
        tokenW=tokenW,
        tokenH=tokenH,
    )

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute IoU for each class and return mean IoU."""
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

    return np.nanmean(iou_per_class), iou_per_class

def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Dice coefficient (F1 Score) per class and return mean Dice Score."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)

        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class), dice_per_class

def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    """Save a side-by-side comparison of input, ground truth, and prediction."""

    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(gt_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_metrics_summary(results, output_dir):
    """Save metrics summary to a text file and create bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Class IoU:\n")
        f.write("-" * 40 + "\n")
        for i, (name, iou) in enumerate(zip(class_names, results['class_iou'])):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {iou_str}\n")

    print(f"\nSaved evaluation metrics to {filepath}")

    fig, ax = plt.subplots(figsize=(10, 6))

    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    ax.bar(range(n_classes), valid_iou, color=[color_palette[i] / 255 for i in range(n_classes)],
           edgecolor='black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU (Mean: {results["mean_iou"]:.4f})')
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_iou'], color='red', linestyle='--', label='Mean')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved per-class metrics chart to '{output_dir}/per_class_metrics.png'")

def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_file = os.path.abspath(__file__)

    parser = argparse.ArgumentParser(description='Segmentation prediction/inference script')
    parser.add_argument('--model_path', type=str, default=os.path.join(script_dir, 'segmentation_head.pth'),
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, default=resolve_offroad_test_images_dir(script_file),
                        help='Path to validation dataset')
    parser.add_argument('--output_dir', type=str, default=os.path.join(script_dir, 'predictions'),
                        help='Directory to save prediction visualizations')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for validation')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of comparison visualizations to save (predictions saved for ALL images)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    print(f"Loading dataset from {args.data_dir}...")
    valset = MaskDataset(data_dir=args.data_dir, transform=transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(valset)} samples")

    print("Loading DINOv2 backbone...")
    BACKBONE_SIZE = "small"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)
    print("Backbone loaded successfully!")

    sample_img, _, _ = valset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(sample_img)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")

    print(f"Loading model from {args.model_path}...")
    loaded = torch.load(args.model_path, map_location="cpu")
    state_dict = _unwrap_state_dict(loaded)
    classifier = build_classifier_for_checkpoint(
        state_dict,
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14,
    )
    classifier.load_state_dict(state_dict, strict=True)
    classifier = classifier.to(device)
    classifier.eval()
    print("Model loaded successfully!")

    masks_dir = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    print(f"\nRunning evaluation and saving predictions for all {len(valset)} images...")

    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    all_class_iou = []
    all_class_dice = []
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Processing", unit="batch")
        for batch_idx, (imgs, labels, data_ids) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)

            output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(output.to(device))
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels_squeezed = labels.squeeze(dim=1).long()
            predicted_masks = torch.argmax(outputs, dim=1)

            iou, class_iou = compute_iou(outputs, labels_squeezed, num_classes=n_classes)
            dice, class_dice = compute_dice(outputs, labels_squeezed, num_classes=n_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels_squeezed)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            all_class_iou.append(class_iou)
            all_class_dice.append(class_dice)

            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)
                pred_img = Image.fromarray(pred_mask)
                pred_img.save(os.path.join(masks_dir, f'{base_name}_pred.png'))

                pred_color = mask_to_color(pred_mask)
                cv2.imwrite(os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                            cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                if sample_count < args.num_samples:
                    save_prediction_comparison(
                        imgs[i], labels_squeezed[i], predicted_masks[i],
                        os.path.join(comparisons_dir, f'sample_{sample_count}_comparison.png'),
                        data_id
                    )

                sample_count += 1

            pbar.set_postfix(iou=f"{iou:.3f}")

    mean_iou = np.nanmean(iou_scores)
    mean_dice = np.nanmean(dice_scores)
    mean_pixel_acc = np.mean(pixel_accuracies)

    avg_class_iou = np.nanmean(all_class_iou, axis=0)
    avg_class_dice = np.nanmean(all_class_dice, axis=0)

    results = {
        'mean_iou': mean_iou,
        'class_iou': avg_class_iou
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean IoU:          {mean_iou:.4f}")
    print("=" * 50)

    save_metrics_summary(results, args.output_dir)

    print(f"\nPrediction complete! Processed {len(valset)} images.")
    print(f"\nOutputs saved to {args.output_dir}/")
    print(f"  - masks/           : Raw prediction masks (class IDs 0-9)")
    print(f"  - masks_color/     : Colored prediction masks (RGB)")
    print(f"  - comparisons/     : Side-by-side comparison images ({args.num_samples} samples)")
    print(f"  - evaluation_metrics.txt")
    print(f"  - per_class_metrics.png")

if __name__ == "__main__":
    main()

