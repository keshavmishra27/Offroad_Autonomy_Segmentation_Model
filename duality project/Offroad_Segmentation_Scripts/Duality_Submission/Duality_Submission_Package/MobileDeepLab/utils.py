import torch
import numpy as np
from PIL import Image
from config import NUM_CLASSES
import time

def compute_iou(pred, target, num_classes=NUM_CLASSES):
    """
    Compute per-class IoU and mean IoU.
    pred: Tensor of shape (B, H, W) containing integer class predictions
    target: Tensor of shape (B, H, W) containing integer class targets
    """
    pred = pred.view(-1)
    target = target.view(-1)

    mask = (target >= 0) & (target < num_classes)
    pred_v = pred[mask]
    target_v = target[mask]

    intersect = pred_v[pred_v == target_v]

    area_intersect = torch.bincount(intersect, minlength=num_classes)
    area_pred = torch.bincount(pred_v, minlength=num_classes)
    area_target = torch.bincount(target_v, minlength=num_classes)

    area_union = area_pred + area_target - area_intersect

    iou = area_intersect.float() / torch.clamp(area_union, min=1).float()

    valid = area_union > 0
    mean_iou = iou[valid].mean().item() if valid.any() else 0.0

    iou_tensor = torch.where(valid, iou, torch.tensor(float('nan'), device=iou.device))

    return iou_tensor.tolist(), mean_iou

def get_color_map():
    """
    Return a predefined color map for 10 classes to colorize segmentation masks.
    """
    color_map = {
        0: (34, 139, 34),
        1: (154, 205, 50),
        2: (218, 165, 32),
        3: (139, 69, 19),
        4: (160, 82, 45),
        5: (255, 105, 180),
        6: (139, 115, 85),
        7: (128, 128, 128),
        8: (210, 180, 140),
        9: (135, 206, 235),
    }
    return color_map

def save_colored_mask(mask, path):
    """
    Saves a given integer mask as a colorized PNG.
    mask: Numpy array or PyTorch tensor of shape (H, W) containing 0-9 indices
    path: Path to save the PNG image
    """
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()

    color_map = get_color_map()
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, color in color_map.items():
        colored_mask[mask == idx] = color

    img = Image.fromarray(colored_mask)
    img.save(path)

def count_parameters(model):
    """Returns total and trainable parameter count"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def measure_inference_time(model, input_size, device, n_runs=100):
    """
    Measures inference time carefully.
    Warms up GPU with 10 runs first, then averages over n_runs.
    """
    model.eval()
    dummy_input = torch.randn(*input_size, device=device)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for _ in range(10):
                _ = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for _ in range(n_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()

                _ = model(dummy_input)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                timings.append((end_time - start_time) * 1000)

    return np.mean(timings), np.std(timings), np.min(timings), np.max(timings)

def get_gpu_memory_usage():
    """Returns current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0
