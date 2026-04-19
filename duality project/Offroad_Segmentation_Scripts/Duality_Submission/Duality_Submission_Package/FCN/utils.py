import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from config import NUM_CLASSES

def compute_iou(pred, target, num_classes=NUM_CLASSES):
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
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()

    color_map = get_color_map()
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, color in color_map.items():
        colored_mask[mask == idx] = color

    img = Image.fromarray(colored_mask)
    img.save(path)

def plot_fcn_variants(log_dir):
    """
    Plots the validation IoU of FCN-8s, 16s, and 32s if their logs exist.
    """
    variants = {
        'FCN-8s': 'log_fcn8s.csv',
        'FCN-16s': 'log_fcn16s.csv',
        'FCN-32s': 'log_fcn32s.csv'
    }

    plt.figure(figsize=(10, 6))
    plotted_any = False

    for name, filename in variants.items():
        file_path = os.path.join(log_dir, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            plt.plot(df['epoch'], df['val_mean_iou'], label=name)
            plotted_any = True

    if plotted_any:
        plt.title('Validation Mean IoU across FCN Variants')
        plt.xlabel('Epoch')
        plt.ylabel('Val Mean IoU')
        plt.legend()
        plt.grid(True)
        out_path = os.path.join(log_dir, 'fcn_variants_comparison.png')
        plt.savefig(out_path)
        print(f"[Info] Saved FCN variants comparison to {out_path}")
    else:
        print("[Info] No FCN logs found to plot variants.")
