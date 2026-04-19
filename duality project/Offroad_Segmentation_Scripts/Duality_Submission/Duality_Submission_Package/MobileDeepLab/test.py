import os
import time
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import DesertDataset, get_test_dataloader
from model import MobileDeepLab
from utils import save_colored_mask, compute_iou, count_parameters

CLASS_NAMES = {
    0: "Trees",
    1: "Lush Bushes",
    2: "Dry Grass",
    3: "Dry Bushes",
    4: "Ground Clutter",
    5: "Flowers",
    6: "Logs",
    7: "Rocks",
    8: "Landscape",
    9: "Sky"
}

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = os.path.join(RUNS_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Train the model first.")

    model = MobileDeepLab(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    output_dir = os.path.join(RUNS_DIR, 'predictions')
    os.makedirs(output_dir, exist_ok=True)

    total_params, trainable_params = count_parameters(model)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print("\n--- Model Size ---")
    print(f"Total Parameters: {total_params:,}")
    print(f"File Size: {file_size_mb:.2f} MB")

    probe_dataset = DesertDataset(root_dir=TEST_DIR, is_train_or_val=True)
    test_has_truth = len(probe_dataset.mask_files) > 0 and len(probe_dataset.mask_files) == len(probe_dataset.rgb_files)

    if test_has_truth:
        print("\n[Info] Found ground truth masks. Will compute IoU.")
        test_dataset = probe_dataset
    else:
        print("\n[Info] No ground truth masks found. Will only run inference and save predictions.")
        test_dataset = DesertDataset(root_dir=TEST_DIR, is_train_or_val=False)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    inference_times = []
    class_ious = {i: [] for i in range(NUM_CLASSES)}

    print("\n--- Running Inference ---")
    with torch.no_grad():
        with autocast(device_type='cuda', enabled=USE_AMP):

            dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
            for _ in range(10):
                _ = model(dummy_input)
            torch.cuda.synchronize()

            loop = tqdm(test_loader, desc="Testing")
            for batch in loop:
                if test_has_truth:
                    images, masks, rgb_paths = batch
                    masks = masks.to(device)
                else:
                    images, rgb_paths = batch

                images = images.to(device)

                torch.cuda.synchronize()
                start_time = time.time()

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                torch.cuda.synchronize()
                end_time = time.time()

                inference_times.append((end_time - start_time) * 1000)

                if test_has_truth:
                    iou_list, _ = compute_iou(preds, masks, NUM_CLASSES)
                    for c_idx, iou_val in enumerate(iou_list):
                        if not torch.isnan(torch.tensor(iou_val)):
                            class_ious[c_idx].append(iou_val)

                for i in range(images.size(0)):
                    pred_mask = preds[i]
                    rgb_path = rgb_paths[i]
                    basename = os.path.basename(rgb_path)
                    filename, _ = os.path.splitext(basename)

                    out_path = os.path.join(output_dir, f"{filename}_pred.png")
                    save_colored_mask(pred_mask, out_path)

    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    print(f"\n--- Benchmark Results ---")
    print(f"Average Inference Time: {avg_inference_time:.2f} ms")
    if avg_inference_time < TARGET_INFERENCE_MS:
        print(f"Target <{TARGET_INFERENCE_MS}ms Check: PASS")
    else:
        print(f"Target <{TARGET_INFERENCE_MS}ms Check: FAIL")

    if test_has_truth:
        print("\n--- Per-Class IoU Table ---")
        print(f"{'Class ID':<10} | {'Class Name':<15} | {'IoU':<10}")
        print("-" * 40)
        valid_ious = []
        for c_idx in range(NUM_CLASSES):
            valid_vals = class_ious[c_idx]
            if len(valid_vals) > 0:
                c_iou = sum(valid_vals) / len(valid_vals)
                valid_ious.append(c_iou)
                print(f"{c_idx:<10} | {CLASS_NAMES[c_idx]:<15} | {c_iou:.4f}")
            else:
                print(f"{c_idx:<10} | {CLASS_NAMES[c_idx]:<15} | N/A")

        mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
        print("-" * 40)
        print(f"Mean IoU: {mean_iou:.4f}")

    print(f"\nColorized predictions saved to {output_dir}")

if __name__ == '__main__':
    test()
