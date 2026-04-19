import os
import time
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TEST_DIR, RUNS_DIR, NUM_CLASSES, MODEL_VARIANT
from dataset import DesertDataset
from model import FCN32s, FCN16s, FCN8s
from utils import save_colored_mask, compute_iou

def test():
    parser = argparse.ArgumentParser(description='Test FCN variants')
    parser.add_argument('--model', type=str, default=MODEL_VARIANT, choices=['fcn8s', 'fcn16s', 'fcn32s'], help='Model variant to test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = os.path.join(RUNS_DIR, f'best_{args.model}.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model checkpoint at {model_path}. Train the {args.model} model first.")

    if args.model == 'fcn32s':
        model = FCN32s(num_classes=NUM_CLASSES)
    elif args.model == 'fcn16s':
        model = FCN16s(num_classes=NUM_CLASSES)
    else:
        model = FCN8s(num_classes=NUM_CLASSES)

    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    output_dir = os.path.join(RUNS_DIR, 'predictions', args.model)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading test images...")
    test_has_truth = False

    probe_dataset = DesertDataset(root_dir=TEST_DIR, is_train_or_val=True)
    if len(probe_dataset.mask_files) > 0 and len(probe_dataset.mask_files) == len(probe_dataset.rgb_files):
        test_has_truth = True
        print("[Info] Found ground truth masks for test set. Will compute IoU.")
        test_dataset = probe_dataset
    else:
        print("[Info] No ground truth masks found for test set. Will only save predictions.")
        test_dataset = DesertDataset(root_dir=TEST_DIR, is_train_or_val=False)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    inference_times = []
    test_ious = []

    with torch.no_grad():
        loop = tqdm(test_loader, desc=f"Testing [{args.model}]")
        for batch in loop:
            if test_has_truth:
                images, masks, rgb_paths = batch
                masks = masks.to(device)
            else:
                images, rgb_paths = batch

            images = images.to(device)

            start_time = time.time()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            end_time = time.time()

            inference_times.append(end_time - start_time)

            if test_has_truth:
                _, batch_mean_iou = compute_iou(preds, masks)
                test_ious.append(batch_mean_iou)

            for i in range(images.size(0)):
                pred_mask = preds[i].cpu().numpy()
                rgb_path = rgb_paths[i]
                basename = os.path.basename(rgb_path)
                filename, _ = os.path.splitext(basename)

                out_path = os.path.join(output_dir, f"{filename}_pred.png")
                save_colored_mask(pred_mask, out_path)

    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    print(f"\n{args.model} Test Run Complete.")
    print(f"Average Inference Time per Image: {avg_inference_time:.4f} seconds")

    if test_has_truth:
        mean_iou = sum(test_ious) / len(test_ious)
        print(f"Test Mean IoU: {mean_iou:.4f}")

    print(f"Colorized predictions saved to {output_dir}")

if __name__ == '__main__':
    test()
