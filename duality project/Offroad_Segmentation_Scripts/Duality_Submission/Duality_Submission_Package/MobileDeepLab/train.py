import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import *
from dataset import get_dataloader
from model import MobileDeepLab
from utils import compute_iou, get_gpu_memory_usage

def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def train():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training MobileDeepLab but not available.")

    device = torch.device('cuda')

    os.makedirs(RUNS_DIR, exist_ok=True)
    log_path = os.path.join(RUNS_DIR, 'log.csv')
    best_model_path = os.path.join(RUNS_DIR, 'best_model.pth')

    train_loader = get_dataloader(TRAIN_DIR, is_train=True, batch_size=BATCH_SIZE)
    val_loader = get_dataloader(VAL_DIR, is_train=False, batch_size=BATCH_SIZE)

    model = MobileDeepLab(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=10000)

    scaler = GradScaler(enabled=USE_AMP)

    best_iou = 0.0

    log_data = []

    total_epochs = PHASE1_EPOCHS + PHASE2_EPOCHS

    for epoch in range(1, total_epochs + 1):
        phase = 1 if epoch <= PHASE1_EPOCHS else 2

        if epoch == 1:
            print("--- Starting PHASE 1: Freezing backbone ---")
            set_requires_grad(model.low_level_features, False)
            set_requires_grad(model.high_level_features, False)

            optimizer = optim.Adam([
                {'params': model.aspp.parameters()},
                {'params': model.low_level_conv.parameters()},
                {'params': model.decoder_conv.parameters()},
                {'params': model.classifier.parameters()}
            ], lr=PHASE1_LR)

            scheduler = CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)

        elif epoch == PHASE1_EPOCHS + 1:
            print("--- Starting PHASE 2: Unfreezing backbone ---")
            set_requires_grad(model.low_level_features, True)
            set_requires_grad(model.high_level_features, True)

            optimizer = optim.Adam([
                {'params': model.low_level_features.parameters(), 'lr': BACKBONE_LR},
                {'params': model.high_level_features.parameters(), 'lr': BACKBONE_LR},
                {'params': model.aspp.parameters(), 'lr': HEAD_LR},
                {'params': model.low_level_conv.parameters(), 'lr': HEAD_LR},
                {'params': model.decoder_conv.parameters(), 'lr': HEAD_LR},
                {'params': model.classifier.parameters(), 'lr': HEAD_LR}
            ])
            scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

        model.train()
        train_loss = 0.0
        start_time = time.time()

        for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]"):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda', enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        end_time = time.time()
        batch_time_ms = (end_time - start_time) / len(train_loader) * 1000

        model.eval()
        val_loss = 0.0
        all_mean_ious = []

        with torch.no_grad():
            for images, targets, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{total_epochs} [Val]"):
                images, targets = images.to(device), targets.to(device)

                with autocast(device_type='cuda', enabled=USE_AMP):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                _, mean_iou = compute_iou(preds, targets, NUM_CLASSES)
                all_mean_ious.append(mean_iou)

        val_loss /= len(val_loader.dataset)
        epoch_mean_iou = sum(all_mean_ious) / len(all_mean_ious) if all_mean_ious else 0.0

        current_lr = optimizer.param_groups[-1]['lr']
        gpu_mem = get_gpu_memory_usage()

        print(f"Epoch {epoch} | Phase {phase} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {epoch_mean_iou:.4f} | LR: {current_lr:.6f} | Batch Time: {batch_time_ms:.1f}ms | GPU Mem: {gpu_mem:.1f}MB")

        log_data.append({
            'epoch': epoch,
            'phase': phase,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'mean_iou': epoch_mean_iou,
            'lr': current_lr,
            'batch_time_ms': batch_time_ms,
            'gpu_memory_mb': gpu_mem
        })

        pd.DataFrame(log_data).to_csv(log_path, index=False)

        if epoch_mean_iou > best_iou:
            best_iou = epoch_mean_iou
            torch.save(model.state_dict(), best_model_path)
            print("--> Saved new best model")

        scheduler.step()

    print("Training Complete!")

if __name__ == '__main__':
    train()
