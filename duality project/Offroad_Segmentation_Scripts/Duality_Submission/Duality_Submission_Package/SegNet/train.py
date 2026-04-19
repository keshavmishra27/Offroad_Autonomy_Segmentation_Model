import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from torch.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

from config import TRAIN_DIR, VAL_DIR, RUNS_DIR, BATCH_SIZE, LR, EPOCHS, NUM_CLASSES
from dataset import DesertDataset
from model import SegNet
from utils import compute_iou

def train():
    os.makedirs(RUNS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    cudnn.benchmark = True

    print("Loading datasets...")
    train_dataset = DesertDataset(root_dir=TRAIN_DIR, is_train_or_val=True)
    val_dataset = DesertDataset(root_dir=VAL_DIR, is_train_or_val=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Found {len(train_dataset)} training samples")
    print(f"Found {len(val_dataset)} validation samples")

    model = SegNet(in_channels=3, out_channels=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')

    best_val_iou = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):

        model.train()
        train_loss = 0.0
        train_ious = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for images, masks, _ in loop:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            _, batch_mean_iou = compute_iou(preds, masks)
            train_ious.append(batch_mean_iou)

            loop.set_postfix(loss=loss.item())

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_iou = sum(train_ious) / len(train_ious)

        model.eval()
        val_loss = 0.0
        val_ious = []

        with torch.no_grad():
            loop_val = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]")
            for images, masks, _ in loop_val:
                images = images.to(device)
                masks = masks.to(device)

                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                _, batch_mean_iou = compute_iou(preds, masks)
                val_ious.append(batch_mean_iou)

                loop_val.set_postfix(loss=loss.item())

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_iou = sum(val_ious) / len(val_ious)

        print(f"Epoch {epoch} Summary: Train Loss: {epoch_train_loss:.4f} | Train IoU: {epoch_train_iou:.4f} | Val Loss: {epoch_val_loss:.4f} | Val IoU: {epoch_val_iou:.4f}")

        history.append({
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'train_mean_iou': epoch_train_iou,
            'val_loss': epoch_val_loss,
            'val_mean_iou': epoch_val_iou
        })

        if epoch_val_iou > best_val_iou:
            best_val_iou = epoch_val_iou
            torch.save(model.state_dict(), os.path.join(RUNS_DIR, 'best_model.pth'))
            print("[Info] Best model updated and saved based on Validation mean IoU.")

        df = pd.DataFrame(history)
        df.to_csv(os.path.join(RUNS_DIR, 'log.csv'), index=False)

    print("Training Complete!")

if __name__ == '__main__':
    train()
