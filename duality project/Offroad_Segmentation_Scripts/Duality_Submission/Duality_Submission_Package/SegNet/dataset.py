import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import RAW_CLASSES, IMAGE_SIZE

_MAPPING = torch.zeros(10001, dtype=torch.long)
for new_idx, raw_val in enumerate(RAW_CLASSES):
    _MAPPING[raw_val] = new_idx

def remap_classes(mask_tensor):
    """
    Vectorized class remapping without python loops
    mask: tensor with actual pixel values e.g., 100, 10000
    """
    mask_tensor = torch.clamp(mask_tensor, max=10000)
    return _MAPPING[mask_tensor.long()]

class DesertDataset(Dataset):
    def __init__(self, root_dir, is_train_or_val=True):
        """
        root_dir: directory like Train/ or Val/ containing images and potentially masks.
        is_train_or_val: True if loading both images and masks, False for test set (images only).
        """
        self.root_dir = root_dir
        self.is_train_or_val = is_train_or_val

        self.rgb_files = []
        self.mask_files = []

        for dirpath, _, filenames in os.walk(self.root_dir):
            for file in filenames:
                file_lower = file.lower()
                if file_lower.endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(dirpath, file)
                    parent_dir_name = os.path.basename(dirpath).lower()

                    if any(x in parent_dir_name for x in ['mask', 'segmentation']) or any(x in file_lower for x in ['mask', 'segmentation']):
                        self.mask_files.append(filepath)
                    else:
                        if 'image' in parent_dir_name or 'rgb' in parent_dir_name or not any(x in parent_dir_name for x in ['mask', 'label', 'segmentation']):
                            self.rgb_files.append(filepath)

        self.rgb_files.sort()

        if self.is_train_or_val:
            self.mask_dict = {os.path.basename(f): f for f in self.mask_files}

        self.image_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST)
        ])

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        image = Image.open(rgb_path).convert('RGB')

        image = self.image_transform(image)

        if self.is_train_or_val:
            basename = os.path.basename(rgb_path)
            if basename in self.mask_dict:
                mask_path = self.mask_dict[basename]
            else:
                raise FileNotFoundError(f"Mask for {basename} not found.")

            mask = Image.open(mask_path)
            mask_np = np.array(mask).astype(np.int32)
            mask_pil_safe = Image.fromarray(mask_np)
            mask_resized = self.mask_transform(mask_pil_safe)
            mask_tensor = torch.from_numpy(np.array(mask_resized)).long()

            mapped_mask = remap_classes(mask_tensor)

            return image, mapped_mask, rgb_path
        else:
            return image, rgb_path
