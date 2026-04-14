"""
Visualize Script — Project 5 (Custom CNN)
Colorizes raw segmentation mask images for visual inspection.
Same logic as projects 1-4.

Usage:
  python visualize5.py
"""

import cv2
import numpy as np
import os
from pathlib import Path

# Input folder containing raw segmentation mask images
input_folder = r"d:\kfiles\aarvak files\Offroad_Autonomy_Segmentation_Model\duality project\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Segmentation"

# Output folder for colorized images
output_folder = os.path.join(input_folder, "colorized_custom_cnn")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all image files in the folder
image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
image_files = [f for f in Path(input_folder).iterdir()
               if f.is_file() and f.suffix.lower() in image_extensions]

print(f"Found {len(image_files)} image files to process")

# Dictionary to store color mappings (pixel value -> colour)
color_map = {}

# Process each file
for image_file in sorted(image_files):
    print(f"Processing: {image_file.name}")

    im = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)

    if im is None:
        print(f"  Skipped: Could not read {image_file.name}")
        continue

    u    = np.unique(im)
    im2  = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)

    for v in u:
        if v not in color_map:
            np.random.seed(int(v) % (2 ** 31))
            color_map[v] = np.random.randint(0, 255, (3,), dtype=np.uint8)
        im2[im == v] = color_map[v]

    output_path = os.path.join(output_folder, f"{image_file.stem}.png")
    cv2.imwrite(output_path, im2)
    print(f"  Saved: {output_path}")

print(f"\nProcessing complete! Colorized images saved to: {output_folder}")
print(f"Total unique values found: {len(color_map)}")
