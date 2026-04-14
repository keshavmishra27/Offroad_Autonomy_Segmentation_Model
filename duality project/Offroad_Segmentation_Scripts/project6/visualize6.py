"""
Visualisation Script — Project 6 (SegFormer MiT-B0)

Reads raw class-ID prediction masks (saved by test_segmentation6.py) from the
predictions6/masks/ folder and converts every one to a colourised RGB image,
saving results to predictions6/masks/colorized/.

This mirrors the visualize1.py behaviour but uses the project-wide colour
palette so the colours are consistent across all projects.

Usage:
  python visualize6.py

No arguments needed — paths are resolved relative to this script.
"""

import cv2
import numpy as np
import os
from pathlib import Path

# ── Paths (relative to this script) ──────────────────────────────────────────
script_dir   = Path(__file__).parent.resolve()
input_folder = script_dir / 'predictions6' / 'masks'
output_folder = input_folder / 'colorized'

# ── Semantic colour palette (same as all other projects) ─────────────────────
color_palette = np.array([
    [0,   0,   0  ],   # 0 Background – black
    [34,  139, 34 ],   # 1 Trees – forest green
    [0,   255, 0  ],   # 2 Lush Bushes – lime
    [210, 180, 140],   # 3 Dry Grass – tan
    [139, 90,  43 ],   # 4 Dry Bushes – brown
    [128, 128, 0  ],   # 5 Ground Clutter – olive
    [139, 69,  19 ],   # 6 Logs – saddle brown
    [128, 128, 128],   # 7 Rocks – gray
    [160, 82,  45 ],   # 8 Landscape – sienna
    [135, 206, 235],   # 9 Sky – sky blue
], dtype=np.uint8)

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]
n_classes = len(class_names)


def mask_to_color(mask_np: np.ndarray) -> np.ndarray:
    """Convert a class-ID mask (H×W uint8) to a BGR image for cv2."""
    h, w  = mask_np.shape
    rgb   = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_classes):
        rgb[mask_np == c] = color_palette[c]
    # cv2 uses BGR
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
    if not input_folder.exists():
        print(f'[ERROR] Input folder not found: {input_folder}')
        print('        Run test_segmentation6.py first to generate prediction masks.')
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
    image_files = sorted([
        f for f in input_folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
        and 'colorized' not in str(f)  # skip any already-colourised files
    ])

    if not image_files:
        print(f'[ERROR] No mask images found in {input_folder}')
        return

    print(f'Found {len(image_files)} mask file(s) to colourise.')
    print(f'Output → {output_folder}\n')

    processed, skipped = 0, 0
    for mask_file in image_files:
        print(f'  Processing: {mask_file.name}')

        # Read as grayscale (class IDs 0-9)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f'    [SKIP] Could not read {mask_file.name}')
            skipped += 1
            continue

        unique_ids = np.unique(mask)
        print(f'    Classes present: {[class_names[i] for i in unique_ids if i < n_classes]}')

        color_img = mask_to_color(mask)
        out_path  = output_folder / f'{mask_file.stem}_colorized.png'
        cv2.imwrite(str(out_path), color_img)
        print(f'    Saved: {out_path.name}')
        processed += 1

    print(f'\nDone! {processed} image(s) colourised, {skipped} skipped.')
    print(f'Colourised masks saved to: {output_folder}')

    # Print palette legend
    print('\nColour legend:')
    print('-' * 40)
    for i, name in enumerate(class_names):
        r, g, b = color_palette[i]
        print(f'  {i:2d}  {name:<16s}  RGB({r:3d}, {g:3d}, {b:3d})')


if __name__ == '__main__':
    main()
