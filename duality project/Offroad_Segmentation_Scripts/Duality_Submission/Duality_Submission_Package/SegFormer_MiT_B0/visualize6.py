import cv2
import numpy as np
import os
from pathlib import Path

script_dir   = Path(__file__).parent.resolve()
input_folder = script_dir / 'predictions6' / 'masks'
output_folder = input_folder / 'colorized'

color_palette = np.array([
    [0,   0,   0  ],
    [34,  139, 34 ],
    [0,   255, 0  ],
    [210, 180, 140],
    [139, 90,  43 ],
    [128, 128, 0  ],
    [139, 69,  19 ],
    [128, 128, 128],
    [160, 82,  45 ],
    [135, 206, 235],
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
        and 'colorized' not in str(f)
    ])

    if not image_files:
        print(f'[ERROR] No mask images found in {input_folder}')
        return

    print(f'Found {len(image_files)} mask file(s) to colourise.')
    print(f'Output → {output_folder}\n')

    processed, skipped = 0, 0
    for mask_file in image_files:
        print(f'  Processing: {mask_file.name}')

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

    print('\nColour legend:')
    print('-' * 40)
    for i, name in enumerate(class_names):
        r, g, b = color_palette[i]
        print(f'  {i:2d}  {name:<16s}  RGB({r:3d}, {g:3d}, {b:3d})')

if __name__ == '__main__':
    main()
