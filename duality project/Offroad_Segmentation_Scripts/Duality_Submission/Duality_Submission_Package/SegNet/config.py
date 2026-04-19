import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Up to `duality project/` (parent of Offroad_Segmentation_Scripts/)
_DATA_ROOT = os.path.normpath(os.path.join(BASE_DIR, '..', '..', '..', '..'))

TRAIN_DIR = os.path.join(_DATA_ROOT, 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset', 'train')
VAL_DIR = os.path.join(_DATA_ROOT, 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset', 'val')
TEST_DIR = os.path.join(_DATA_ROOT, 'Offroad_Segmentation_testImages', 'Offroad_Segmentation_testImages')
RUNS_DIR = os.path.join(BASE_DIR, 'runs')

NUM_CLASSES = 10
IMAGE_SIZE = 256
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 15

RAW_CLASSES = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
