import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "..", ".."))

TRAIN_DIR = os.path.join(_DATA_ROOT, "Offroad_Segmentation_Training_Dataset", "Offroad_Segmentation_Training_Dataset", "train")
VAL_DIR = os.path.join(_DATA_ROOT, "Offroad_Segmentation_Training_Dataset", "Offroad_Segmentation_Training_Dataset", "val")
TEST_DIR = os.path.join(_DATA_ROOT, "Offroad_Segmentation_testImages", "Offroad_Segmentation_testImages")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

NUM_CLASSES = 10
IMAGE_SIZE = 512
BATCH_SIZE = 16

PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 35

PHASE1_LR = 1e-3
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4

USE_AMP = True
TARGET_INFERENCE_MS = 50
BENCHMARK_RESOLUTIONS = [256, 384, 512, 640]
NUM_WORKERS = 4

RAW_CLASSES = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
