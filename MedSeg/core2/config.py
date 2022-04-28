from pathlib import Path
import os
import torch

# BASE DIR
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "input"
MODEL_DIR = BASE_DIR / "model"
LOG_DIR = BASE_DIR / "logs"

# Data URLs and md5
SPLEEN_DATA_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
SPLEEN_DATA_MD5 = "410d4a301da4e5b2f6f86ec3ddba524e"

# Data utilities
RNG_SEED = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 2
NUM_TRAIN_EPOCHS = 580
NUM_WORKERS = 2
LR = 1e-4

# Monai Directory
os.environ["MONAI_DATA_DIRECTORY"] = str(DATA_DIR)

# Model Utilities
EMB_SIZE = 720
PATCH_SIZE = 32
IMG_SIZE = 240
PROJ_OP_CHANNEL = 3
N_SLICES = 155
