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
RNG_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_TRAIN_EPOCHS = 100
NUM_WORKERS = os.cpu_count()
