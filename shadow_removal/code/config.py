from pathlib import Path
import os
import torch

# Directories
BASE_DIR = Path(__file__).parent.parent
CODE_DIR = BASE_DIR / "code"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "input"

# Training Parameters
DEVICE = torch.device("cuda:0")
N_EPOCHS = 100
BATCH_SIZE = 4
N_WORKERS = os.cpu_count()
N_TIMES_PER_BATCH = 2
