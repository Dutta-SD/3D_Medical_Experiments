from components import MedSegModel
from train.Trainer import Trainer
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from dataset.trainds import TRAIN_DATALOADER
from dataset.valds import VAL_DATALOADER
from monai.losses.dice import DiceLoss
import config
import json
from torch import save
from torch.cuda.amp import GradScaler
import torch
# ---------------------
torch.backends.cudnn.benchmark = True
# ---------------------
device = config.DEVICE
model = MedSegModel().to(device)
optimizer = AdamW(model.parameters())
scheduler = OneCycleLR(optimizer, 0.01, total_steps=config.NUM_TRAIN_EPOCHS)
criterion = DiceLoss(
    smooth_nr=0,
    smooth_dr=1e-5,
    squared_pred=True,
    to_onehot_y=False,
    sigmoid=True,
).to(device)
scaler = GradScaler()
# -----------------------------
metrics = {}
trainer = Trainer(
    model=model,
    device=device,
    criterion=criterion,
    optimizer=optimizer,
    training_DataLoader=TRAIN_DATALOADER,
    validation_DataLoader=VAL_DATALOADER,
    lr_scheduler=scheduler,
    scaler=scaler,
    epochs=config.NUM_TRAIN_EPOCHS,
    notebook=False,
)
# -----------------------------

train_loss, val_loss, learning_rates = trainer.run_trainer()

# Save Model Weights
save(model.state_dict(), config.MODEL_DIR / "multimodal-model-weights-2.pth")

# Save Metric Dicts
metrics["training_loss"] = train_loss
metrics["validation_loss"] = val_loss
metrics["learning_rates"] = learning_rates

with open(config.OUTPUT_DIR / "metric-2.json", "tw+") as f:
    json.dump(metrics, f)
