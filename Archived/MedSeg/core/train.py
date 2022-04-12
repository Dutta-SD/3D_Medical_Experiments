# Baseline UNet model
import pytorch_lightning as pl
from dataloaders import SpleenImageDataLoader
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
)
from monai.data import list_data_collate, decollate_batch
import torch
from monai.inferers import sliding_window_inference
from model_defn import unet_baseline
import config

from model_defn import fpunet
from pl.callbacks import ModelCheckpoint


class SegmentationModel3D(pl.LightningModule):
    """
    Baseline Unet model
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._model = model
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose(
            [EnsureType(), AsDiscrete(argmax=True, to_onehot=True, num_classes=2)]
        )
        self.post_label = Compose(
            [EnsureType(), AsDiscrete(to_onehot=True, num_classes=2)]
        )
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self._imp_transforms = Compose([])

    def forward(self, x: torch.Tensor):
        return self._model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 3e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)

        tensorboard_logs = {"train_loss": loss.item()}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (160, 160, 160)
        sw_batch_size = 2

        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        return {"val_loss": loss, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        return {"log": tensorboard_logs}


# Train The model
if __name__ == "__main__":
    net = SegmentationModel3D(fpunet)

    # set up loggers and checkpoints
    log_dir = config.LOG_DIR
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir)

    # dataloader
    dl = SpleenImageDataLoader(
        data_dir=config.DATA_DIR,
        num_val_samples=10,
    )

    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=config.NUM_TRAIN_EPOCHS,
        logger=tb_logger,
        checkpoint_callback=True,
        num_sanity_val_steps=1,
        precision=16,
        gradient_clip_val=0.2,
        default_root_dir=config.MODEL_DIR,
    )

    # train
    trainer.fit(
        net,
        datamodule=dl,
    )
