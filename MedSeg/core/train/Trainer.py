import numpy as np
import torch
import config


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        training_DataLoader: torch.utils.data.Dataset,
        validation_DataLoader: torch.utils.data.Dataset = None,
        lr_scheduler: torch.optim.lr_scheduler = None,
        scaler: torch.cuda.amp.GradScaler = None,
        epochs: int = 100,
        epoch: int = 0,
        notebook: bool = False,
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.scaler = scaler

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc="Progress")
        for i in progressbar:
            self.epoch += 1  # epoch counter

            # Train
            self._train()

            # Validate
            if self.validation_DataLoader is not None:
                self._validate()

            # LR Scheduling
            if self.lr_scheduler is not None:
                if (
                    self.validation_DataLoader is not None
                    and self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau"
                ):
                    self.lr_scheduler.step(
                        self.validation_loss[i]
                    )  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.step()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(
            enumerate(self.training_DataLoader),
            "Training",
            total=len(self.training_DataLoader),
            leave=False,
        )

        for i, batch in batch_iter:
            x1, x2, y = self.get_image_and_labels_from_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)  # zerograd the parameters
            # Enable fp16
            with torch.cuda.amp.autocast():
                out = self.model(x1, x2)  # one forward pass
                loss = self.criterion(out, y)  # calculate loss
                loss_value = loss.item()
                train_losses.append(loss_value)
            self.scaler.scale(loss).backward()  # one backward pass
            self.scaler.step(self.optimizer)  # update the parameters
            self.scaler.update()

            batch_iter.set_description(
                f"Training: (loss {loss_value:.4f})"
            )  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]["lr"])

        batch_iter.close()

    @torch.no_grad()
    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(
            enumerate(self.validation_DataLoader),
            "Validation",
            total=len(self.validation_DataLoader),
            leave=False,
        )

        for i, batch in batch_iter:
            x1, x2, y = self.get_image_and_labels_from_batch(batch)

            with torch.cuda.amp.autocast():
                out = self.model(x1, x2)
                loss = self.criterion(out, y)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f"Validation: (loss {loss_value:.4f})")

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()

    def get_image_and_labels_from_batch(self, batch: dict):
        # TODO: Might be a bottleneck
        # Data -- Oth and 1st modality
        x1 = batch["image"][:, 0, :, :, :].to(config.DEVICE).unsqueeze(1)
        x2 = batch["image"][:, 1, :, :, :].to(config.DEVICE).unsqueeze(1)

        # Log
        # LOG.info(f"Shape of Inputs: {x1.shape} & {x2.shape}")

        # Labels -- 0th prediction
        y_true = batch["label"][:, 0, :, :, :].to(config.DEVICE).unsqueeze(1)

        # Log
        # LOG.info(f"Shape of Output is: {y_true.shape}")
        return x1, x2, y_true
