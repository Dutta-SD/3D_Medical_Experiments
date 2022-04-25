# Training Loop
import torch
import torch.optim as opt
import components as cps
import config
import utils as uts
from datasets import (
    # testDataset,
    trainDataset,
)
import os

# Logging
import logging

logging.basicConfig(
    level=logging.DEBUG,
    filename=config.LOG_DIR / "train_log.log",
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger('PIL').setLevel(logging.WARNING)

# define the model components
actor_model = cps.ActorUNet(n_input_channels=4).to(config.DEVICE)
critic_model = cps.Critic(extra_channel=1).to(config.DEVICE)
shdw_gen_model = cps.PseudoShadowGenerator()
logging.debug(f"Configured Models")

# optimizers
optimzr = opt.Adam(
    list(actor_model.parameters()) + list(critic_model.parameters()),
)
logging.debug(f"Configurig Optimizers")
# crit_optim = opt.Adam(critic_model.parameters())

# DataLoaders
train_dl = torch.utils.data.DataLoader(
    trainDataset.TrainDataISTD(),
    batch_size=config.BATCH_SIZE,
    num_workers=config.N_WORKERS,
    pin_memory=True,
)
logging.debug("Train DataLoader")

# test_dl = torch.utils.data.DataLoader(
#     testDataset.TestDataISTD(),
#     batch_size=config.BATCH_SIZE,
#     num_workers=config.N_WORKERS,
#     pin_memory=True,
# )

# Other parameters
sim_params = {
    "num_iters": 1e4,
    "padding": 10,
    "k_shape": 5,
    "alpha": 0.90,
}

# Training Loop
for epoch in range(config.N_EPOCHS):
    # Set in training Mode
    logging.debug(f"EPOCH: {epoch}")
    actor_model.train()
    critic_model.train()

    for batch_idx, x in enumerate(train_dl):
        # Get one batch, move to GPU
        batch = x["shadow_image"].to(config.DEVICE)
        logging.debug(f"Batch No: {batch_idx} got")
        optimzr.zero_grad()

        x_shdw_free = None

        for step in range(config.N_TIMES_PER_BATCH):
            # 1. generate shadow shape
            logging.debug(f"Starting STEP: {step}")
            shpe = shdw_gen_model.simulate_shape(
                padding=sim_params["padding"],
                num_iters=sim_params["num_iters"],
            )
            logging.debug(f"Shape Generated. Shape Dimension: {shpe.shape}")

            # 2. Apply shadow on batch
            x_shadowfied = cps.modify_generated_shape(
                shpe, sim_params["k_shape"], batch, sim_params["alpha"]
            )
            logging.debug(f"Shape converted to Tensor and Applied on Batch. ")

            # Pass through actor NOTE: Possible Shape Error
            x_shdw_free = actor_model(x_shadowfied)
            logging.debug(f"Shadow Free Batch Generated. Shape: {x_shdw_free.shape}")

            # Pass through critic
            reward = critic_model(x_shadowfied, x_shdw_free).mean()
            logging.debug(f"Critic Model Computation Done. Reward shape {reward.shape}")

            # BackProp Reward
            reward.backward()
            optimzr.step()

            # Update Parameters based on reward -- Go for next iteration
            rew_value = reward.item()
            cps.update_shadow_params(rew_value, sim_params)
            logging.debug(f"Shadow Params Updated")

        if epoch % 10 == 9:
            logging.debug("Saving Tensor as Image")
            uts.save_batch_as_image(
                x_shdw_free,
                os.path.join(
                    config.LOG_DIR, f"Output_EPOCH_{epoch}_BATCH_{batch_idx}.png"
                ),
            )
            logging.debug(f"Saved Image. Tensor Shape: {x_shdw_free.shape}")
