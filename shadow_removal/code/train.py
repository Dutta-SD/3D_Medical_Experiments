# Training Loop
import torch
import torch.optim as opt
import components as cps
import config
import utils as uts
from datasets import testDataset, trainDataset

# define the model components
actor_model = cps.ActorUNet().to(config.DEVICE)
critic_model = cps.Critic().to(config.DEVICE)
shdw_gen_model = cps.PseudoShadowGenerator()
shdw_upd_model = cps.apply_shadow_mask_2_batch

# optimizers
optimizer = opt.Adam(
    list(actor_model.parameters()) + list(critic_model.parameters()),
)

# DataLoaders
train_dl = torch.utils.data.DataLoader(
    trainDataset.TrainDataISTD(),
    batch_size=config.BATCH_SIZE,
    num_workers=config.N_WORKERS,
    pin_memory=True,
)

test_dl = torch.utils.data.DataLoader(
    testDataset.TestDataISTD(),
    batch_size=config.BATCH_SIZE,
    num_workers=config.N_WORKERS,
    pin_memory=True,
)

# Other parameters
padding = 3
num_iters = 1e4

# Training Loop
for epoch in range(config.N_EPOCHS):
    # Set in training Mode
    actor_model.train()
    critic_model.train()

    for batch_idx, x in enumerate(train_dl):
        # Get one batch, move to GPU
        batch = x["shadow_image"].to(config.DEVICE)

        for step in range(config.N_TIMES_PER_BATCH):
            # 1. generate shadow on batch
            # initial padding = 3, num_iters = 1e4
            shape = shdw_gen_model.simulate_shape(padding=padding, num_iters=num_iters)
