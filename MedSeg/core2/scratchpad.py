""" # Test
import torch
from torch import rand
from components import MedSegModel
import config
import time

x = rand(8, 1, 240, 240, 155).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
y = rand(8, 1, 240, 240, 155).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# -----------------------
# model1 = Projection(n_output_channels=3, n_slices=155).to(config.DEVICE)
# model2 = PatchEmbedding(img_size=240, emb_size=720, patch_size=32).to(config.DEVICE)
# model3 = QKV(emb_size=720).to(config.DEVICE)
# model4 = MultiAttentionHead(emb_size=720).to(config.DEVICE)
# model5 = Fusion().to(config.DEVICE)
model = MedSegModel().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# -----------------------
start = time.time()
# out1 = model1(x)
# out1 = model2(out1)
# qkv = model3(out1)
# attn = model4(*qkv)
# final_vecs = model5(attn, attn)
final = model(x, y)
stop = time.time()
print(f"Time Needed is {stop - start} s")
# ----------------------
"""

from dataset import train_loader

print(*(next(iter(train_loader))).keys())
print((next(iter(train_loader)))["image"].shape)
