import torch
from tqdm import tqdm

from mabe.models.archs import define_network
from mabe.utils import parse

opt = parse("options/00_baseline.yml")
model = define_network(opt["network"])
x1 = torch.randn((4, 7, 224, 224))
x2 = torch.randn((4, 7, 224, 224))
out = model(x1, x2)
for y in out:
    print(y.shape)

"""
torch.Size([4, 2048])
torch.Size([4, 2048])
torch.Size([4, 128])
torch.Size([4, 128])
"""
