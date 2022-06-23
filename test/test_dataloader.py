from tqdm import tqdm

from mabe.data import create_dataloader, create_dataset
from mabe.utils import parse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

phase = "train"
opt = parse("options/31_supervised.yml")
dataset_opt = opt["datasets"]["val"]
dataset_opt.update({"phase": phase})
train_set = create_dataset(dataset_opt)
# sampler = DistributedSampler(train_set, shuffle=False)
# train_loader = create_dataloader(
#     train_set, dataset_opt, seed=opt["manual_seed"], sampler=None
# )
train_loader = DataLoader(train_set, 128, False)
num_nan = 0
for idx, data in tqdm(enumerate(train_loader)):
    # print(idx)
    # for k, v in data.items():
    #     print(k, v.shape)
    # break
    label = data["label"]
    if torch.isnan(label).any():
        print(label)
        break
    # num_nan += (label == np.nan).sum()
    # print(num_nan)

"""
bs=128, num_workers=64
cv2:            384it [05:28,  1.17it/s]
cv2_gray:       576it [06:03,  1.58it/s]
jpeg4py:        206it [02:45,  1.25it/s]
jpeg4py_wo_cvt: 448it [04:29,  1.67it/s]

bs=128, num_workers=8
jpeg4py_wo_cvt: 152it [09:31,  3.76s/it]
"""
