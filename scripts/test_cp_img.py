import os

from tqdm import tqdm

dst_root = "/cache"
names = os.listdir(dst_root)
names = list(set([name.replace(".npy", "") for name in names]))


for name in tqdm(names):
    assert len(os.listdir(f"{dst_root}/{name}")) == 1800
