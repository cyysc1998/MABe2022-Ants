import os

import numpy as np
from tqdm import tqdm

# root = "/home/admin/workspace/mouse/frames_512_npy"
root = "/cache"
frames = {}
names = os.listdir(root)
for name in tqdm(names):
    path = f"{root}/{name}"
    k = name.replace(".npy", "")
    v = np.load(path)
    frames[k] = v
