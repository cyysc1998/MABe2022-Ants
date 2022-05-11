import os

import numpy as np
from tqdm import tqdm

frame_dir = f"/cache"
names = os.listdir(frame_dir)
for name in tqdm(names):
    path = f"{frame_dir}/{name}"
    frames = np.load(path)
    assert frames.shape[0] == 1800, path
