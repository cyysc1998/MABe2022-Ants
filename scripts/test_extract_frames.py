import os

from tqdm import tqdm

root = "../data/mouse"
# frame_dir = f"{root}/frames"
frame_dir = f"{root}/frames_512"
dirs = os.listdir(frame_dir)
print(len(dirs))
for dir in tqdm(dirs):
    names = os.listdir(f"{frame_dir}/{dir}")
    assert len(names) == 1801, f"{frame_dir}/{dir}"
