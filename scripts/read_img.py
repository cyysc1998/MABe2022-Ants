import glob
import os

import cv2
from tqdm import tqdm

import jpeg4py as jpeg

root = "/home/admin/workspace/mouse/frames_512"
video_names = os.listdir(root)
print(len(video_names))
paths = glob.glob(f"{root}/{video_names[0]}/*.jpg")

# jpeg4py
for path in tqdm(paths):
    img = jpeg.JPEG(path).decode()
    # print(type(img), img.shape)
    # print(img[0, 0])

# cv2
for path in tqdm(paths):
    img = cv2.imread(path)
    # print(type(img), img.shape)
    # print(img[0, 0])

"""
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1801/1801 [00:05<00:00, 332.60it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1801/1801 [00:10<00:00, 173.06it/s]
"""
