import glob

import cv2
import numpy as np
from tqdm import tqdm

from decord import VideoReader, cpu, gpu

# root = "/cache/video_clips_512"
root = "/home/admin/workspace/MABe/Round2/data/mouse/video_clips_512"
paths = glob.glob(f"{root}/*.avi")
print(len(paths))

for path in tqdm(paths[:100], desc="decord"):
    cap = VideoReader(path, ctx=cpu(0))
    # print(len(cap))
    frames = []
    for i in range(len(cap)):
        frame = cap[i].asnumpy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    # print(len(frames))

for path in tqdm(paths[:100], desc="opencv"):
    cap = cv2.VideoCapture(path)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(num_frame)
    frames = []
    while 1:
        success, frame = cap.read()
        if success:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        else:
            break
    cap.release()
    # print(len(frames))

"""
decord: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [03:25<00:00,  2.06s/it]
opencv: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [03:53<00:00,  2.33s/it]
"""
