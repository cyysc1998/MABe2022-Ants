import glob
import multiprocessing
import os

import cv2
import numpy as np
from tqdm import tqdm

root = "../data/mouse"
frame_number_map = np.load(f"{root}/frame_number_map.npy", allow_pickle=True).item()
for k, v in frame_number_map.items():
    print(k, v)
    break


# paths = glob.glob(f"{root}/video_clips/*.avi")
# frame_dir = f"/home/admin/workspace/mouse/frames"
paths = glob.glob(f"{root}/video_clips_512/*.avi")
frame_dir = f"/home/admin/workspace/mouse/frames_512_npy"
os.makedirs(frame_dir, exist_ok=True)
print(len(paths))


def extract_frames(path):
    name = os.path.basename(path).replace(".avi", "")
    cap = cv2.VideoCapture(path)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(num_frame)

    frames = []
    while 1:
        success, frame = cap.read()
        if success:
            frames.append(frame[:, :, 0])
        else:
            break
    frames = np.stack(frames)
    np.save(f"{frame_dir}/{name}.npy", frames)
    cap.release()


pbar = tqdm(total=len(paths))
update = lambda *args: pbar.update()
pool = multiprocessing.Pool(64)
for path in paths:
    pool.apply_async(extract_frames, (path,), callback=update)
print("Start")
pool.close()
pool.join()
print("Done")
