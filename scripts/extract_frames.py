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
frame_dir = f"/home/admin/workspace/mouse/frames_512"
print(len(paths))


def extract_frames(path):
    name = os.path.basename(path).replace(".avi", "")
    os.makedirs(f"{frame_dir}/{name}", exist_ok=True)
    cap = cv2.VideoCapture(path)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(num_frame)

    i = 0
    while 1:
        success, frame = cap.read()
        if success:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"{frame_dir}/{name}/{i}.jpg", gray)
            i += 1
        else:
            break
    cap.release()


pbar = tqdm(total=len(paths))
update = lambda *args: pbar.update()
pool = multiprocessing.Pool(32)
for path in paths:
    pool.apply_async(extract_frames, (path,), callback=update)
print("Start")
pool.close()
pool.join()
print("Done")
