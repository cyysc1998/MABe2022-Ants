import glob
import multiprocessing
import os

import cv2
import numpy as np
from tqdm import tqdm

# keypoints
root = "../data/mouse"
keypoint_paths = [
    os.path.join(root, "user_train.npy"),
    os.path.join(root, "submission_keypoints.npy"),
]
keypoints = {}
for path in keypoint_paths:
    keypoint = np.load(path, allow_pickle=True).item()["sequences"]
    keypoints = {**keypoints, **keypoint}
print(len(keypoints))
num_frame = 1800

# videos
paths = glob.glob(f"{root}/video_clips_512/*.avi")
print(len(paths))
frame_dir = os.path.join(root, "frames_cropped_224")
os.makedirs(frame_dir, exist_ok=True)

# crop
crop_size = 512
padbbox = 50


def extract_frames(path):
    # video
    name = os.path.basename(path).replace(".avi", "")
    if os.path.exists(f"{frame_dir}/{name}.npy"):
        return
    os.makedirs(f"{frame_dir}/{name}", exist_ok=True)
    cap = cv2.VideoCapture(path)
    # num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # keypoints
    kp = keypoints[name]["keypoints"]

    frames = []
    frame_idx = 0
    while 1:
        success, frame = cap.read()
        if success:
            cv2.imwrite(f"{frame_dir}/{name}/{frame_idx}_full.jpg", frame)
            # keypoint
            allcoords = np.int32(kp[frame_idx].reshape(-1, 2))
            allcoords = allcoords[allcoords.sum(1) > 0]
            if allcoords.shape[0] == 0:
                allcoords = np.int32([[0, crop_size]])
            minvals = (
                max(np.min(allcoords[:, 0]) - padbbox, 0),
                max(np.min(allcoords[:, 1]) - padbbox, 0),
            )
            maxvals = (
                min(np.max(allcoords[:, 0]) + padbbox, crop_size),
                min(np.max(allcoords[:, 1]) + padbbox, crop_size),
            )
            bbox = (*minvals, *maxvals)
            bbox = np.array(bbox)
            bbox = np.int32(bbox)
            # crop
            frame = frame[bbox[0] : bbox[2], bbox[1] : bbox[3]]
            # resize
            frame = cv2.resize(frame, (224, 224), cv2.INTER_CUBIC)
            # save
            cv2.imwrite(f"{frame_dir}/{name}/{frame_idx}.jpg", frame)
            frames.append(frame[:, :, 0])
            frame_idx += 1
        if frame_idx >= num_frame:
            break
    frames = np.stack(frames)
    print(frames.shape)
    # np.save(f"{frame_dir}/{name}.npy", frames)
    cap.release()


# for path in tqdm(paths):
#     extract_frames(path)
pbar = tqdm(total=len(paths))
update = lambda *args: pbar.update()
pool = multiprocessing.Pool(32)
for path in paths:
    pool.apply_async(extract_frames, (path,), callback=update)
print("Start")
pool.close()
pool.join()
print("Done")
