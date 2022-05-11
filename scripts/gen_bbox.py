"""
Prepare bounding boxes to be used for cropping frames during training from keypoints.
"""
import os

import numpy as np
from tqdm import tqdm

datafolder = "../data/mouse"
# keypoint_path = os.path.join(datafolder, "submission_keypoints.npy")
keypoint_path = os.path.join(datafolder, "user_train.npy")
new_keypoint_path = keypoint_path.replace(".npy", "_bbox.npy")
keypoints = np.load(keypoint_path, allow_pickle=True).item()

padbbox = 50
crop_size = 512
for sk in tqdm(keypoints["sequences"].keys()):
    kp = keypoints["sequences"][sk]["keypoints"]
    bboxes = []
    for frame_idx in range(len(kp)):
        allcoords = np.int32(kp[frame_idx].reshape(-1, 2))
        minvals = max(np.min(allcoords[:, 0]) - padbbox, 0), max(
            np.min(allcoords[:, 1]) - padbbox, 0
        )
        maxvals = min(np.max(allcoords[:, 0]) + padbbox, crop_size), min(
            np.max(allcoords[:, 1]) + padbbox, crop_size
        )
        bbox = (*minvals, *maxvals)
        bbox = np.array(bbox)
        bbox = np.int32(bbox * 224 / 512)
        bboxes.append(bbox)
    keypoints["sequences"][sk]["bbox"] = np.array(bboxes)

keypoints = np.save(new_keypoint_path, keypoints)
# keypoints = np.load(new_keypoint_path, allow_pickle=True).item()
