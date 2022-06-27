import numpy as np
import random
random.seed(0)
root = "../data/ants"
keypoints_train_path = f"{root}/user_train.npy"
keypoints_test_path = f"{root}/submission_keypoints.npy"
keypoints_train = np.load(keypoints_train_path, allow_pickle=True).item()["sequences"]
keypoints_test = np.load(keypoints_test_path, allow_pickle=True).item()["sequences"]
train_keys, test_keys = list(keypoints_train.keys()), list(keypoints_test.keys())

random.shuffle(train_keys)
random.shuffle(test_keys)
keypoints = train_keys + test_keys

frame_number_map_path = f"{root}/frame_number_map_all_data.npy"
frame_number_map = {}
i = 0
for k in keypoints:
    print(k)
    n = 900
    frame_number_map[k] = [i, i + n]
    i += n
print(len(keypoints))
print(len(frame_number_map.keys()))
np.save(frame_number_map_path, frame_number_map)
