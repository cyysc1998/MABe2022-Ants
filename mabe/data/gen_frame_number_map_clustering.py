import numpy as np
import random
import os
random.seed(0)

root = "../data/ants"
datafolder = '../data/ants'
frame_number_map_training = np.load(os.path.join(datafolder, 'frame_number_map_training.npy'), allow_pickle=True).item()
keypoints = list(frame_number_map_training.keys())

random.shuffle(keypoints)
select_train_keys = keypoints[:len(keypoints)//256*30]

frame_number_map_path = f"{root}/frame_number_map_clustering_test.npy"
frame_number_map = {}
i = 0
for k in select_train_keys:
    print(k)
    n = 900
    frame_number_map[k] = [i, i + n]
    i += n
print(len(select_train_keys))
print(len(frame_number_map.keys()))
np.save(frame_number_map_path, frame_number_map)
# frame_number_map = np.load(frame_number_map_path, allow_pickle=True).item()
# print(len(frame_number_map))
# for k, v in frame_number_map.items():
#     print(k, v)
