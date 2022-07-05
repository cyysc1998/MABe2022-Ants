import numpy as np

root = "../data/mouse"
keypoints_path = f"{root}/user_train.npy"
keypoints = np.load(keypoints_path, allow_pickle=True).item()["sequences"]
frame_number_map_path = f"{root}/frame_number_map_testing.npy"
frame_number_map = {}
i = 0
for k, v in keypoints.items():
    # print(k, v['keypoints'].shape)
    n = v["keypoints"].shape[0]
    frame_number_map[k] = [i, i + n]
    i += n
np.save(frame_number_map_path, frame_number_map)
frame_number_map = np.load(frame_number_map_path, allow_pickle=True).item()

for k, v in frame_number_map.items():
    print(k, v)
print(len(frame_number_map))
