import random

import numpy as np

random.seed(0)

num_video = 1948
num_frame = 900

index_list = []
for i in range(9):
    index = list(range(num_video * num_frame))
    random.shuffle(index)
    index_list.append(index)
np.savetxt("../data/ants/meta_info_val_0.txt", np.array(index_list), fmt="%s")
