from tqdm import tqdm

from mabe.data import create_dataloader, create_dataset
from mabe.utils import parse

phase = "train"
opt = parse("options/00_baseline.yml")
dataset_opt = opt["datasets"]["train"]
dataset_opt.update({"phase": phase})
train_set = create_dataset(dataset_opt)
train_loader = create_dataloader(
    train_set, dataset_opt, seed=opt["manual_seed"], phase=phase
)
for idx, data in tqdm(enumerate(train_loader)):
    # print(idx)
    # for k, v in data.items():
    #     print(k, v.shape)
    # break
    pass

"""
bs=128, num_workers=64
cv2:            384it [05:28,  1.17it/s]
cv2_gray:       576it [06:03,  1.58it/s]
jpeg4py:        206it [02:45,  1.25it/s]
jpeg4py_wo_cvt: 448it [04:29,  1.67it/s]

bs=128, num_workers=8
jpeg4py_wo_cvt: 152it [09:31,  3.76s/it]
"""
