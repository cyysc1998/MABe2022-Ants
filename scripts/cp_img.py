import multiprocessing
import os
import shutil

from tqdm import tqdm

src_root = "/home/admin/workspace/mouse/frames_cropped_224_npy"
dst_root = "/cache"
names = os.listdir(src_root)
names = list(set([name.replace(".npy", "") for name in names]))


def cp(name):
    src_dir = os.path.join(src_root, name)
    dst_dir = os.path.join(dst_root, name)
    os.makedirs(dst_dir, exist_ok=True)
    for i in range(1800):
        src_path = os.path.join(src_dir, f"{i}.jpg")
        shutil.copy(src_path, dst_dir)


# for name in tqdm(names):
#     cp(name)

pbar = tqdm(total=len(names))
update = lambda *args: pbar.update()
pool = multiprocessing.Pool(128)
for name in names:
    pool.apply_async(cp, (name,), callback=update)
print("Start")
pool.close()
pool.join()
print("Done")
