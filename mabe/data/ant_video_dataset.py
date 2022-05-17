import os

import cv2
import numpy as np
import random
import torch
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from tqdm import tqdm


class AntVideoDataset(torch.utils.data.Dataset):
    """
    Reads frames from video files
    """

    def __init__(self, opt):
        """
        Initializing the dataset with images and labels
        """
        self.opt = opt
        self.video_dir = opt["video_dir"]
        self.num_frame = opt["num_frame"]
        self.frame_skip = opt["frame_skip"]
        self.num_prev_frames = opt["num_prev_frames"]
        self.num_next_frames = opt["num_next_frames"]
        self.frame_size = opt["frame_size"]
        self.has_label = opt["has_label"]

        # frame_number_map
        self.frame_number_map = np.load(
            opt["frame_number_map_path"], allow_pickle=True
        ).item()
        self.video_names = list(self.frame_number_map.keys())
        # IMPORTANT: the frame number map should be sorted
        frame_nums = np.array([self.frame_number_map[k] for k in self.video_names])
        assert np.all(np.diff(frame_nums[:, 0]) > 0), "Frame number map is not sorted"

        # labels
        self.labels = None
        if self.has_label:
            self.labels = np.load(opt["keypoint_path"], allow_pickle=True).item()
            self.labels = self.labels["sequences"]

    def __len__(self):
        return len(self.video_names) * self.num_frame

    def __getitem__(self, idx):
        ret = {}
        ret.update({"idx": idx})
        video_idx = idx // self.num_frame
        frame_idx = idx % self.num_frame
        video_name = self.video_names[video_idx]
        video_path = os.path.join(self.video_dir, video_name)
        
        frames = self.idx2clip(frame_idx)
        pos_frame_idx = self.sample_clip_idx()
        pos_frames = self.idx2clip(pos_frame_idx)
        
        ret.update({"x": frames})
        ret.update({"pos_x": pos_frames)
        ret.update({"seq_id": video_idx})
        

        if self.labels is not None:
            label = self.labels[video_name]["annotations"].T
            label = label[frame_idx]
            ret.update({"label": label})

        return ret


    def sample_clip_idx(self, base_idx):
        random_list = [
            *np.arange(0, base_idx - self.num_next_frames * self.frame_skip),
            *np.arange(base_idx + self.num_next_frames * self.frame_skip + 1, self.num_frames + 1)
        ]
        pos_idx = random.choice(random_list)
        return pos_idx

    
    def idx2clip(frame_idx):
        indices = np.array(
            list(
                range(
                    frame_idx - self.num_prev_frames * self.frame_skip,
                    frame_idx + self.num_next_frames * self.frame_skip + 1,
                    self.frame_skip,
                )
            )
        ).clip(0, self.num_frame - 1)
        
        frames = []
        for fnum in index:
            frame_path = os.path.join(video_path, f"{fnum}.jpg")
            frame = read_image(frame_path, mode=ImageReadMode.GRAY)
            frames.append(frame)
        frames = torch.cat(frames)
        return frames
        





