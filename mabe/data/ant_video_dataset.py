import os
import random

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode


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
        self.num_segments = opt["num_segments"]
        self.num_frame_per_segment = opt["num_frames_per_segment"]
        self.segment_duration = opt["segment_duration"]

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

        if self.labels is not None:
            label = self.labels[video_name]["annotations"].T
            label = label[frame_idx]
            ret.update({"label": label})

        
        # print(frame_idxs)
        frames_list = self._sample_indices(self.num_segments, self.num_frame_per_segment, self.segment_duration)
        frames = []
        for fnum in frames_list:
            frame_path = os.path.join(video_path, f"{fnum}.jpg")
            frame = read_image(frame_path, mode=ImageReadMode.RGB)
            frames.append(frame)
        frames = torch.stack(frames, dim=0) # (T, C, H, W)
        ret.update({"x": frames})

        return ret

    def _sample_indices(self, num_segments, num_frames_per_segment, segment_duration):
        """
        Samples indices for segments.
        """
        indices = []
        time_duration = (num_segments - 1) * segment_duration + num_frames_per_segment
        assert time_duration < self.num_frame // 2, "Not enough frames to sample"
        assert num_segments % 2 == 1
        center_frame_idx = random.randint(0, self.num_frame)
        left_frame_idx = center_frame_idx -  segment_duration * num_segments // 2 - num_frames_per_segment // 2
        left_frame_idxs = [left_frame_idx + i * segment_duration for i in range(0, num_segments)]
        for i in left_frame_idxs:
            for j in range(num_frames_per_segment):
                indices.append(i + j)
                
        indices = np.array(indices).clip(0, self.num_frame - 1)
        return indices
