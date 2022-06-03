import os

import cv2
import math
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
        self.is_train = (
            'train' in os.path.basename(opt["frame_number_map_path"])
        )
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
        
        if self.is_train:
            frame_idx, pos_frame_idx = self.sample_ssl_sequence(self.num_frame, self.num_prev_frames, self.frame_skip)
        else:
            pos_frame_idx = frame_idx
            
        frames = self.idx2clip(frame_idx, video_path)
        pos_frames = self.idx2clip(pos_frame_idx, video_path)
        
        ret.update({"x1": frames})
        ret.update({"x2": pos_frames})
        ret.update({"seq_id": video_idx})
        

        if self.labels is not None:
            label = self.labels[video_name]["annotations"].T
            label = label[frame_idx]
            ret.update({"label": label})

        return ret


    def sample_clip_idx(self, base_idx):
        random_list = [
            *np.arange(0, base_idx - self.num_next_frames * self.frame_skip),
            *np.arange(base_idx + self.num_next_frames * self.frame_skip + 1, self.num_frame + 1)
        ]
        pos_idx = random.choice(random_list)
        return pos_idx

    
    def idx2clip(self, frame_idx, video_path):
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
        for fnum in indices:
            frame_path = os.path.join(video_path, f"{fnum}.jpg")
            frame = read_image(frame_path, mode=ImageReadMode.GRAY)
            frames.append(frame)
        frames = torch.cat(frames)
        return frames
        
        
        
    def sample_ssl_sequence(self, sequence, num_steps, stride):
        '''
            Pytorch version of https://github.com/tensorflow/models/blob/14342e4ecf/official/projects/video_ssl/ops/video_ssl_preprocess_ops.py
        '''
        sequence_length = sequence
        max_offset = lambda: sequence_length - (num_steps - 1) * stride \
            if sequence_length > (num_steps - 1) * stride \
            else \
            lambda: sequence_length
        max_offset = max_offset()

        def cdf(k, power=1.0):
            """Cumulative distribution function for x^power."""
            p = -math.pow(k, power + 1) / (
                power * math.pow(max_offset, power + 1)) + k * (power + 1) / (
                    power * max_offset
                )
            return p

        u = torch.Tensor(1).uniform_(0, 1)
        k_low = torch.tensor([0.0])
        k_up = max_offset
        k = math.floor(max_offset / 2)

        c = lambda k_low, k_up, k: math.fabs(k_up - k_low) > 1.0

        b = lambda k_low, k_up, k: \
            [k_low, k, math.floor((k + k_low) / 2.0)] if cdf(k) > u else \
            [k, k_up, math.floor((k_up + k) / 2.0)]

        while c(k_low, k_up, k):
            k_low, k_up, k = b(k_low, k_up, k)

        delta = k

        choice_1 = \
            torch.randint(0, max_offset, [1]) \
            if max_offset == sequence_length else \
            torch.randint(0, max_offset - delta, [1])

        choice_2 = \
            torch.randint(0, max_offset, [1]) \
            if max_offset == sequence_length else \
            choice_1 + delta

        indices = [choice_1, choice_2]
        random.shuffle(indices)

        return indices





