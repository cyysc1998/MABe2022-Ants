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

        self.keypoints = np.load(opt["keypoint_path"], allow_pickle=True).item()

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
        
        frames = self.idx2clip(frame_idx, video_path, is_full=False)
        full_frames = self.idx2clip(frame_idx, video_path, is_full=True)
        pos_frame_idx = self.sample_clip_idx(frame_idx)
        pos_frames = self.idx2clip(pos_frame_idx, video_path, is_full=False)
        full_pos_frames = self.idx2clip(pos_frame_idx, video_path, is_full=True)
        
        ret.update({"x1": frames})
        ret.update({"x2": pos_frames})
        frame_keypoints = self.keypoints[video_name]["keypoints"][frame_idx-1]
        pos_frame_keypoints = self.keypoints[video_name]["keypoints"][pos_frame_idx-1]
        y1, x1, y2, x2 = frame_keypoints
        ret.update({"x1_a": self.crop(full_frames, [x1, y1], size=112)})
        ret.update({"x1_b": self.crop(full_frames, [x2, y2], size=112)})
        pos_y1, pos_x1, pos_y2, pos_x2 = pos_frame_keypoints
        ret.update({"x2_a": self.crop(full_pos_frames, [pos_x1, pos_y1], size=112)})
        ret.update({"x2_b": self.crop(full_pos_frames, [pos_x1, pos_y1], size=112)})
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

    
    def idx2clip(self, frame_idx, video_path, is_full=False):
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
            if not is_full:
                frame_path = os.path.join(video_path, f"{fnum}.jpg")
            else:
                frame_path = os.path.join(video_path, f"{fnum}_full.jpg")
            frame = read_image(frame_path, mode=ImageReadMode.GRAY)
            frames.append(frame)
        frames = torch.cat(frames)
        return frames


    def crop(self, frame, center, size=112):
        """
        Crop the frame with the given center and size
        frame 7 * W * H
        center (y, x)
        """
        center = np.array(center) * 512
        y1, x1 = center[0] - size // 2, center[1] - size // 2
        y2, x2 = y1 + size, x1 + size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[2], y2)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        crop_frame = frame[:, x1:x2, y1:y2].float()
        crop_frame = torch.nn.functional.interpolate(
            crop_frame.unsqueeze(0), size=size, mode="bilinear", align_corners=False
        )[0]
        return crop_frame
        
        





