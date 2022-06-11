import glob
import multiprocessing
import os
from re import sub
import sunau

import cv2
import numpy as np
from tqdm import tqdm

def main():
    user_train = '../data/ants/user_train.npy'
    submission_keypoints = '../data/ants/submission_keypoints.npy'
    train_keypoints = np.load(user_train, allow_pickle=True).item()['sequences']
    test_keypoints = np.load(submission_keypoints, allow_pickle=True).item()['sequences']
    keypoints = {**train_keypoints, **test_keypoints}

    crop_keypoints = {}
    padbbox = 115
    for i, (seq_name, seq_frame) in enumerate(keypoints.items()):
        seq_keypoints = seq_frame['keypoints'] # (900, 4)
        if seq_keypoints.shape[0] != 900:
            continue
        seq_keypoints = seq_keypoints.reshape(900, -1, 2) # (900, 2, 2)
        crop_k = np.zeros((900, 2, 2))
        for idx in range(900):
            minvals = (
                        max(np.min(seq_keypoints[idx, :, 0]) - padbbox, 0),
                        max(np.min(seq_keypoints[idx, :, 1]) - padbbox, 0),
                    )
            crop_k[idx] = seq_keypoints[idx, :, :] - minvals
        crop_keypoints[seq_name] = crop_k
        print('seq_name:', seq_name, i)
    np.save('../data/ants/crop_keypoints.npy', crop_keypoints)


def crop():
    user_train = '../data/ants/user_train.npy'
    submission_keypoints = '../data/ants/submission_keypoints.npy'
    train_keypoints = np.load(user_train, allow_pickle=True).item()['sequences']
    test_keypoints = np.load(submission_keypoints, allow_pickle=True).item()['sequences']
    keypoints = {**train_keypoints, **test_keypoints}

    np.save('../data/ants/all_keypoints.npy', keypoints)


if __name__ == '__main__':
    crop()