import numpy as np
import torch

import torchvision
import torchvision.transforms as T


class TransformsSimCLR:
    def __init__(self, size, pretrained, n_channel, train):
        self.train_transforms = T.Compose(
            [
                T.RandomResizedCrop(size=size, scale=(0.25, 1.0)),
                T.RandomGrayscale(p=0.2),
                T.ColorJitter(0.4, 0.4, 0.4, 0.4),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                # Taking the means of the normal distributions of the 3 channels
                # since we are moving to grayscale
                T.Normalize(
                    mean=np.mean([0.485, 0.456, 0.406]).repeat(n_channel),
                    std=np.sqrt(
                        (np.array([0.229, 0.224, 0.225]) ** 2).sum() / 9
                    ).repeat(n_channel),
                )
                if pretrained is True
                else T.Lambda(lambda x: x),
            ]
        )

        self.train_transforms_td = T.Compose(
            [
                T.RandomResizedCrop(size=size, scale=(0.25, 1.0)),
                T.RandomGrayscale(p=0.2),
                T.ColorJitter(0.4, 0.4, 0.4, 0.4),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                TemporalDifference(p=0.5),
                # Taking the means of the normal distributions of the 3 channels
                # since we are moving to grayscale
                T.Normalize(
                    mean=np.mean([0.485, 0.456, 0.406]).repeat(n_channel),
                    std=np.sqrt(
                        (np.array([0.229, 0.224, 0.225]) ** 2).sum() / 9
                    ).repeat(n_channel),
                )
                if pretrained is True
                else T.Lambda(lambda x: x),
            ]
        )

        self.validation_transforms = T.Compose(
            [
                T.Resize(size=size),
                # Taking the means of the normal distributions of the 3 channels
                # since we are moving to grayscale
                T.Normalize(
                    mean=np.mean([0.485, 0.456, 0.406]).repeat(n_channel),
                    std=np.sqrt(
                        (np.array([0.229, 0.224, 0.225]) ** 2).sum() / 9
                    ).repeat(n_channel),
                )
                if pretrained is True
                else T.Lambda(lambda x: x),
            ]
        )

        self.train = train

    def __call__(self, x):
        if self.train == True:
            return self.train_transforms(x)
        elif self.train == 'td':
            return self.train_transforms_td(x)
        else:
            return self.validation_transforms(x)


class TemporalDifference(object):
    """blur a single image on CPU"""
    def __init__(self, p=0.5):
       self.p = p

    def __call__(self, img):
        assert len(img.shape) == 4, f"Img shape is {img.shape}"
        B, C, H, W = img.shape
        if torch.rand(1) < self.p:
            img = img.permute(1, 0, 2, 3)
            img[1:] = img[1:] - img[0:C-1]
            img = img.permute(1, 0, 2, 3)
        
        return img

