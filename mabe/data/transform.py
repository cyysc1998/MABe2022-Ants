import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as T


class TransformsSimCLR:
    def __init__(self, size, pretrained, n_channel, train):
        self.train_transforms = T.Compose(
            [
                T.RandomResizedCrop(size=size, scale=(0.25, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                # GaussianBlur(kernel_size=int(0.1 * size[0]), n_channel=n_channel),
                T.GaussianBlur(kernel_size=int(0.1 * size[0]) if size[0] % 2 == 1 else int(0.1 * size[0])+1),
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
                T.RandomErasing(),
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
        if self.train:
            return self.train_transforms(x)
        else:
            return self.validation_transforms(x)

        
class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size, n_channel):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(n_channel, n_channel, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=n_channel)
        self.blur_v = nn.Conv2d(n_channel, n_channel, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=n_channel)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )
        
        self.n_channel = n_channel


    def __call__(self, img):

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(self.n_channel, 1)

        self.blur_h.weight.data.copy_(x.view(self.n_channel, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(self.n_channel, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        return img