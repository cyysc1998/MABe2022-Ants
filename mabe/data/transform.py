import numpy as np
import torchvision
import torchvision.transforms as T


class TransformsSimCLR:
    def __init__(self, size, pretrained, n_channel, train):
        self.train_transforms = T.Compose(
            [
                T.RandomResizedCrop(size=size, scale=(0.25, 1.0)),
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
