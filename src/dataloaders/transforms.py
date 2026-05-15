import random

import torchvision.transforms.functional as F
import numpy as np
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode
from PIL import Image

from src.dataloaders.Itransforms import ITransforms


class RandomCrop:
    """
    Compared to Pytorch's RandomCrop, this implementation returns a crop size between crop_size and the image size.
    Therefore, it needs to be followed by a resize operation to get a fixed size output.
    """
    def __init__(self, crop_size: int | tuple[int, int]):
        if isinstance(crop_size, int):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.crop_size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        img = F.crop(img, i, j, h, w)

        return img

class RandomSpNoise:

    def __init__(self, p: float=0.25, mean: float=0, var: float=0.005):
        self.p = p
        self.mean = mean
        self.var = var

    def __call__(self, img):
        if random.random() < self.p:
            img = np.asarray(img)
            img = np.array(img / 255, dtype=float)

            noise = np.random.normal(self.mean, self.var ** 0.5, img.shape)
            img = img + noise

            if img.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.

            img = np.clip(img, low_clip, 1.0)
            img = np.uint8(img * 255)
            img = Image.fromarray(np.uint8(img))

        return img


class TransformsSupervisedLearningTrain(ITransforms):

    def __init__(self, output_dim: int=304):
        self.output_dim = output_dim

    def compose(self) -> T.Compose:
        """Returns the standard ImageNet transforms for linear probing."""
        transform = T.Compose([
            T.Grayscale(num_output_channels=3),
            RandomSpNoise(),
            T.RandomApply([T.RandomRotation(degrees=45, interpolation=InterpolationMode.BILINEAR)], p=0.6),
            T.RandomApply(
                [
                    T.ColorJitter(brightness=(0.2, 0.8), contrast=(0.2, 0.8), saturation=(0.2, 0.8))
                ], p=0.25),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            RandomCrop(self.output_dim),
            T.Resize((self.output_dim, self.output_dim)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.339], [0.138]),
        ])
        return transform


class TestTransforms(ITransforms):

    def __init__(self, output_dim: int=304):
        self.output_dim = output_dim

    def compose(self) -> T.Compose:
        """Returns the standard ImageNet transforms for linear probing."""
        transform = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize((self.output_dim, self.output_dim)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.339], [0.138]),
        ])
        return transform


class TransformsLinearProbingTrain(ITransforms):

    def __init__(
            self,
            output_dim: int=304,
            in_channels: int=3
    ):
        self.output_size_global = output_dim
        self.in_channels = in_channels
        if in_channels == 3:
            self.normalize = T.Normalize([0.339, 0.339, 0.339], [0.138, 0.138, 0.138])
        else:
            self.normalize = T.Normalize([0.339], [0.138])

    def compose(self) -> T.Compose:
        """Returns the standard ImageNet transforms for linear probing."""
        transform = T.Compose([
            T.Grayscale(num_output_channels=self.in_channels),
            T.RandomApply([T.RandomRotation(degrees=45, interpolation=InterpolationMode.BILINEAR)], p=0.6),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomResizedCrop(size=self.output_size_global, scale=(0.8, 1.0)),  # , ratio=(0.8, 1.25)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            self.normalize,
        ])
        return transform