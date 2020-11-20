import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import tifffile as tiff
import pandas as pd
import os
import numpy as np

from skimage import io, transform

class GroundTruthDataset(Dataset):
    """ Ground Truth dataset. """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = os.listdir()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.images[idx])
        image = io.imread(img_name) // Something with tiff?
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


