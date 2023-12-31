# Copied from https://github.com/ssg-research/WAFFLE/blob/main/src/data_handle.py

import glob
import os

import numpy as np
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import torchvision as tv
from PIL import Image

random_seed = 1234

# Class Pattern is a dataset for the watermarking dataset.
class Pattern(data.Dataset):
    # This is used to load images from the defined pattern image file

    def __init__(self, root_dir: str, train: bool, transform: tv.transforms.Compose, download: bool,
                 use_probs: bool = False, return_image_path: bool = False, n_classes: int  = 10) -> None:
        """
        Args:
            root_dir(string): file containing name of all images
            train (bool): gets only train or test set (not used here since ImageNet is used for out-of-distribution data only, used for API consistency)
            transform (torch func): transforms PIL image to torch data
            download (bool): indicates if the ImageNet should be downloaded (used for API consistency, doesn't do anything here)
            nrows (int, opt): gets only first nrows image to the dataset
            use_probs(bool, optional): labels will be avector containing probabilities for each class
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_probs = use_probs
        self.return_image_path = return_image_path
        self.n_classes = n_classes

        self.data_frame = []
        self.label_frame = []
        for i in range(0, self.n_classes):
            dirs = os.path.join(self.root_dir, '%d' % i)
            for idx, fimg in enumerate(glob.glob(os.path.join(dirs, '*.png'))):
                
                if idx < 10:
                    self.data_frame.append(fimg)
                    self.label_frame.append(i)


    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> (torch.Tensor, int):
        img_name = self.data_frame[idx]
        image = Image.open(img_name)
        image = image.convert('RGB')
        image = self.transform(image)
        if self.use_probs:
            label = np.asarray(self.label_frame.iloc[idx, :], dtype=np.float32)
        else:
            label = self.label_frame[idx]

        if self.return_image_path:
            sample = [image, label, img_name]
        else:
            sample = [image, label]
        return sample




