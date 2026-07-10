from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        super().__init__()
        self.data = data
        self.mode = mode

        transforms = [tv.transforms.ToPILImage()]
        if mode == 'train':
            transforms.append(tv.transforms.RandomHorizontalFlip())
            transforms.append(tv.transforms.RandomVerticalFlip())
        transforms.append(tv.transforms.ToTensor())
        transforms.append(tv.transforms.Normalize(train_mean, train_std))
        self._transform = tv.transforms.Compose(transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image = imread(row['filename'])
        image = gray2rgb(image)
        image = self._transform(image)
        label = torch.tensor([row['crack'], row['inactive']], dtype=torch.float)
        return image, label
