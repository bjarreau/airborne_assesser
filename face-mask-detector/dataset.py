import torch
import torch.utils.data
import glob
from PIL import Image
import subprocess
import os
import uuid

class ImageClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, dir, transform=None):
        self.files = os.listdir(dir)
        self.dir = dir
        self.transform = transform
        if "with_mask" is in dir:
            self.label = 1
        else:
            self.label = 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform is not None:
            image = self.transform(image)
        return image, self.label