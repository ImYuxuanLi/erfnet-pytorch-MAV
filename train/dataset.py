import numpy as np
import os
import csv

from PIL import Image

from torch.utils.data import Dataset

def load_image(file):
    return Image.open(file)

class MAV(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        filelist = root + subset + '.csv'
        
        with open(filelist) as f:
            content = csv.reader(f)
            self.filenames = [root + i[0] for i in content]
        with open(filelist) as f:
            content = csv.reader(f)
            self.filenamesGt = [root + i[1] for i in content]

        self.co_transform = co_transform # ADDED THIS

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(filename, 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)