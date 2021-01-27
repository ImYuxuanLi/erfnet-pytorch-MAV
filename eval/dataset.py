# Code with dataset loader for MAV (adapted from bodokaiser/piwise code)
# Jan 2021
# Yuxuan Li
#######################

import numpy as np
import os
import csv

from PIL import Image

from torch.utils.data import Dataset

def load_image(file):
    return Image.open(file)

class MAV(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, subset='val'):
        filelist = root + subset + '.csv'
        
        with open(filelist) as f:
            content = csv.reader(f)
            self.filenames = [root + i[0] for i in content]
        with open(filelist) as f:
            content = csv.reader(f)
            self.filenamesGt = [root + i[1] for i in content]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(filename, 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label, filename, filenameGt

    def __len__(self):
        return len(self.filenames)