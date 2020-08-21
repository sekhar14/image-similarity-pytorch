import os
import torch
from skimage import io, transform
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class AnimalDataset(Dataset):
    def __init__(self, directory_name, transform=None):
        self.directory_name = directory_name
        self.all_images = os.listdir(directory_name)
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.directory_name, self.all_images[idx])
        image = cv2.imread(img_name)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image / 255.0
        return image
