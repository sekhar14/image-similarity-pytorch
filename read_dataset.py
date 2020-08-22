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
        gaussian = np.random.normal(0, 10 ** 0.5, (512, 512))
        noisy_image = np.zeros(image.shape, np.float32)
        noisy_image[:, :, 0] = image[:, :, 0] + gaussian
        noisy_image[:, :, 1] = image[:, :, 1] + gaussian
        noisy_image[:, :, 2] = image[:, :, 2] + gaussian
        if self.transform:
            image = self.transform(image)
            noisy_image = self.transform(noisy_image)
        return noisy_image, image
