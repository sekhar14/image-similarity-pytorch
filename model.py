import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.maxpool = nn.MaxPool2d(3)
        self.upsample = nn.Upsample(size=(512, 512))
        self.convt1 = nn.ConvTranspose2d(16, 8, kernel_size=3)
        self.convt2 = nn.ConvTranspose2d(8, 3, kernel_size=3)
        self.convt3 = nn.ConvTranspose2d(8, 3, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        input_size = x.size()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        required_feature = x.clone()
        x = self.relu(self.convt1(x))
        x = self.convt2(x)
        return x, required_feature


