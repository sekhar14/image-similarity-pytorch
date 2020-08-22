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
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3)
        self.convt1 = nn.ConvTranspose2d(2, 3, kernel_size=3)
        self.maxpool2d = nn.MaxPool2d(3, stride=5)
        self.upsample = nn.Upsample((506, 506))
        self.convt2 = nn.ConvTranspose2d(3, 3, kernel_size=3)
        self.convt3 = nn.ConvTranspose2d(3, 3, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = self.dropout(self.relu(self.conv3(x)))
        x = self.maxpool2d(x)
        required_feature = x.clone()
        x = self.upsample(x)
        x = self.relu(self.convt1(x))
        x = self.relu(self.convt2(x))
        x = self.relu(self.convt2(x))
        return x, required_feature


