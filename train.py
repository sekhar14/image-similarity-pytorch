import torch
import torch.nn as nn
from torchvision import transforms, utils
from read_dataset import AnimalDataset
from torch.utils.data import Dataset, DataLoader
from model import Autoencoder
from custom_transforms import ReSize, ToTensor, Blurr, Sharpen



BATCH_SIZE = 24
dataset = AnimalDataset("./dataset",
                        transform=transforms.Compose([
                           ToTensor()
                        ]))


dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
epochs = 30

model = Autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

for epoch in range(epochs):
    epoch_loss = 0.0
    for random_image, image in dataloader:
        image = image.float()
        random_image = random_image.float()
        image = image.cuda()
        random_image = random_image.cuda()
        output, _ = model(random_image)
        loss = criterion(output, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data.cpu().numpy()
    print(f"epoch ::: {epoch}   loss ::: {epoch_loss}")

torch.save(model.state_dict(), "./trained_model8/trained_weights.h5")
