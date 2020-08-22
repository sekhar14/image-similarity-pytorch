import faiss
import numpy as np
import torch
from model import Autoencoder
import os
import argparse
import cv2
from custom_transforms import ToTensor
import math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("k")
parser.add_argument("target_folder")
args = parser.parse_args()


C = 3 ## number of columns in plot
R = math.ceil(int(args.k) / C) + 1 ## number of rows in plot
index = faiss.IndexFlatL2(20402)
model = Autoencoder().cuda()
model.load_state_dict(torch.load("./trained_model8/trained_weights.h5"))
model.eval()
totensor = ToTensor()

all_images = os.listdir(args.target_folder)
vectors = []
for image_file in all_images:
    image = cv2.imread(os.path.join(args.target_folder, image_file))
    image = totensor(image)
    image = torch.from_numpy(image)
    img = image.float()
    img = torch.unsqueeze(img, 0)
    img = img.cuda()
    _, embedding = model(img)
    vectors.append(torch.flatten(embedding).data.cpu().numpy())
all_embeddings = np.vstack(vectors)
index.add(all_embeddings)
print("indexing ready ..")
print(f"total images to search {index.ntotal}")
print("Enter file name to continue. press -1 to exit.")
image_name = input()
while image_name != -1 and image_name:
    fig = plt.figure(figsize=(8,8))
    image = cv2.imread(image_name)
    fig.add_subplot(R, C, 1)
    plt.imshow(image)
    image = totensor(image)
    image = torch.from_numpy(image)
    img = image.float()
    img = torch.unsqueeze(img, 0)
    img = img.cuda()
    _, embedding = model(img)
    embedding = torch.flatten(embedding).data.cpu().numpy()
    embedding = embedding.reshape(1, -1)
    D, I = index.search(embedding, int(args.k))
    idxes = I.flatten().tolist()
    for i, idx in enumerate(idxes):
        img = os.path.join(args.target_folder, all_images[idx])
        image = cv2.imread(img)
        fig.add_subplot(R, C, i+4)
        plt.imshow(image)
    plt.show()
    plt.pause(0.0001)
    image_name = input("Enter file name to continue. press -1 to exit.\n")







