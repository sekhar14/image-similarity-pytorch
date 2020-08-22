import faiss
import numpy as np
import torch
from model import Autoencoder
import os
import argparse
import cv2
from custom_transforms import ToTensor
import math
from shutil import copy2

ncentroids = 1024
parser = argparse.ArgumentParser()
parser.add_argument("k")
parser.add_argument("target_folder")
args = parser.parse_args()


model = Autoencoder().cuda()
model.load_state_dict(torch.load("./trained_model8/trained_weights.h5"))
model.eval()
totensor = ToTensor()

## calculate vectors for all images
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
ncentroids = int(args.k)
niter = 20
verbose = True
d = all_embeddings.shape[1]
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(all_embeddings)

## make k no of folders
for i in range(int(args.k)):
    os.mkdir(f"cluster_{i}")

## go through all the vector to find the mean
for i, vector in enumerate(vectors):
    D, I = kmeans.index.search(vector.reshape(1, -1), 1)
    cluster_no = I.flatten().tolist()[0]
    copy2(os.path.join(args.target_folder, all_images[i]),
            f"cluster_{cluster_no}")







