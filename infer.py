import torch
from model import Autoencoder
import os
import torch
from skimage import io, transform
import numpy as np
import cv2
from scipy import spatial
import matplotlib.pyplot as plt




model = Autoencoder().cuda()
model.load_state_dict(torch.load("./trained_model5/trained_weights.h5"))
model.eval()

w = 10
h = 10
fig = plt.figure(figsize=(8, 8))
columns = 3
rows = 3

all_images = os.listdir("testing")
vectors = []
ind = 0
for idx, image_file in enumerate(all_images):
    if image_file == "713.jpg":
        ind = idx
    image = cv2.imread(os.path.join("./testing", image_file))
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = torch.from_numpy(image)
    img = image.float()
    img = torch.unsqueeze(img, 0)
    #img = torch.unsqueeze(img, 0)
    img = img.permute(0, 3, 1, 2)
    img = img.cuda()
    _, embedding = model(img)
    vectors.append(torch.flatten(embedding.squeeze(0)).data.cpu().numpy())
src = vectors[ind]
dists = []
for idx, vec in enumerate(vectors):
    dists.append(np.linalg.norm(src - vec))

sorted_ = np.argsort(dists)
for i, idx in enumerate(sorted_[:9]):
    img = os.path.join("./testing", all_images[idx])
    image = io.imread(img)
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(image)
plt.show()






