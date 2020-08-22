import torch
from model import Autoencoder
import os
import torch
from skimage import io, transform
import numpy as np
import cv2
from scipy import spatial
import matplotlib.pyplot as plt
from custom_transforms import Blurr, ReSize, ToTensor, Sharpen
from scipy.spatial import distance


model = Autoencoder().cuda()
model.load_state_dict(torch.load("./trained_model7/trained_weights.h5"))
model.eval()

blurr = Blurr()
resize = ReSize(scale_factor=1.0)
totensor = ToTensor()
sharpen = Sharpen()

w = 10
h = 10
fig = plt.figure(figsize=(8, 8))
columns = 3
rows = 4

all_images = os.listdir("testing")
vectors = []
ind = 5
for idx, image_file in enumerate(all_images):
    if image_file == "163.jpg":
        ind = idx
    image = cv2.imread(os.path.join("./testing", image_file))
    for transform in [sharpen, resize, totensor]:
        image = transform(image)
    image = torch.from_numpy(image)
    img = image.float()
    img = torch.unsqueeze(img, 0)
    img = img.cuda()
    _, embedding = model(img)
    vectors.append(torch.flatten(embedding.squeeze(0)).data.cpu().numpy())
src = vectors[ind]
dists = []
for idx, vec in enumerate(vectors):
    dists.append(abs(distance.euclidean(src, vec)))

sorted_ = np.argsort(dists)
for s in sorted_[:5]:
    print(dists[s])
for i, idx in enumerate(sorted_[:10]):
    img = os.path.join("./testing", all_images[idx])
    image = io.imread(img)
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(image)
plt.show()






