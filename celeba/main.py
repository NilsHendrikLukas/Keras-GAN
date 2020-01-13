# Extract the downloaded zipfile
import zipfile

with zipfile.ZipFile("celeba.zip","r") as zip_ref:
  zip_ref.extractall("data_faces/")

# Obtain the image list
import os
root = 'data_faces/img_align_celeba'
img_list = os.listdir(root)
print("Found {} images! ".format(len(img_list)))

# Group the images to the ID into a dictionary
id_idx = {}
with open("celeba_ids.txt", 'r') as celeb_ids:
  for i, x in enumerate(celeb_ids):
    fname, key = x.split()
    id_idx.setdefault(key, []).append(fname)

# Sort with number of contributed images per individual and plot amount
import numpy as np
import matplotlib.pyplot as plt

sorted_idx = sorted(id_idx.items(), key=lambda kv: len(kv[1]), reverse=True)
x, y = np.arange(len(id_idx.items())), [len(x[1]) for x in sorted_idx]

# Make cumulative
for i in range(1, len(y)):
    y[i] += y[i - 1]

# Plot the graph
plt.plot(x, y)
plt.title("CelebA Number of Images per Individual")
plt.xlabel("Number of Individuals")
plt.ylabel("Total Images")
plt.savefig("imagesperindividual.svg")
plt.show()