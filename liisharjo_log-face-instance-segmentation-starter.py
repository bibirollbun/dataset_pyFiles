import os
from PIL import Image
import matplotlib.pyplot as plt
import math

images_folder_path = "/kaggle/input/log-face-train/train/img"

image_files = [f for f in os.listdir(images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif' '.JPEG', '.JPG'))]
image_name = image_files[0]
image_path = os.path.join(images_folder_path, image_name)

img = Image.open(image_path)

plt.imshow(img)
plt.axis('off')
plt.show()


segmentations_folder_path = "/kaggle/input/log-face-train/train/segmentations"
image_segmentations_folder = os.path.join(segmentations_folder_path, image_name.split('.')[0])

mask_files = [f for f in os.listdir(image_segmentations_folder) if f.endswith('.png')]
num_images = len(mask_files)

cols = 5
rows = math.ceil(num_images / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = axes.flatten()

for i, filename in enumerate(mask_files):
    path = os.path.join(image_segmentations_folder, filename)
    mask = Image.open(path)

    axes[i].imshow(mask, cmap="gray")
    axes[i].set_title(filename, fontsize=8)
    axes[i].axis('off')

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


sizes = set()

for img_file in image_files:
    img_path = os.path.join(images_folder_path, img_file)
    try:
        with Image.open(img_path) as img:
            sizes.add(img.size)
    except Exception as e:
        print(f"Error loading {img_file}: {e}")

print(f"Found {len(sizes)} unique sizes:")
for size in sorted(sizes):
    print(f"Width: {size[0]}, Height: {size[1]}")


import cv2
import glob
import numpy as np
import random

preds = []
for imgname in glob.glob(f"/kaggle/input/log-face-test/test/img/*"):
  img = cv2.imread(imgname)
  h, w, _ = img.shape
  this_preds = np.zeros((10, h, w))
  for i in range(10):
    x = random.randint(100, w - 100)
    y = random.randint(100, h - 100)
    size = random.randint(10, 70)
    cv2.circle(this_preds[i], center=(x, y), radius=size, color=255, thickness=-1)
  preds.append((imgname.split("/")[-1], w, h, this_preds))
predicted_mask = preds



import pandas as pd

# Encode numpy into RLE
def rle_encode(mask):
    # Flatten mask into numpy array
    pixels = mask.flatten()
    # Add padding to handle edge cases
    pixels = np.concatenate([[0], pixels, [0]])
    # Find the start and end positions of each run
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Encode each instance mask separately and join them with spaces
csv_rows = []
for imgname, w, h, preds in predicted_mask:
    encoded_masks = []
    for i in range(preds.shape[0]):
        encoded_pixels = rle_encode(preds[i])
        encoded_masks.append(encoded_pixels)
    csv_rows.append((imgname.split(".")[0], w, h, ";".join(encoded_masks)))
# Create a df and save as a CSV file, each instance of an image is separated by semicolon (;)
df = pd.DataFrame(csv_rows, columns=['image_id', 'width', 'height', 'encoded_mask'])
df.to_csv('submission.csv', index=False)


print("Done")


files.upload(); # upload your kaggle.json file


import json

!mkdir /root/.kaggle/
!mv kaggle.json /root/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle config set -n path -v{/content}


!kaggle competitions submit -c log-instance-segmentation -f submission.csv -m "Your submission description"

