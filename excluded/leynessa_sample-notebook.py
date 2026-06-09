# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import json
import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

# Load annotation JSON
json_path = "/kaggle/input/garbage-segmentation/garbage_segmentation_dataset/train_json.json"
image_root = "/kaggle/input/garbage-segmentation/garbage_segmentation_dataset/train"

with open(json_path, 'r') as f:
    coco = json.load(f)

# Create image dict for quick lookup
image_dict = {img['id']: img for img in coco['images']}

# Group annotations by image
from collections import defaultdict
ann_by_img = defaultdict(list)
for ann in coco['annotations']:
    ann_by_img[ann['image_id']].append(ann)

# Visualize 10 images
fig, axs = plt.subplots(5, 2, figsize=(20, 25))
axs = axs.flatten()
count = 0

for img_id, anns in ann_by_img.items():
    if count >= 10:
        break
    img_info = image_dict[img_id]
    img_path = os.path.join(image_root, img_info['file_name']).replace('\\', '/')

    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    axs[count].imshow(image)
    axs[count].set_title(img_info.get('file_name', ''))

    patches = []
    for ann in anns:
        if 'segmentation' in ann:
            for seg in ann['segmentation']:
                poly = Polygon(np.array(seg).reshape(-1, 2), closed=True, edgecolor='red', fill=False, linewidth=2)
                axs[count].add_patch(poly)

    axs[count].axis('off')
    count += 1

plt.tight_layout()
plt.show()



# Create DataFrame for the submission (image_id, pred_str)
submission_df = pd.DataFrame({
    "id": test_image_ids,  # Add an ID column here
    "image_id": test_image_ids,  # Repeat for consistency
    "pred_str": predictions
})

# Save the final submission to CSV
submission_df.to_csv("submission.csv", index=False)

