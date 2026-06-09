import json
import matplotlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import csv
import random


import cv2
import torch
from math import floor, ceil
from torch.utils.data import Dataset, DataLoader, random_split


def draw_rectangle(ax, top_left_x, top_left_y, width, height, color='red'):
    top_left = (top_left_x, top_left_y)
    bottom_right = (top_left_x + width, top_left_y + height)

    # Plotting each edge of the rectangle on the specified axes
    ax.plot([top_left[0], top_left[0]], [top_left[1], bottom_right[1]], '-', color=color)           # Left edge
    ax.plot([bottom_right[0], bottom_right[0]], [top_left[1], bottom_right[1]], '-', color=color)   # Right edge
    ax.plot([top_left[0], bottom_right[0]], [top_left[1], top_left[1]], '-', color=color)           # Top edge
    ax.plot([top_left[0], bottom_right[0]], [bottom_right[1], bottom_right[1]], '-', color=color)   # Bottom edge


class ObjectDetectionDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        with open(f'{self.dataset_dir}/annotations.json', 'r') as file:
            self.data = json.load(file)
            self.indicies = list(self.data.keys())

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        idx = self.indicies[idx]
        annotes = [annt + annt[:1] for annt in self.data[f"{idx}"]]
        img = Image.open(f'{self.dataset_dir}/frames/frame_{idx}.jpg')
        boxes = [self.get_bbox(annt) for annt in annotes]

        return img, boxes

    def get_bbox(self, polygon):
        x_min = min([point[0] for point in polygon])
        x_max = max([point[0] for point in polygon])
        y_min = min([point[1] for point in polygon])
        y_max = max([point[1] for point in polygon])

        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        width = floor(x_max - x_min)
        height = floor(y_max - y_min)

        return cx, cy, width, height


data_folder = '/kaggle/input/image-innovators/dataset'
train_folder = os.path.join(data_folder, "train")
dataset = ObjectDetectionDataset(train_folder)


img, boxes = dataset[np.random.randint(len(dataset))]

plt.figure(figsize=(16, 16))
plt.subplot(121)
plt.imshow(img)

for box in boxes:
    cx, cy, width, height = box
    draw_rectangle(plt.gca(), cx - width // 2, cy - height // 2, width, height)

plt.show()


#This Generates a random submission table in the correct format
#The number of images is correct, but the values in cells do not necessarily have to fall within
#the generated range. 
num_images = 433  # from 0 to 432

output_file = "submission.csv"

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=",")

    # Write the header
    writer.writerow(["image_id", "x_min", "y_min", "x_max", "y_max", "class_id"])

    # Write the data rows
    for image_id in range(num_images):
        num = random.randint(1, 2) # number of prediction for a single image
        x_min = ';'.join([str(random.randint(1, 1000)) for _ in range(num)])
        y_min = ';'.join([str(random.randint(1, 1000)) for _ in range(num)])
        x_max = ';'.join([str(random.randint(1, 1000)) for _ in range(num)])
        y_max = ';'.join([str(random.randint(1, 1000)) for _ in range(num)])
        class_id = 0
        writer.writerow([image_id, x_min, y_min, x_max, y_max, class_id])



!head submission.csv




