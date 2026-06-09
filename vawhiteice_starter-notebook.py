import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from glob import glob

# For display
from IPython.display import display


# Path to the image dataset
DATASET_PATH = "/kaggle/input/vehicle-type-classification-challenge/train/train"

# Mapping folder names to labels
labels = ['Bus', 'Suv', 'Sedan', 'Other']

# Load all images with paths and labels
image_paths = []
image_labels = []

for label in labels:
    files = glob(os.path.join(DATASET_PATH, label, "*.jpg"))  # or .png based on your file type
    image_paths.extend(files)
    image_labels.extend([label] * len(files))

df = pd.DataFrame({
    "image_path": image_paths,
    "label": image_labels
})

df.head()


print(df.shape)
print(df.head())


!ls /kaggle/input/vehicle-type-classification-challenge/


plt.figure(figsize=(6,4))
sns.countplot(x="label", data=df)
plt.title("Class Distribution")
plt.show()


def show_images(label, n=5):
    sample_paths = df[df.label == label].sample(n).image_path.tolist()
    plt.figure(figsize=(15,3))
    for i, img_path in enumerate(sample_paths):
        img = Image.open(img_path)
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(label)
    plt.tight_layout()
    plt.show()

for lbl in labels:
    show_images(lbl)


TEST_PATH = "/kaggle/input/vehicle-type-classification-challenge/test/test"
test_images = sorted(glob(os.path.join(TEST_PATH, "*.jpg")))

submission = pd.DataFrame({
    "image_id": [os.path.basename(path) for path in test_images],
    "label": [random.choice(labels) for _ in test_images]
})

submission.head()


# Save the file
submission.to_csv("random_submission.csv", index=False)
print(" Submission file created: random_submission.csv")




