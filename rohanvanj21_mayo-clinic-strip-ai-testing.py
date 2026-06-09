# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df = pd.read_csv("/kaggle/input/mayo-clinic-strip-ai/train.csv")
print(df.head())


print("Num rows:", len(df))
print("Unique patients:", df['patient_id'].nunique())
print("Images per patient (avg):", df.groupby('patient_id').size().mean())


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x='label')
plt.title('Label Distribution (CE vs LAA)')


# Count occurrences of each label
label_counts = df['label'].value_counts()
print("Label counts:")
print(label_counts)


center_counts = df['center_id'].value_counts()
print("Sample count per center:")
print(center_counts)


pd.crosstab(df['center_id'], df['label'], normalize='index') * 100


from PIL import Image
import matplotlib.pyplot as plt

# Path to your image file (example)
img_path = '/kaggle/input/mayo-clinic-strip-ai/train/008e5c_0.tif'

# Load image with PIL
img = Image.open(img_path)

# Display image with matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')  # Hide axis for cleaner view
plt.title('Sample CE Visualization')
plt.show()


img_path = '/kaggle/input/mayo-clinic-strip-ai/train/6baf51_0.tif'
Image.MAX_IMAGE_PIXELS = None

# Load image with PIL
img = Image.open(img_path)

# Display image with matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')  # Hide axis for cleaner view
plt.title('Sample LAA Visualization')
plt.show()


def get_image_size(image_id):
    path = f'/kaggle/input/mayo-clinic-strip-ai/train/{image_id}.tif'
    with Image.open(path) as img:
        return img.size  # returns (width, height)

# Apply to all images (warning: slow for big datasets)
df['image_size'] = df['image_id'].apply(get_image_size)

# Split width and height into separate columns
df['width'] = df['image_size'].apply(lambda x: x[0])
df['height'] = df['image_size'].apply(lambda x: x[1])

# Plot distributions
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
df['width'].hist(bins=30)
plt.title('Image Width Distribution')
plt.xlabel('Width (pixels)')
plt.ylabel('Count')

plt.subplot(1,2,2)
df['height'].hist(bins=30)
plt.title('Image Height Distribution')
plt.xlabel('Height (pixels)')
plt.ylabel('Count')
plt.show()


df['area'] = df['width'] * df['height']

# Find smallest and largest by area
smallest = df.loc[df['area'].idxmin()]
largest = df.loc[df['area'].idxmax()]

print("Smallest image:")
print(f"Image ID: {smallest['image_id']}")
print(f"Width: {smallest['width']} px, Height: {smallest['height']} px")
print(f"Area: {smallest['area']} pixels² ({smallest['area'] / 1_000_000:.2f} MP)")

print("\nLargest image:")
print(f"Image ID: {largest['image_id']}")
print(f"Width: {largest['width']} px, Height: {largest['height']} px")
print(f"Area: {largest['area']} pixels² ({largest['area'] / 1_000_000:.2f} MP)")


pad smaller images or 

