import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pydicom
from random import choice


data_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection'
print(os.listdir(data_dir))


# load the csv files
train_df = pd.read_csv(os.path.join(data_dir,'train.csv'))
localizers_df = pd.read_csv(os.path.join(data_dir,'train_localizers.csv'))


# pick a random series where an aneurysm is present
positive_cases = train_df[train_df['Aneurysm Present'] == 1]
example = choice(list(positive_cases.SeriesInstanceUID.unique()))
# this example was shown to be wrong in another thread
example = '1.2.826.0.1.3680043.8.498.10843288560910004558081082597234683103'
print(example)


# find corresponding location
example_location = localizers_df[localizers_df.SeriesInstanceUID == example].sample(1)
example_location.head()


# extract image file and coordinates
sop_instance = example_location.SOPInstanceUID.item()
coordinates = eval(example_location.coordinates.item())
print(coordinates)


# load the dataset
image_file = os.path.join(data_dir,'series',example,sop_instance+'.dcm')
dataset = pydicom.dcmread(image_file)
pixels = dataset.pixel_array
if 'f' in coordinates.keys():
    # multiframe dicom
    pixels = pixels[coordinates['f']]


# plot the slice and highlight the marked location
ax = plt.subplot(1,1,1)
plt.pcolormesh(pixels,cmap = 'bone')
highlight_circle = plt.Circle((coordinates['x'], coordinates['y']), radius=20, color='red', fill=False, linewidth=1)
ax.add_patch(highlight_circle)
plt.axis('square')
_ = plt.axis('off')




