import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
import gc
import pickle


train = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
train.tail()


train.groupby('Modality').count()


len(train[train['Aneurysm Present'] > 0]['SeriesInstanceUID'].unique())


train_loc = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')
train_loc.tail()


len(train_loc.SeriesInstanceUID.unique())


path_series = '/kaggle/input/rsna-intracranial-aneurysm-detection/series/'


import ast


sample = train_loc.iloc[np.random.randint(len(train_loc))]
dcm = pydicom.dcmread(path_series+sample.SeriesInstanceUID+'/'+sample.SOPInstanceUID+'.dcm')
#ps = dcm.PixelSpacing
xy = ast.literal_eval(sample.coordinates)
x = np.rint(xy['x']).astype(int)
y = np.rint(xy['y']).astype(int)
z = dcm.InstanceNumber
loc = sample.location
print(z,y,x)
plt.imshow(dcm.pixel_array)
plt.title(loc)


all_slices = [pydicom.dcmread(path_series+sample.SeriesInstanceUID+'/'+dcm) for dcm in os.listdir(path_series+sample.SeriesInstanceUID)]
all_slice_IN = [int(dcm.InstanceNumber) for dcm in all_slices]
all_slices = [dcm.pixel_array for dcm in all_slices]


all_slices = [x for _, x in sorted(zip(all_slice_IN, all_slices))]


image = np.stack(all_slices)
image.shape


z,y,x


print(y,x)
plt.imshow(image[z])


print(z,x)
plt.imshow(image[:,y])


print(z,y)
plt.imshow(image[:,:,x])


path = path_series+'1.2.826.0.1.3680043.8.498.99892390884723813599532075083872271516/'
IN = []
SL = []
for s in os.listdir(path):
    image = pydicom.dcmread(path+s)
    IN.append(image.InstanceNumber)
    SL.append(image.SliceLocation)
#   image.PixelSpacing)
#   plt.imshow(image.pixel_array)
#   plt.show()

plt.plot(IN,SL,'.')

