# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

### Sample code for visualisation
%matplotlib inline

labels = ['peach', 'nectarine', 'orange', 'mandarin', 'lingonberry', 'cranberry', 'plum', 'apricot']

def draw_rectangle(ax, top_left_x, top_left_y, width, height, color='red'):
    top_left = (top_left_x, top_left_y)
    bottom_right = (top_left_x + width, top_left_y + height)

    # Plotting each edge of the rectangle on the specified axes
    ax.plot([top_left[0], top_left[0]], [top_left[1], bottom_right[1]], '-', color=color)  # Left edge
    ax.plot([bottom_right[0], bottom_right[0]], [top_left[1], bottom_right[1]], '-', color=color)  # Right edge
    ax.plot([top_left[0], bottom_right[0]], [top_left[1], top_left[1]], '-', color=color)  # Top edge
    ax.plot([top_left[0], bottom_right[0]], [bottom_right[1], bottom_right[1]], '-', color=color)  # Bottom edge

#Reading in image filenames
train_dirname='/kaggle/input/team-16-codebreakers/data/train/images/'
test_dirname = '/kaggle/input/team-16-codebreakers/data/test/images/'
train_filenames=[]
test_filenames = []

for _,_,filenames in os.walk(train_dirname):
    for filename in filenames:
        train_filenames.append((train_dirname+filename,filename))
for _,_,filenames in os.walk(test_dirname):
    for filename in filenames:
        test_filenames.append((test_dirname+filename,filename))

#Image visualisation with bounding box
idx=-1
img=Image.open(train_filenames[idx][0])
train_csv=pd.read_csv('/kaggle/input/team-16-codebreakers/train.csv')
row=train_csv.loc[train_csv['filename']==train_filenames[idx][1]].iloc[0]
print("Image file",train_filenames[idx])
print("Corresponding info",row)

top_left_x = int(row['xmin'])
top_left_y = int(row['ymin'])
width = int(row['xmax']) - int(row['xmin'])+1
height = int(row['ymax']) - int(row['ymin'])+1

# Plot the image and the bbox
fig, ax = plt.subplots()
ax.imshow(img)
draw_rectangle(ax, top_left_x, top_left_y, width, height)
plt.title(labels[row['class']])
plt.show()

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


##Sample code for making a submission
def predict(filename):
    img=Image.open(filename)
    xmin,ymin = 0,0
    xmax, ymax=img.width, img.height
    preditedClass = 0
    return xmin, ymin, xmax, ymax, preditedClass

predictions = []
for long_filename, short_filename in test_filenames:
    xmin, ymin, xmax, ymax, preditedClass = predict(long_filename)
    predictions.append([short_filename, xmin, ymin, xmax, ymax, preditedClass])

filenames, xmins, ymins, xmaxs, ymaxs, preditedClasses = zip(*predictions)

prediction_df = pd.DataFrame({
    'filename': filenames,
    'xmin': xmins,
    'ymin': ymins,
    'xmax': xmaxs,
    'ymax': ymaxs,
    'class': preditedClasses
})
prediction_df.to_csv('submission.csv', index=False)

