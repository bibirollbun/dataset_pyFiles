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


import cv2
import os
import pandas as pd

base_dir = "/kaggle/input/street-food-image-classification/train_images"

train_path = "/kaggle/input/street-food-image-classification/train.csv"
df = pd.read_csv(train_path)



def load_image_cv2(image_id):
    image_path = os.path.join(base_dir, image_id)
    image = cv2.imread(image_path)  
    return image

images = []
labels = []

for _, row in df.iterrows():
    img = load_image_cv2(row['image_id'])
    if img is not None:  
        images.append(img)
        labels.append(row['image_id'].split('/')[0])  



images[1]

