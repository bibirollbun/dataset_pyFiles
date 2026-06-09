# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!unzip -q "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
!unzip -q "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"


from sklearn.model_selection import train_test_split

def prepare_data(train_path, val_size=0.2, random_state=42):
    train_filenames = os.listdir(train_path)
    train_categories = ['dog' if filename.split(".")[0] == 'dog' else 'cat' for filename in train_filenames]

    df = pd.DataFrame({
        'filename': train_filenames,
        'category': train_categories
    })

    train_df, val_df = train_test_split(df, test_size=val_size, stratify=df["category"], random_state=random_state)
    return train_df, val_df


train_path = "/kaggle/working/train"
train_df, val_df = prepare_data(train_path)
print(f"Total Training Images: {len(train_df)}")
print(f"Total Validation Images: {len(val_df)}")


train_df.head()

