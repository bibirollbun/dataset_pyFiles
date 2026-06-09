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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error



import zipfile


with zipfile.ZipFile('/kaggle/input/nyc-taxi-trip-duration/test.zip') as z:
    # List all files in the ZIP archive
    print(z.namelist())

    # Read a specific CSV file
    with z.open('test.csv') as f:
        test_df = pd.read_csv(f)

with zipfile.ZipFile('/kaggle/input/nyc-taxi-trip-duration/train.zip') as z:
    # List all files in the ZIP archive
    print(z.namelist())

    # Read a specific CSV file
    with z.open('train.csv') as f:
        train_df = pd.read_csv(f)


test_df.head()


train_df.head()


train_df.isnull().sum()


sns.histplot(train_df['trip_duration'], bins=100, kde=True)
plt.title('Trip Duration Distribution')
plt.xlabel('Trip Duration (seconds)')
plt.ylabel('Frequency')
plt.show()




