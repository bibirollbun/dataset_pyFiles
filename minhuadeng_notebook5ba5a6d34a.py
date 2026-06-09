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


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


train_df=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv') 
test_df=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train_df.columns


train_df.isnull().sum()


train_df['Episode_Length_minutes']=train_df['Episode_Length_minutes'].fillna('median')
train_df['Guest_Popularity_percentage']=train_df['Guest_Popularity_percentage'].fillna('median')
train_df['Number_of_Ads']=train_df['Number_of_Ads'].fillna('median')

test_df['Episode_Length_minutes']=test_df['Episode_Length_minutes'].fillna('median')
test_df['Guest_Popularity_percentage']=test_df['Guest_Popularity_percentage'].fillna('median')
test_df['Number_of_Ads']=test_df['Number_of_Ads'].fillna('median')




train_df.isnull().sum()


test_df.isnull().sum()




