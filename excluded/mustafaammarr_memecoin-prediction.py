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


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv("/kaggle/input/extracting-features-from-chunks/train.csv",index_col=0)
y = train.has_graduated
train.columns
features=['consumed_gas_mean','buy_count','buy_percentage','virtual_token_balance_after_mean']
X=train[features]
model = RandomForestRegressor(random_state=1)
model.fit(X,y)
test= pd.read_csv('/kaggle/input/extracting-features-from-chunks/test.csv')
t =test[features]
predictions = model.predict(t)
out = pd.DataFrame({'mint':test.mint,'has_graduated':predictions})
out.to_csv('MemeCoinSubmissions.csv',index=False)




