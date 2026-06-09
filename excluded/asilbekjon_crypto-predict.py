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
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')


train=pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')


train


train.shape


train.info()


X=train.drop('label',axis=1)
y=train.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
X_train = pd.DataFrame(X_train).replace([np.inf, -np.inf], np.nan).ffill().bfill()
X_test = pd.DataFrame(X_test).replace([np.inf, -np.inf], np.nan).ffill().bfill()

scaler=StandardScaler()
X_train_prepared=scaler.fit_transform(X_train)
X_test_prepared=scaler.transform(X_test)

