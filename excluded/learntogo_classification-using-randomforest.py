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


# 设置随机数种子
seed = 42
np.random.seed(seed)


# 导入对应的包
import pandas as pd
import numpy as np
import seaborn as sns
import random
import torch
import time
import matplotlib.pyplot as plt
%matplotlib inline

import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder


# 载入数据集
model_data = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
submit_test_data = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')


# 这里将所有的特征和目标类型全部分离出来，表示为X和Y
X = model_data.drop(['id','Target'], axis=1).to_numpy()
y = model_data['Target'].to_numpy()


# 这里导入调参需要的所有包
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# 采用调参过程中的最大参数，决策树数量151，不设置最大深度
rfc_most = RandomForestClassifier(n_estimators= 151, oob_score= True, random_state= 42)
rfc_most.fit(X, y)
submit_data = submit_test_data.drop(['id'], axis=1).to_numpy()
y_final_pred = rfc_most.predict(submit_data)
output = pd.read_csv('/kaggle/input/playground-series-s4e6/sample_submission.csv')
output['Target'] = pd.Series(y_final_pred)
output.to_csv('/kaggle/working/submission.csv', index=False)

