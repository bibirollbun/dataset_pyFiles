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
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import optuna

import warnings
warnings.filterwarnings('ignore')




df_train= pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").set_index('id')
df_test= pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')
df_subm= pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


plt.figure(figsize=(12, 10))
sns.heatmap(data=df_train.corr(), annot=True, linewidths=0.2);


import lightgbm as lgb
from sklearn.model_selection import train_test_split


#特定の属性だけで実施する。(cloud/sunshine）
df_train= pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").set_index('id')
df_test= pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')

X_target = df_train[(df_train['cloud'] > 77 )]

X_target= df_train[['cloud']]

y = df_train[['rainfall']]

X_asses = df_test[['cloud']]

print(X_target)




# グラフの作成
plt.scatter(X_target, y)

# グラフの装飾
plt.title("cloud")
plt.xlabel("X-cloud")
plt.ylabel("Y-rain")
           
# グラフの表示
plt.show()
# 箱ひげ図を作成
plt.boxplot(X_target)

# グラフを表示
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X_target, y, test_size=0.25, random_state=0)


import lightgbm as lgb

# LightGBMの学習
model = lgb.LGBMRegressor(objective= 'binary', metric='auc', random_state=15,
                         learning_rate=0.01)
model.fit(X_train, y_train)


# 予測と評価
y_pred = model.predict(X_asses)


# Save Submission

df_subm['rainfall'] = y_pred
df_subm.to_csv('submission.csv', index=False)
df_subm.head()

