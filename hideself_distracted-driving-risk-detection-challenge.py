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
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier


train_data=pd.read_csv('/kaggle/input/distracted-driving-risk-detection-challenge/kaggle_train.csv')
train_data.head()


test_data=pd.read_csv('/kaggle/input/distracted-driving-risk-detection-challenge/kaggle_test.csv')
test_data.head()


# label_source这一列排除
train_data=train_data.drop('label_source',axis=1)
test_data=test_data.drop('label_source',axis=1)

features=['observation_hour', 'speed', 'rpm', 'acceleration', 'throttle_position',
       'engine_temperature', 'engine_load_value', 'heart_rate',
       'current_weather', 'visibility', 'precipitation', 'accidents_onsite',
       'design_speed', 'accidents_time']
X=train_data[features]
y=train_data['risk_level']

# 重新编码标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# 定义基学习器
estimators = [
    ('xgbc', XGBClassifier(n_estimators=350,max_depth=3,learning_rate=0.2,colsample_bytree=0.8,subsample=0.85,objective='multi:softmax',random_state=42)),
    ('lgbm', LGBMClassifier(n_estimators=200,learning_rate=0.2,max_depth=3,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=42,verbose=-1)),
    ('rf', RandomForestClassifier(n_estimators=200,max_depth=None,min_samples_leaf=1,min_samples_split=2,bootstrap=False,random_state=42)),
    ('catc',CatBoostClassifier(iterations=500,depth=8,learning_rate=0.1,l2_leaf_reg=3,random_strength=0.1,random_state=42,verbose=False)),
]

# final_estimator=final_estimator =  KNeighborsClassifier()


# 软投票：基于概率加权平均
voting_clf_soft = VotingClassifier(
    estimators=estimators,
    voting='soft',
)



# # 测试
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)

# stacking_clf.fit(X_train,y_train)

# y_pred = stacking_clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"准确率: {accuracy:.4f}")

# 提交
voting_clf_soft.fit(X, y)

y_pred=voting_clf_soft.predict(test_data)

y_pred = label_encoder.inverse_transform(y_pred)

output=pd.DataFrame({'id':test_data.index,'risk_level':y_pred})
output.to_csv('/kaggle/working/submission.csv', index=False)
print('Your submission was successfully saved!')

