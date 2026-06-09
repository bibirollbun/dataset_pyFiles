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
import os 
import time
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


pip install xgboost == 1.7.6


submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train.head()


test.head()


submission.head()


train.info()


test.info()


train.describe()


le = LabelEncoder()

train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.fit_transform(test['Sex'])


train


test


plt.figure(figsize = (10, 6))
sns.histplot(train['Age'], bins=40, kde=False, color='skyblue')

plt.title('Age Distribution')
plt.show()


feature_list = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

poly = PolynomialFeatures(degree = 2, include_bias = False)

x_poly_train = poly.fit_transform(train[feature_list])
x_poly_test = poly.fit_transform(test[feature_list])

features = poly.get_feature_names_out(feature_list)

train_v2 = pd.DataFrame(x_poly_train, columns = features)
test_v2 = pd.DataFrame(x_poly_test, columns = features)

train_v2['Calories'] = train['Calories']


train_v2['BMI'] = (train_v2['Weight'] / ((train_v2['Height'] / 100) ** 2)).round(0)
test_v2['BMI'] = (test_v2['Weight'] / ((test_v2['Height'] / 100) ** 2)).round(0)


train_v2


train_v2.info()


test_v2.info()


train_v2['sex'] = train['Sex']
test_v2['sex'] = test['Sex']

sample = train_v2.sample(n= 10000, random_state=42)

x = sample.drop('Calories', axis = 1)
y = sample['Calories']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 42)

rf = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)

rf.fit(x_train, y_train)
importances = rf.feature_importances_
feat_labels = x.columns

indices = np.argsort(rf.feature_importances_)[::-1]

for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


x = train_v2.drop('Calories', axis = 1)
y = train_v2['Calories']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = XGBRegressor(
    max_depth = 8,
    colsample_bytree = 0.75,
    subsample=0.9,
    n_estimators=1000,
    learning_rate=0.04,
    gamma=0.01, 
    early_stopping_rounds=100,
    eval_metric="rmse",
)

model.fit(
    x_train, y_train,
    eval_set = [(x_val, y_val)],
    verbose = 100
)

oof_pred = model.predict(x_val)
test_pred = model.predict(test_v2)

rmse = np.sqrt(mean_squared_error(y_val, oof_pred))
print(f"\nValidation RMSE: {rmse:.4f}")


print(test_pred.shape)
print(submission.shape)


submission


test_pred


test_pred = np.clip(test_pred,1,314)
submission['Calories'] = test_pred
submission.to_csv("submission.csv", index=False)
submission.head()




