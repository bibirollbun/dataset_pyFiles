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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')

train_df.head()


y = train_df['accident_risk']
X = train_df.drop("accident_risk", axis=1)


X.info()


cat_train = X.select_dtypes(include=['object']).copy()  #MSSubClass is nominal
cat_train.columns


numeric_ = X.select_dtypes(exclude=['object']).copy()
numeric_.columns


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")

plt.figure(figsize=(10,6))
plt.title("Before transformation of SalePrice")
dist = sns.distplot(np.log1p(y),norm_hist=False)

y = np.log1p(y)


from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

CatFeat = ['road_type', 'lighting', 'weather', 'road_signs_present',
           'public_road', 'time_of_day', 'holiday', 'school_season']


encoder = OrdinalEncoder()
X[CatFeat] = encoder.fit_transform(X[CatFeat])
test_df[CatFeat] = encoder.transform(test_df[CatFeat])

X = X.drop('school_season', axis=1)
test_df = test_df.drop('school_season', axis=1)



X.describe()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=10000,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    random_state=0,
                    colsample_bytree=0.9,
                    min_child_weight=3,         
                    reg_alpha=0.1,              
                    reg_lambda=1.5,             
                    gamma=0.0)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    early_stopping_rounds=200,
    verbose=False
)


from sklearn.metrics import mean_squared_error

predict = np.expm1(model.predict(X_test))


mse = mean_squared_error(predict, y_test)

print(np.sqrt(mse))


out_predict = np.expm1(model.predict(test_df))


output = pd.DataFrame({'id' : test_df.index,
                      'accident_risk' : out_predict})
output.to_csv('Submission.csv', index=False)

