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


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train.info()


train.isnull().sum()


col = train.columns


for i in col:
    print(train[i].value_counts())
    print("-"*30)


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler


num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy = 'median')),
    ('scaler',StandardScaler())
])

cat_pipeline = Pipeline([
    ('ordinal_encoder',OrdinalEncoder()),
    ('imputer',SimpleImputer(strategy = 'most_frequent')),
    ('cat_encoder',OneHotEncoder(sparse_output = False))
])


from sklearn.compose import ColumnTransformer


train.columns


train.info()



for col in train.columns:
    if train[col].dtype == bool:
        train[col] = train[col].astype(int)
        test[col] = test[col].astype(int)
        


def separateColumn(df,include_bool = False):
    if include_bool:
        num_cols = df.select_dtypes(include = ['int64','float64','bool']).columns.tolist()
        cat_cols = df.select_dtypes(exclude = ['int64','float64','bool']).columns.tolist()
    else:
        num_cols = df.select_dtypes(include = ['int64','float64']).columns.tolist()
        cat_cols = df.select_dtypes(exclude = ['int64','float64']).columns.tolist()
    return num_cols,cat_cols
        
        


num_attrib,cat_attrib = separateColumn(train)
del num_attrib[-1]
preprocessPipeline = ColumnTransformer([
    ('num',num_pipeline,num_attrib),
    ('cat',cat_pipeline,cat_attrib)
])


train.columns.tolist()


X_train = preprocessPipeline.fit_transform(train.drop('accident_risk', axis=1))


y_train = train['accident_risk']


from lightgbm import LGBMRegressor

from sklearn.model_selection import RandomizedSearchCV


model = LGBMRegressor(
    n_estimators=10000,       #
    learning_rate=0.01,       #
    boosting_type='gbdt',
    random_state=42
)



model.fit(
    X_train, y_train
)


X_test = preprocessPipeline.fit_transform(test)


y_pred2 = model.predict(X_test)


submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': y_pred2.flatten() 
})
submission.to_csv('submission.csv', index=False)

