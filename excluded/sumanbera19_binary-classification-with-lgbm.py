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


import numpy as numpy
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


df=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")


df


df.isna().sum()


df.info()


test=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


test


test.info()


categorical_cols = df.select_dtypes(include=['object']).columns


categorical_cols 


df=df.sample(n=50000,random_state=42)


for feature in categorical_cols:
    df[feature] = df[feature].astype("category")
    test[feature]  = test[feature] .astype("category")



x=df.drop(columns=['y'])
y=df['y']


from lightgbm import LGBMClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Suppose categorical columns have been converted to category dtype
categorical_cols = list(x.select_dtypes(include='category').columns)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lgbm = LGBMClassifier(
    max_depth=4,
    n_estimators=15000,
    learning_rate=0.06,
    reg_alpha=0.8,
    reg_lambda=3.0,
    colsample_bytree=0.5,
    subsample=0.8,
    categorical_feature=categorical_cols,  # list of column names
    seed=42,
    verbosity=-1
)

lgbm.fit(X_train, y_train)
y_lgbm = lgbm.predict(X_test)

print("ACCURACY OF LGBM IS:", accuracy_score(y_test, y_lgbm))



submission=pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")




test.info()


test.info()


# for col in categorical_cols:
#     test[col] = test[col].astype('category')
#     test[col] = test[col].cat.set_categories(x[col].cat.categories)

submission['y'] = lgbm.predict(test)
submission.to_csv('submission.csv', index=False)
print(submission.head())



submission 




