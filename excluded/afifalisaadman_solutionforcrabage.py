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


from catboost import CatBoostRegressor


train_db = pd.read_csv("/kaggle/input/crab-age-no-2/train.csv")
train_db.head(10)


from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()

train_db['Sex'] = lbl.fit_transform(train_db.Sex)


train_db.head()


train_db = train_db.drop(columns=["id"])


train_db.head()


import seaborn as sns

sns.heatmap(train_db.corr(),annot=True,cmap='coolwarm')


from sklearn.model_selection import train_test_split as tts

X = train_db.drop(columns=['Age'])

X_Train,X_Test,Y_Train,Y_Test = tts(X,train_db.Age,test_size=0.30)


import lightgbm as lgb
lgr = lgb.LGBMRegressor(
    max_depth=11,
    random_state=44,
    n_estimators=10000,
    learning_rate=0.01,
    force_col_wise=True,
    verbose=-1,
    subsample=0.8,
    eval_metric='logloss',
    use_lable_encoder=False
)
lgr.fit(X_Train,Y_Train)


from sklearn.metrics import mean_absolute_error as mae

mae(Y_Test,lgr.predict(X_Test))


test = pd.read_csv("/kaggle/input/crab-age-no-2/test.csv")

test_x = test.drop(columns=['id'])


test_x['Sex'] = lbl.transform(test_x.Sex)


submit = pd.DataFrame({
    'id':test.id,
    'Age':lgr.predict(test_x)
})





submit.head()


submit.to_csv("submission.csv",index=False)

