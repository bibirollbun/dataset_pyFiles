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


sub_dir = "/kaggle/input/playground-series-s5e4/sample_submission.csv"
train_dir = "/kaggle/input/playground-series-s5e4/train.csv"
test_dir = "/kaggle/input/playground-series-s5e4/test.csv"


sub_df = pd.read_csv(sub_dir)
train_df = pd.read_csv(train_dir)
test_df = pd.read_csv(test_dir)


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb


train_df.info()


x_df = train_df.drop(columns=["Listening_Time_minutes"])
y_df = train_df["Listening_Time_minutes"]


categoriacl_cols = ["Podcast_Name","Episode_Title","Genre","Publication_Day","Publication_Time","Episode_Sentiment"]
for col in categoriacl_cols:
    x_df[col] = x_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')


x_train,x_cv,y_train,y_cv = train_test_split(x_df,y_df,test_size = 0.2,random_state=42)


y_train.head()


train_data = lgb.Dataset(x_train, label = y_train,categorical_feature = ["Podcast_Name","Episode_Title","Genre","Publication_Day","Publication_Time","Episode_Sentiment"])
test_data = lgb.Dataset(x_cv, label = y_cv, categorical_feature = ["Podcast_Name","Episode_Title","Genre","Publication_Day","Publication_Time","Episode_Sentiment"],reference = train_data)


params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate':0.1,
    'verbose' : -1
}


num_round = 100


watchlist=[(x_train,y_train),(x_cv,y_cv)]
model = lgb.train(params = params,
                  train_set = train_data,
                  num_boost_round = num_round,
                  valid_sets = [test_data])


y_pred = model.predict(x_cv, num_iteration=model.best_iteration)


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_cv,y_pred))
print(rmse)


y_pred_test = model.predict(test_df,num_iteration = model.best_iteration)


sub_df['Listening_Time_minutes']=y_pred_test
sub_df.to_csv('submission.csv',index=False)

