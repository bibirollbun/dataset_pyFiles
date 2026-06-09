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


# # !pip install -q kaggle
# import os
# import sys

# def is_kaggle():
#     return 'KAGGLE_URL_BASE' in os.environ

# if not is_kaggle():
#     !mkdir -p ~/.kaggle
#     !cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
#     !chmod 600 ~/.kaggle/kaggle.json

# # Download dataset
# competition_name = 'playground-series-s5e4'
# if not is_kaggle():
#     !kaggle competitions download -c {competition_name}
#     !unzip -q {competition_name}.zip -d {competition_name}
#     data_path = f'./{competition_name}/'
# else:
#     # In Kaggle, data is available at this path:
#     data_path = '../input/'


# print("Available files:")
# !ls -la {data_path}


# import pandas as pd

# # # train_df = pd.read_csv(f"{data_path}/train.csv")
# # # test_df = pd.read_csv(f"{data_path}/test.csv")


!pip install optuna
!pip install catboost


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from catboost import CatBoostRegressor, Pool
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")


train_df = pd.read_csv(f"/kaggle/input/playground-series-s5e4/train.csv",index_col='id')
test_df = pd.read_csv(f"/kaggle/input/playground-series-s5e4/test.csv",index_col='id')


print(train_df.shape)
train_df.info()
display(train_df.describe())
train_df.head()


num_cols = test_df.select_dtypes(include=np.number).columns
cat_cols = train_df.select_dtypes(exclude=np.number).columns


print(train_df.isnull().sum())
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')
train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])

test_df[num_cols] = num_imputer.transform(test_df[num_cols])
test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])
print(train_df.isnull().sum())
print(test_df.isnull().sum())


train_df['total_popularity_percentge'] =train_df['Host_Popularity_percentage'] +train_df['Guest_Popularity_percentage']
train_df['popularity_ratio'] = train_df['Host_Popularity_percentage'] /train_df['Guest_Popularity_percentage']

test_df['total_popularity_percentge'] =test_df['Host_Popularity_percentage'] +test_df['Guest_Popularity_percentage']
test_df['popularity_ratio'] = test_df['Host_Popularity_percentage'] /test_df['Guest_Popularity_percentage']


train_df['Number_of_Ads'] = train_df['Number_of_Ads'].apply(lambda x: 20 if x > 20 else x)
test_df['Number_of_Ads'] = test_df['Number_of_Ads'].apply(lambda x: 20 if x > 20 else x)

train_df['Ad_Density'] = train_df['Number_of_Ads'] / train_df['Episode_Length_minutes']
test_df['Ad_Density'] = test_df['Number_of_Ads'] / test_df['Episode_Length_minutes']


# binning
train_df['Episode_Length_Bins'] = pd.cut(train_df['Episode_Length_minutes'],bins=[-1,15,30,60,120,float('inf')], labels=[1,2,3,4,5])


test_df['Episode_Length_Bins'] = pd.cut(test_df['Episode_Length_minutes'],bins=[-1,15,30,60,120,float('inf')], labels=[1,2,3,4,5])


drop_zero_len = train_df[train_df['Episode_Length_minutes'] == 0].index
train_df.drop(drop_zero_len,inplace=True)


# cyclic feature
train_df['Episode_Length_Sin'] = np.sin(2 * np.pi *
                                        train_df['Episode_Length_minutes']/60)
train_df['Episode_Length_Cos'] = np.cos(2 * np.pi *
                                        train_df['Episode_Length_minutes']/60)
train_df['Episode_Length_Int'] = np.floor(train_df['Episode_Length_minutes'])
train_df['Episode_Length_Dec'] = train_df['Episode_Length_minutes'] - train_df['Episode_Length_Int']


test_df['Episode_Length_Sin'] = np.sin(2 * np.pi *
                                        test_df['Episode_Length_minutes']/60)
test_df['Episode_Length_Cos'] = np.cos(2 * np.pi *
                                        test_df['Episode_Length_minutes']/60)
test_df['Episode_Length_Int'] = np.floor(test_df['Episode_Length_minutes'])
test_df['Episode_Length_Dec'] = test_df['Episode_Length_minutes'] - test_df['Episode_Length_Int']



# Mapping
weekday_map = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
train_df['T_Publication_Day'] = train_df['Publication_Day'].map(weekday_map)
test_df['T_Publication_Day'] = test_df['Publication_Day'].map(weekday_map)

time_map = {'Morning':1,'Afternoon':2,'Evening':3,'Night':4}
train_df['T_Publication_Time'] = train_df['Publication_Time'].map(time_map)
test_df['T_Publication_Time'] = test_df['Publication_Time'].map(time_map)


train_df['Weekday_Sin'] = np.sin(2 * np.pi * train_df['T_Publication_Day'] / 7)
train_df['Weekday_Cos'] = np.cos(2 * np.pi * train_df['T_Publication_Day'] / 7)

test_df['Weekday_Sin'] = np.sin(2 * np.pi * test_df['T_Publication_Day'] / 7)
test_df['Weekday_Cos'] = np.cos(2 * np.pi * test_df['T_Publication_Day'] / 7)

train_df['Time_Sin'] = np.sin(2 * np.pi * train_df['T_Publication_Time'] / 4)
train_df['Time_Cos'] = np.cos(2 * np.pi * train_df['T_Publication_Time'] / 4)

test_df['Time_Sin'] = np.sin(2 * np.pi * test_df['T_Publication_Time'] / 4)
test_df['Time_Cos'] = np.cos(2 * np.pi * test_df['T_Publication_Time'] / 4)


train_df.drop(columns=['T_Publication_Day','T_Publication_Time'],inplace=True)
test_df.drop(columns=['T_Publication_Day','T_Publication_Time'],inplace=True)


train_df.shape, test_df.shape


train_df.info()


from itertools import combinations
cat_cols = train_df.select_dtypes(exclude=np.number).columns
for col1, col2 in combinations(cat_cols, 2):
  train_df[f'{col1}-{col2}'] = train_df[col1].astype(str) + '-' + train_df[col2].astype(str)
  test_df[f'{col1}-{col2}'] = test_df[col1].astype(str) + '-' + test_df[col2].astype(str)


# convert to categorical columns
cat_cols = train_df.select_dtypes(exclude=np.number).columns
for col in cat_cols:
  train_df[col] = train_df[col].astype('category')
  test_df[col] = test_df[col].astype('category')


train_df.shape, test_df.shape


# train_df.to_csv('TRAIN_df.csv')
# test_df.to_csv('TEST_df.csv')


X = train_df.drop('Listening_Time_minutes',axis=1)
y = train_df['Listening_Time_minutes']


cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_cols)


# def objective(trial):
#     params = {
#         "iterations": 1000,
#         "depth": trial.suggest_int("depth", 4, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
#         "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
#         "border_count": trial.suggest_int("border_count", 32, 255),
#         "task_type": "GPU",
#         "eval_metric": "RMSE",
#         "early_stopping_rounds": 50,
#         "verbose": 1000,
#     }

#     model = CatBoostRegressor(**params)
#     model.fit(
#         train_pool,
#         eval_set=valid_pool,
#         use_best_model=True,
#     )

#     preds = model.predict(valid_pool)
#     rmse =root_mean_squared_error(y_valid, preds)
#     return rmse

# # Optuna study
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=30)

# print("Best RMSE:", study.best_value)
# print("Best Params:", study.best_params)


best_params = {
    'depth': 10,
    'learning_rate': 0.10435237895597081,
    'l2_leaf_reg': 0.8460665640004575,
    'bagging_temperature': 0.4916331159348723,
    'border_count': 172
}

# Initialize final model
final_model = CatBoostRegressor(
    **best_params,
    iterations=1200,
    task_type="GPU",
    eval_metric="RMSE",
    early_stopping_rounds=50,
    verbose=100
)

# Fit model
final_model.fit(
    train_pool,
    eval_set=valid_pool,
    use_best_model=True,
)

# Predict on train and valid sets
train_preds = final_model.predict(train_pool)
valid_preds = final_model.predict(valid_pool)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(train_pool.get_label(), train_preds))
valid_rmse = np.sqrt(mean_squared_error(valid_pool.get_label(), valid_preds))

print(f"Final Train RMSE: {train_rmse:.5f}")
print(f"Final Validation RMSE: {valid_rmse:.5f}")



final_model.get_feature_importance(prettified=True)


test_pred = final_model.predict(test_df)


id = test_df.index.to_list()
res = []
for i in range(len(id)):
  res.append({
      'id': id[i],
      'Listening_Time_minutes': test_pred[i]
  })
submit = pd.DataFrame(res)
submit.to_csv('submission.csv',index=False)

