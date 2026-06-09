# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

df_train.shape,df_test.shape,df_sub.shape


df_train.isnull().sum()


df_test.isnull().sum()


df_train["Episode_Length_minutes"] = df_train.groupby(["Podcast_Name", "Genre"])["Episode_Length_minutes"]\
    .transform(lambda x: x.fillna(x.mean()))
global_mean = df_train["Episode_Length_minutes"].mean()
df_train["Episode_Length_minutes"] = df_train["Episode_Length_minutes"].fillna(global_mean)

df_train.isnull().sum()


df_test["Episode_Length_minutes"] = df_test.groupby(["Podcast_Name", "Genre"])["Episode_Length_minutes"]\
    .transform(lambda x: x.fillna(x.mean()))
    
global_mean = df_test["Episode_Length_minutes"].mean()
df_test["Episode_Length_minutes"] = df_test["Episode_Length_minutes"].fillna(global_mean)

df_test.isnull().sum()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_train, x='Episode_Length_minutes', y='Listening_Time_minutes', color='blue')

plt.title('Episode Length vs Listening Time for Digital Digest', fontsize=16)
plt.xlabel('Episode Length (minutes)', fontsize=14)
plt.ylabel('Listening Time (minutes)', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()


from sklearn.linear_model import LinearRegression
import pandas as pd

target = 'Listening_Time_minutes'

def df_encode(df):
    # 对所有类别特征进行 One-Hot 编码，包括 Podcast_Name 和 Episode_Title
    df = pd.get_dummies(df, columns=['Genre', 'Publication_Day', 'Publication_Time'], drop_first=True)


    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    if 'Guest_Popularity_percentage' in df.columns:
        df.drop(columns=['Guest_Popularity_percentage'], inplace=True)
        
    if 'Number_of_Ads' in df.columns:
        df.drop(columns=['Number_of_Ads'], inplace=True)

    if 'Podcast_Name' in df.columns:
        df.drop(columns=['Podcast_Name'], inplace=True)
        
    if 'Episode_Title' in df.columns:
        df.drop(columns=['Episode_Title'], inplace=True)

    if 'Episode_Sentiment' in df.columns:
        df.drop(columns=['Episode_Sentiment'], inplace=True)
        
    if 'Host_Popularity_percentage' in df.columns:
        df.drop(columns=['Host_Popularity_percentage'], inplace=True)
        
    return df
    
    

df_train = df_encode(df_train)
df_train.isnull().sum()

df_test = df_encode(df_test)
df_test.isnull().sum()



df_train.head(10)


def df_processing(df):
    X = df.copy()
    try:
        y = X.pop(target)
        return X, y
    except:
        pass
        return X


X_tr, y_tr = df_processing(df_train)
X_tr.sample(5)


X_ts = df_processing(df_test)
X_ts.sample(5)


from sklearn.compose import make_column_transformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder

features_trans = make_column_transformer(
    (StandardScaler(), X_tr.select_dtypes('number').columns.tolist()),
    (OneHotEncoder(), X_tr.select_dtypes(exclude='number').columns.tolist()),
    remainder='drop', 
    sparse_threshold=0)


from sklearn.model_selection import train_test_split
seed = 42

X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=seed)

[d.shape for d in [X_train, X_val, y_train, y_val]]

X_train.sample(5)


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor
)
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, roc_auc_score

# ✅ 设置目标变量
X, y = X_tr, y_tr


# 模型超参数（根据实际情况调整）
lgb_params = {'random_state': 42}
cat_params = {'random_seed': 42}
xgb_params = {'random_state': 42}

#lgb_params = {'random_state': 42, 'device': 'gpu'}
#cat_params = {'random_seed': 42, 'task_type': 'GPU'}
#xgb_params = {'random_state': 42, 'tree_method': 'hist', 'device': 'cuda'}

# 定义多个回归器
estimators = [
    ('rfr', LGBMRegressor(random_state=42)),
    ('etr', ExtraTreesRegressor(n_estimators=50, max_depth=10, n_jobs=-1,random_state=42)),
    ('hgb', HistGradientBoostingRegressor(random_state=42)),
    ('cat', CatBoostRegressor(**cat_params, verbose=0)),
    ('lgb', LGBMRegressor(**lgb_params, verbose=-1)),
    ('xgb', XGBRegressor(**xgb_params)),
]

# Stacking Regressor Pipeline
stack_model = make_pipeline(
    StandardScaler(),
    StackingRegressor(
        estimators=estimators,
        final_estimator=LGBMRegressor(**lgb_params, verbose=-1),
        n_jobs=-1
    )
)

# KFold交叉验证
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

y_true, y_pred = list(), list()

for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
    X_train, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test_fold = X.iloc[test_idx], y.iloc[test_idx]

    stack_model.fit(X_train.values, y_train_fold.values)
    preds = stack_model.predict(X_test.values)

    y_true.extend(y_test_fold)
    y_pred.extend(preds)
    

# 模型评估
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"OOF RMSE: {rmse:.4f}")
print(f"OOF R²:   {r2:.4f}")


# %% [code]
# Predictions on the Test Set
# Use the trained LightGBM model to make predictions on the test set
import os


test_preds = stack_model.predict(X_ts)

# Create the submission DataFrame
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Listening_Time_minutes': test_preds
})

# Save the submission file to the local directory as "submission.csv"
from datetime import datetime   
# 生成时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式化时间 YYYYMMDD_HHMMSS

# 生成文件名
filename = f"submission_{timestamp}.csv"

submission.to_csv(filename, index=False)
print("\nSubmission file created: submission.csv")

# List current directory files to verify the submission file creation
print("\nCurrent Directory Files:")
print(os.listdir('.'))

