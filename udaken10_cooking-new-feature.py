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
import datetime
from datetime import datetime
import numpy as np


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')
sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

train['expected_day'] = (train.index.values) % 365 + 1

train['day'] = train['expected_day']
train.drop('expected_day', axis=1, inplace=True)

train_1 = train[0:365]
train_2 = train[365:730]
train_3 = train[730:1095]
train_4 = train[1095:1460]
train_5 = train[1460:1825]
train_6 = train[1825:2190]

train_1['day_of_year'] = train_1['day']
train_1

from datetime import datetime

train_1['date'] = train_1['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))

train_2['day_of_year'] = train_2['day']
train_3['day_of_year'] = train_3['day']
train_4['day_of_year'] = train_4['day']
train_5['day_of_year'] = train_5['day']
train_6['day_of_year'] = train_6['day']
train_2['date'] = train_2['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))
train_3['date'] = train_3['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))
train_4['date'] = train_4['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))
train_5['date'] = train_5['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))
train_6['date'] = train_6['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))

train_2['date'] = train_2['date'].apply(lambda x: x.replace(year=x.year + 1))
train_3['date'] = train_3['date'].apply(lambda x: x.replace(year=x.year + 2))
train_4['date'] = train_4['date'].apply(lambda x: x.replace(year=x.year + 3))
train_5['date'] = train_5['date'].apply(lambda x: x.replace(year=x.year + 4))
train_6['date'] = train_6['date'].apply(lambda x: x.replace(year=x.year + 5))



train_data = pd.concat([train_1, train_2, train_3, train_4, train_5, train_6])
train_data


train_data.set_index('date', inplace=True)
train_data.index = pd.to_datetime(train_data.index)
train_data.head()

target = train_data.rainfall



target


test_1 = test[0:365]
test_2 = test[365:730]

test_1['date'] = test_1['day'].apply(lambda x: datetime.strptime(str(x), '%j'))
test_2['date'] = test_2['day'].apply(lambda x: datetime.strptime(str(x), '%j'))

test_1['date'] = test_1['date'].apply(lambda x: x.replace(year=x.year + 6))
test_2['date'] = test_2['date'].apply(lambda x: x.replace(year=x.year + 7))

test_data = pd.concat([test_1, test_2])

test_data.set_index('date', inplace=True)

test_data.index = pd.to_datetime(test_data.index)

test_data['day_of_year'] = test_data['day']
test_data


train_df = train_data.copy()
test_df = test_data.copy()


target = train_data.rainfall
train_df.drop('rainfall', axis=1, inplace=True)


test_df['day'] = test_df.index.day_of_year
test_df['month'] = test_df.index.month
test_df['year'] = test_df.index.year

train_df['day'] = train_df.index.day_of_year
train_df['month'] = train_df.index.month
train_df['year'] = train_df.index.year



# 新しい特徴量の作成
# make new feature
def cook_new_features(df):

    df["temp_dew_diff"] = df["temparature"] - df["dewpoint"] # ΔT = 気温 (°C) − 露点温度 (°C)
    df["temp_range"] = df["maxtemp"] - df["mintemp"] # 1日の気温差
    df["saturation_index"] = df["humidity"] / (df["dewpoint"] + 1e-6)  # 飽和度指数 = 湿度 (%) ÷ 露点温度 (°C)
    df["pressure_change"] = df["pressure"].diff().fillna(0) # 全日との気圧の変化
    df["cloud_sun_ratio"] = df["cloud"] / (df["sunshine"] + 1e-6) # 雲の量が多く、日照時間が身近ければ多くなる値
    df['3days_moving_ave_cloud'] = df['cloud'].rolling(3).mean() # 雲の量の３日単純移動平均
    df['6das_moving_ave_cloud'] = df['cloud'].rolling(6).mean() # 雲の量の６日単純移動平均
    df["wind_x"] = df["windspeed"] * np.cos(np.radians(df["winddirection"])) # 風速*風向
    df["wind_y"] = df["windspeed"] * np.sin(np.radians(df["winddirection"])) # 風速*風向
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 365) #季節周期の中の日付
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 365) # 季節周期の中の日付
    df['3days_moving_ave'] = df['humidity'].rolling(3).mean() # 湿度の３日間移動平均
    df['6days_moving_ave'] = df['humidity'].rolling(6).mean() # 湿度の６日間移動平均
    df['windspeed*humidity'] = df['windspeed'] * df['humidity'] # 風速と湿度をかけたもの
    df['3days_winspe*humidty'] = df['windspeed*humidity'].rolling(3).mean() # 風速と湿度をかけたものの、３日間単純移動平均
    df['3days_moving_ave_windspeed'] = df['windspeed'].rolling(3).mean() # 風速の３日間単純移動平均
    df['6days_moving_ave_windspeed'] = df['windspeed'].rolling(6).mean() # 風速の６日間単純移動平均
    return df

cook_new_features(train_df)
cook_new_features(test_df)


train_df.fillna(train.median(), inplace = True)
test_df.fillna(train.median(), inplace = True)




col_to_use = ['pressure', 'humidity', 'cloud', 'sunshine','temp_dew_diff', 'temp_range', 'saturation_index',
       'pressure_change', 'cloud_sun_ratio', 'wind_x', 'wind_y', 'day_sin',
       'day_cos', '3days_moving_ave_cloud', '6das_moving_ave_cloud',
       '3days_moving_ave', '6days_moving_ave', 'windspeed*humidity',
       '3days_winspe*humidty', '3days_moving_ave_windspeed',
       '6days_moving_ave_windspeed']



target


train_df = train_df[col_to_use]
test_df = test_df[col_to_use]

train_df['rainfall'] = target


train_df.fillna(train_df.median(), inplace = True)
test_df.fillna(train_df.median(), inplace = True)

train_df.isna().sum()
test_df.isnull().sum()


train_df



from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Define features (X) and target (y)
X = train_df.drop(columns=['rainfall'])
y = train_df['rainfall'].astype(int)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store predictions and scores
predictions = []
scores = []

# Iterate through folds
for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Initialize and train CatBoostClassifier
    model = CatBoostClassifier(
        iterations=1000,  # Adjust as needed
        learning_rate=0.1, # Adjust as needed
        eval_metric='AUC', # Evaluation metric
        loss_function='Logloss', # Loss function for binary classification
        random_seed=42, # for reproducibility
        verbose=100
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=100)

    # Predict probabilities for the validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    scores.append(roc_auc)
    print(f"Fold {fold + 1} ROC AUC: {roc_auc}")
    
    #Predict on test data
    predictions.append(model.predict_proba(test_df)[:,1])

print(f"Mean ROC AUC across folds: {np.mean(scores)}")

# Average the predictions from each fold
final_predictions = np.mean(predictions, axis=0)

sub['rainfall'] = final_predictions
sub.to_csv('submission.csv', index = False)
sub




