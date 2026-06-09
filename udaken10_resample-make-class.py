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


def saturation_vapor_pressure(T):
    return 6.112 * np.exp(17.67 * T / (T + 243.5))

def feature_cooking(df):


    # LCL高さ（Lifting Condensation Level Height）
    df['LCL_height'] = (125 * (df['temparature'] - df['dewpoint'])) / (df['temparature'] - 17.78)

    # 湿球温度（Wet Bulb Temperature）
    df['wet_bulb_temperature'] = df['temparature'] - ((df['temparature'] - df['dewpoint']) * (1 - df['humidity']/100))

    # 季節性指標（日付の正弦と余弦）
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # 温度範囲
    df['temperature_range'] = df['maxtemp'] - df['mintemp']

    # 温度-露点差
    df['temperature_dewpoint_diff'] = df['temparature'] - df['dewpoint']

    # 比湿（Specific Humidity）
    df['e_s'] = saturation_vapor_pressure(df['temparature'])
    df['specific_humidity'] = (0.622 * (df['humidity']/100 * df['e_s'])) / (df['pressure'] - (1 - 0.622) * (df['humidity']/100 * df['e_s']))

    # 風向成分（正弦と余弦）
    df['wind_direction_sin'] = np.sin(df['winddirection'] * np.pi / 180)
    df['wind_direction_cos'] = np.cos(df['winddirection'] * np.pi / 180)

    # 風速の2乗
    df['windspeed_squared'] = df['windspeed'] ** 2

    return df

feature_cooking(train_data)
feature_cooking(test_data)



def cook(df):
    for c in ['pressure', 'maxtemp', 'temparature', 'mintemp','dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection','windspeed']:
        for gap in [1]:
            df[c+f"_shift{gap}"]=df[c].shift(gap)
            df[c+f"_diff{gap}"]=df[c].diff(gap)
    return df

cook(train_data)
cook(test_data)


train_df_rainfall_1 = train_data[train_data['rainfall']==1]
train_df_rainfall_0 = train_data[train_data['rainfall']==0]

train_df_rainfall_1_sampled = train_df_rainfall_1.sample(n=len(train_df_rainfall_0), replace=True)

combined_df = pd.concat([train_df_rainfall_1_sampled, train_df_rainfall_0])


import matplotlib.pyplot as plt
import seaborn as sns

# Concatenate the sampled DataFrame with the rainfall=0 DataFrame
combined_df = pd.concat([train_df_rainfall_1_sampled, train_df_rainfall_0])

# Reset index to avoid issues with the 'date' index
combined_df = combined_df.reset_index()

# Now, perform EDA on the combined DataFrame
for col in combined_df.columns:
  if col not in ['rainfall', 'date']:  # Exclude 'rainfall' and 'date' columns
    plt.figure(figsize=(10, 6))
    # Use 'rainfall' column for hue mapping
    sns.histplot(combined_df, x=col, hue='rainfall', kde=True)  
    plt.title(f'Distribution of {col} by Rainfall')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='rainfall', y=col, data=combined_df)
    plt.title(f'Boxplot of {col} by Rainfall')
    plt.show()

# Calculate and display correlation matrix
correlation_matrix = combined_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# box plotに大きな差がある場合、thresholdを決め、2クラスに分けます

humidity_thre = 80
cloud_thre = 75
sunshine_thre = 5
temparature_range_thre = 4
temparature_dewpoint_diff_thre = 4
sunshine_shift1_thre = 5
sunshine_diff1_thre = 1



train_data['humidity_class'] = (train_data['humidity'] > humidity_thre).astype(int)
train_data['cloud_class'] = (train_data['cloud'] > cloud_thre).astype(int)
train_data['sunshine_class'] = (train_data['sunshine'] < sunshine_thre).astype(int)
train_data['temparature_range_class'] = (train_data['temperature_range'] < temparature_range_thre).astype(int)
train_data['temparature_dewpoint_diff_class'] = (train_data['temperature_dewpoint_diff'] < temparature_dewpoint_diff_thre).astype(int)
train_data['sunshine_shift1_class'] = (train_data['sunshine_shift1'] < sunshine_shift1_thre).astype(int)
train_data['sunshine_diff1_class'] = (train_data['sunshine_diff1'] < sunshine_diff1_thre).astype(int)



test_data['humidity_class'] = (test_data['humidity'] > humidity_thre).astype(int)
test_data['cloud_class'] = (test_data['cloud'] > cloud_thre).astype(int)
test_data['sunshine_class'] = (test_data['sunshine'] < sunshine_thre).astype(int)
test_data['temparature_range_class'] = (test_data['temperature_range'] < temparature_range_thre).astype(int)
test_data['temparature_dewpoint_diff_class'] = (test_data['temperature_dewpoint_diff'] < temparature_dewpoint_diff_thre).astype(int)
test_data['sunshine_shift1_class'] = (test_data['sunshine_shift1'] < sunshine_shift1_thre).astype(int)
test_data['sunshine_diff1_class'] = (test_data['sunshine_diff1'] < sunshine_diff1_thre).astype(int)



# Calculate the correlation matrix
correlation_matrix = train_data.corr()

# Select features with absolute correlation greater than a threshold
threshold = 0.1  # Adjust this threshold as needed
selected_features = correlation_matrix[abs(correlation_matrix['rainfall']) > threshold].index

# Print or use the selected features
print(selected_features)

# Example: Create a new DataFrame with only the selected features
# selected_df = train_data[selected_features]



# selected_featuresから重複しているものを除きます

train_data = train_data[['windspeed', 'rainfall','dewpoint_diff1',
       'humidity_shift1', 'humidity_diff1', 'cloud_shift1', 'cloud_diff1',
       'sunshine_shift1', 'sunshine_diff1', 'humidity_class', 'cloud_class',
       'sunshine_class', 'temparature_range_class',
       'temparature_dewpoint_diff_class', 'sunshine_shift1_class',
       'sunshine_diff1_class']]

test_data = test_data[[ 'windspeed', 
        'dewpoint_diff1',
       'humidity_shift1', 'humidity_diff1', 'cloud_shift1', 'cloud_diff1',
       'sunshine_shift1', 'sunshine_diff1', 'humidity_class', 'cloud_class',
       'sunshine_class', 'temparature_range_class',
       'temparature_dewpoint_diff_class', 'sunshine_shift1_class',
       'sunshine_diff1_class']]

# train_data = train_data[selected_features]
# selected_features = selected_features.drop('rainfall')
# test_data = test_data[selected_features]



from sklearn.model_selection import train_test_split

X = train_data.drop('rainfall', axis=1)
y = train_data['rainfall'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)



from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Define features (X) and target (y)
X = train_data.drop(columns=['rainfall'])
y = train_data['rainfall']

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=6, shuffle=False,
                      # random_state=42
                      )

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
    predictions.append(model.predict_proba(test_data)[:,1])

print(f"Mean ROC AUC across folds: {np.mean(scores)}")

# Average the predictions from each fold
final_predictions = np.mean(predictions, axis=0)

sub['rainfall'] = final_predictions
sub.to_csv('submission.csv', index = False)
sub



import lightgbm as lgb
import xgboost as xgb


# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=6, shuffle=False)

# Initialize lists to store predictions and scores
catboost_predictions = []
lgbm_predictions = []
xgb_predictions = []
scores = []

# Iterate through folds
for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        eval_metric='AUC',
        loss_function='Logloss',
        random_seed=42,
        verbose=100
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=100)
    catboost_predictions.append(cat_model.predict_proba(test_data)[:, 1])

    # LightGBM
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    lgb_model = lgb.train(lgb_params, lgb_train, valid_sets=lgb_eval, num_boost_round=1000)
    lgbm_predictions.append(lgb_model.predict(test_data))

    # XGBoost
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_eval = xgb.DMatrix(X_val, label=y_val)
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.05,
        'max_depth': 6
    }
    xgb_model = xgb.train(xgb_params, xgb_train, num_boost_round=1000, evals=[(xgb_eval, 'eval')], early_stopping_rounds=50, verbose_eval=100)
    xgb_predictions.append(xgb_model.predict(xgb.DMatrix(test_data)))

# Ensemble predictions
final_predictions = (np.array(catboost_predictions).mean(axis=0) + np.array(lgbm_predictions).mean(axis=0) + np.array(xgb_predictions).mean(axis=0)) / 3

sub['rainfall'] = final_predictions
# sub.to_csv('submission.csv', index=False)
sub




