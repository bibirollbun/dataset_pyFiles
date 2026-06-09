import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
        
import warnings
warnings.filterwarnings("ignore", category=Warning)


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
print(sample_submission.shape)
sample_submission.head()


test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
print(test.shape)
test.head()


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
print(train.shape)
train.head()


train.info()


train.info()
# NaN in Episode_Length_minutes, Guest_Popularity_percentage, Number_of_Ads 


train.describe()


test.info()
# NaN in Episode_Length_minutes, Guest_Popularity_percentage


test.describe()


numeric_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                    'Guest_Popularity_percentage', 'Number_of_Ads']

for feature in numeric_features:
    plt.figure(figsize=(12, 2))

    plt.subplot(1, 2, 1)
    plt.title(f'Histogram of {feature} in Train')
    sns.histplot(train[feature], kde=True, bins=100)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.title(f'Histogram of {feature} in Test')
    sns.histplot(test[feature], kde=True, bins=100)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# target
from scipy.stats import skew

plt.figure(figsize=(6, 2))
plt.title(f'Histogram of Listening_Time_minutes in Train')
sns.histplot(train['Listening_Time_minutes'], kde=True, bins=100)
plt.xlabel('Listening_Time_minutes')
plt.ylabel('Frequency')

listening_time_skewness = skew(train['Listening_Time_minutes'])
plt.text(0.95, 0.95, f'Skewness: {listening_time_skewness:.2f}',
         transform=plt.gca().transAxes, ha='right', va='top')

plt.tight_layout()
plt.show()


categorical_features = ['Podcast_Name', 'Episode_Title', 'Genre', 
                        'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

for feature in categorical_features:
    plt.figure(figsize=(16, 3))

    value_order = train[feature].value_counts().index

    plt.subplot(1, 2, 1)
    plt.title(f'Countplot of {feature} in Train')
    sns.countplot(data=train, x=feature, order=value_order) 
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')

    plt.subplot(1, 2, 2)
    plt.title(f'Countplot of {feature} in Test')
    sns.countplot(data=test, x=feature, order=value_order)  
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


train_fil = train.copy()
test_fil = test.copy()


train_fil['Number_of_Ads'].median()


train_fil['Number_of_Ads'].fillna(train_fil['Number_of_Ads'].median(), inplace=True)
train_fil['Number_of_Ads'].isna().sum()


# train_genre_length_median
train_genre_length_median = train_fil.groupby('Genre')['Episode_Length_minutes'].median()
train_genre_length_median.describe()


# Filling NaN in Episode_Length_minutes in train
for index, row in train_fil.iterrows():
    if pd.isnull(row['Episode_Length_minutes']): # if null, 1
        genre = row['Genre'] # find Genre
        median_value = train_genre_length_median[genre]
        train_fil.loc[index, 'Episode_Length_minutes'] = median_value
train_fil['Episode_Length_minutes'].isna().sum()


# test_genre_length_median
test_genre_length_median = test_fil.groupby('Genre')['Episode_Length_minutes'].median()
test_genre_length_median.describe()


# Filling NaN in Episode_Length_minutes in test
for index, row in test_fil.iterrows():
    if pd.isnull(row['Episode_Length_minutes']): # if null, 1
        genre = row['Genre'] # find Genre
        median_value = test_genre_length_median[genre]
        test_fil.loc[index, 'Episode_Length_minutes'] = median_value
test_fil['Episode_Length_minutes'].isna().sum()


# train_podcast_guest_median
train_podcast_guest_median = train_fil.groupby(['Podcast_Name'])['Guest_Popularity_percentage'].median()
train_podcast_guest_median.describe()


# Filling NaN in Guest_Popularity_percentage in train
for index, row in train_fil.iterrows():
    if pd.isnull(row['Guest_Popularity_percentage']): # if null, 1
        name = row['Podcast_Name'] # Podcast_Name
        median_value = train_podcast_guest_median[name]
        train_fil.loc[index, 'Guest_Popularity_percentage'] = median_value
train_fil['Guest_Popularity_percentage'].isna().sum()


# test_podcast_guest_median
test_podcast_guest_median = test_fil.groupby(['Podcast_Name'])['Guest_Popularity_percentage'].median()
test_podcast_guest_median.describe()


# Filling NaN in Guest_Popularity_percentage in test
for index, row in test_fil.iterrows():
    if pd.isnull(row['Guest_Popularity_percentage']): # if null, 1
        name = row['Podcast_Name'] # find Genre
        median_value = test_podcast_guest_median[name]
        test_fil.loc[index, 'Guest_Popularity_percentage'] = median_value
test_fil['Guest_Popularity_percentage'].isna().sum()


train_fil.info()


test_fil.info()


numeric_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                    'Guest_Popularity_percentage', 'Number_of_Ads']

for feature in numeric_features:
    plt.figure(figsize=(12, 2))

    plt.subplot(1, 2, 1)
    plt.title(f'Histogram of {feature} in Train_fil')
    sns.histplot(train_fil[feature], kde=True, bins=100)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.title(f'Histogram of {feature} in Test_fil')
    sns.histplot(test_fil[feature], kde=True, bins=100)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


train_new = train_fil.copy()
test_new = test_fil.copy()
train_new.shape, test_new.shape


# 1. Ads per Length 
train_new['Ads_per_Length'] = train_new['Number_of_Ads'] / train_new['Episode_Length_minutes']

# 2. Publication Day and Time Combination
train_new['Publication_Day_Time'] = (train_new['Publication_Day'].astype(str) + '_' + 
                                     train_new['Publication_Time'].astype(str))

train_new = train_new[['Ads_per_Length', 'Publication_Day_Time']]

print(train_new.shape)
train_new.head()


# 1. Ads per Length 
test_new['Ads_per_Length'] = test_new['Number_of_Ads'] / test_new['Episode_Length_minutes']

# 2. Publication Day and Time Combination
test_new['Publication_Day_Time'] = (test_new['Publication_Day'].astype(str) + '_' + 
                                     test_new['Publication_Time'].astype(str))

test_new = test_new[['Ads_per_Length', 'Publication_Day_Time']]

print(test_new.shape)
test_new.head()


train_enl = pd.concat([train_fil, train_new], axis=1)
test_enl = pd.concat([test_fil, test_new], axis=1)
train_enl.shape, test_enl.shape


train_enl.describe(include='all').T


# target & features
target = train_enl['Listening_Time_minutes'] 
features = train_enl.drop(['id', 'Listening_Time_minutes'], axis=1)
test_features = test_enl.drop(['id'], axis=1)
print(target.shape, features.shape, test_features.shape)
features.head()


# categoric variables in features
features_cat = features.select_dtypes(include=['object'])
test_features_cat = test_features.select_dtypes(include=['object'])
print(features_cat.shape, test_features_cat.shape)
features_cat.head(2)


# onehotencoding for categorical values
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, categories='auto', handle_unknown='ignore')
ohe.fit(features_cat) 

features_ohe = pd.DataFrame(ohe.transform(features_cat), 
                            columns=ohe.get_feature_names_out(features_cat.columns), 
                            index=features_cat.index)
test_features_ohe = pd.DataFrame(ohe.transform(test_features_cat), 
                                 columns=ohe.get_feature_names_out(test_features_cat.columns), 
                                 index=test_features_cat.index)

print(features_ohe.shape, test_features_ohe.shape)
features_ohe.head(2)


# numeric variables in features
features_num = features.select_dtypes(include=['float', 'int'])
test_features_num = test_features.select_dtypes(include=['float', 'int'])
print(features_num.shape, test_features_num.shape)
features_num.head(2)


# standardization for numerical values
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(features_num[features_num.columns])

features_std = pd.DataFrame(scaler.transform(features_num[features_num.columns]), 
                            columns=features_num.columns, index=features_num.index)

test_features_std = pd.DataFrame(scaler.transform(test_features_num[test_features_num.columns]), 
                                 columns=test_features_num.columns, index=test_features_num.index)

print(features_std.shape, test_features_std.shape)
features_std.head(2)


# combine 
features_tf = pd.concat([features_ohe, features_std], axis=1)
test_features_tf = pd.concat([test_features_ohe, test_features_std], axis=1)
print(features_tf.shape, test_features_tf.shape)
features_tf.head(2)


# check duplicate
train_cols_series = pd.Series([features_tf[col].values.tobytes() for col in features_tf.columns])
train_duplicated_mask = train_cols_series.duplicated(keep='first')
train_duplicate_columns = features_tf.columns[train_duplicated_mask].tolist()

test_cols_series = pd.Series([test_features_tf[col].values.tobytes() for col in test_features_tf.columns])
test_duplicated_mask = test_cols_series.duplicated(keep='first')
test_duplicate_columns = test_features_tf.columns[test_duplicated_mask].tolist()

features_tf.drop(columns=train_duplicate_columns, inplace=True)
test_features_tf.drop(columns=test_duplicate_columns, inplace=True)

features_tf.shape, test_features_tf.shape


# split into train and test data
from sklearn.model_selection import (train_test_split, StratifiedKFold)
X_train, X_val, y_train, y_val = train_test_split(features_tf, target, test_size=0.2, random_state=42)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# XGBoost model
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import time

start_time = time.time()
xgb_model = XGBRegressor(random_state=42, n_jobs=-1, 
                         max_depth=9, min_child_weight=7, gamma=0.17323522915498704, 
                         n_estimators=960, learning_rate=0.0849080237694725, 
                         subsample=0.9745401188473625, colsample_bytree=0.6580836121681994
                        ) 
xgb_model.fit(X_train, y_train)

train_xgb_pred = xgb_model.predict(X_train)
val_xgb_pred = xgb_model.predict(X_val)

end_time = time.time()
xgb_time = end_time - start_time

train_xgb_rmse = np.sqrt(mean_squared_error(y_train, train_xgb_pred))
val_xgb_rmse = np.sqrt(mean_squared_error(y_val, val_xgb_pred))

print('XGBoost Results:')
print(f"train_RMSE: {train_xgb_rmse:.6f}")
print(f"val_RMSE: {val_xgb_rmse:.6f}")
print(f"Time: {xgb_time:.6f}")

"""
XGBoost Results:
train_RMSE: 11.199968
val_RMSE: 12.831454
Time: 129.177812
"""


# LightGBM model
from lightgbm import LGBMRegressor

start_time = time.time()
lgbm_model = LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1, 
                           num_leaves=58, max_depth=9, min_child_samples=71,  
                           n_estimators=960, learning_rate=0.0849080237694725, 
                           subsample=0.7559945203362026, colsample_bytree=0.7560186404424365, 
                           reg_alpha=0.03745401188473625, reg_lambda=0.09507143064099162
                          ) 
lgbm_model.fit(X_train, y_train)

train_lgbm_pred = lgbm_model.predict(X_train)
val_lgbm_pred = lgbm_model.predict(X_val)

end_time = time.time()
lgbm_time = end_time - start_time

train_lgbm_rmse = np.sqrt(mean_squared_error(y_train, train_lgbm_pred))
val_lgbm_rmse = np.sqrt(mean_squared_error(y_val, val_lgbm_pred))

print('LightGBM Results:')
print(f"train_RMSE: {train_lgbm_rmse:.6f}")
print(f"val_RMSE: {val_lgbm_rmse:.6f}")
print(f"Time: {lgbm_time:.6f}")

"""
LightGBM Results:
train_RMSE: 12.445245
val_RMSE: 12.925034
Time: 48.478261
"""


# CatBoost model
from catboost import CatBoostRegressor

start_time = time.time()
cat_model = CatBoostRegressor(random_state=42, verbose=0, 
                              iterations=960, learning_rate=0.18323522915498705, 
                              depth=9, l2_leaf_reg=8.080725777960454, random_strength=0.8319939418114051)
cat_model.fit(X_train, y_train)

train_cat_pred = cat_model.predict(X_train)
val_cat_pred = cat_model.predict(X_val)

end_time = time.time()
cat_time = end_time - start_time

train_cat_rmse = np.sqrt(mean_squared_error(y_train, train_cat_pred))
val_cat_rmse = np.sqrt(mean_squared_error(y_val, val_cat_pred))

print('CatBoost Results:')
print(f"train_RMSE: {train_cat_rmse:.6f}")
print(f"val_RMSE: {val_cat_rmse:.6f}")
print(f"Time: {cat_time:.6f}")
"""
CatBoost Results:
train_RMSE: 12.212951
val_RMSE: 12.926469
Time: 61.242178
"""


# Ensemble Model (VotingRegressor)
from sklearn.ensemble import VotingRegressor

ensemble_models = [
    ('XGB', xgb_model),  
    ('LGBM', lgbm_model), 
    ('CatBoost', cat_model) 
]

ensemble_model = VotingRegressor(estimators=ensemble_models)

start_time = time.time()
ensemble_model.fit(X_train, y_train)

train_ensemble_pred = ensemble_model.predict(X_train)
val_ensemble_pred = ensemble_model.predict(X_val)

end_time = time.time()
ensemble_time = end_time - start_time

train_ensemble_rmse = np.sqrt(mean_squared_error(y_train, train_ensemble_pred))
val_ensemble_rmse = np.sqrt(mean_squared_error(y_val, val_ensemble_pred))

print('Ensemble Model (VotingRegressor)')
print(f"train_RMSE: {train_ensemble_rmse:.6f}")
print(f"val_RMSE: {val_ensemble_rmse:.6f}")
print(f"Time: {ensemble_time:.6f}")

"""
Ensemble Model (VotingRegressor)
train_RMSE: 11.899292
val_RMSE: 12.841486
Time: 229.079670
"""


# Ensemble Model (StackingRegressor)
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

base_models = [
    ('XGB', xgb_model),  
    ('LGBM', lgbm_model), 
    ('CatBoost', cat_model)
]

meta_model = LinearRegression()
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)

start_time = time.time()
stacking_model.fit(X_train, y_train)

train_stacking_pred = stacking_model.predict(X_train)
val_stacking_pred = stacking_model.predict(X_val)

end_time = time.time()
stacking_time = end_time - start_time

train_stacking_rmse = np.sqrt(mean_squared_error(y_train, train_stacking_pred))
val_stacking_rmse = np.sqrt(mean_squared_error(y_val, val_stacking_pred))

print('Ensemble Model (StackingRegressor)')
print(f"train_RMSE: {train_stacking_rmse:.6f}")
print(f"val_RMSE: {val_stacking_rmse:.6f}")
print(f"Time: {stacking_time:.6f}")

"""
"""


# Ensemble Model (StackingRegressor)
test_pred = stacking_model.predict(test_features_tf)
test_pred


submission = pd.DataFrame({'id': test.id, 'Listening_Time_minutes': test_pred})
print(submission.shape)
submission.head()


submission.to_csv('submission.csv', index=False)


submission = pd.read_csv('/kaggle/working/submission.csv')
submission.head()




