import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from category_encoders import TargetEncoder

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

seed = 11


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train.head()


test.head()


train.info()


print("% of empty rows for each column in train:")
print(train.isna().sum()/len(train) * 100)
print("-"*50)
print("% of empty rows for each column in test:")
print(test.isna().sum()/len(test) * 100)


train_df = train.drop(['Podcast_Name', 'Episode_Title'], axis=1).copy()
test_df = test.drop(['Podcast_Name', 'Episode_Title'], axis=1).copy()

train_df['Length_missing'] = train_df['Episode_Length_minutes'].isna().astype(int)
test_df['Length_missing'] = test_df['Episode_Length_minutes'].isna().astype(int)

train_df['Guest_Popularity_missing'] = train_df['Guest_Popularity_percentage'].isna().astype(int)
test_df['Guest_Popularity_missing'] = test_df['Guest_Popularity_percentage'].isna().astype(int)

sentiment_map = {"Negative": -1, "Neutral": 0, "Positive":1}

train_df['Episode_Sentiment'] = train_df['Episode_Sentiment'].map(sentiment_map)
test_df['Episode_Sentiment'] = test_df['Episode_Sentiment'].map(sentiment_map)


el_mean = train_df['Episode_Length_minutes'].mean()
train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(el_mean)
test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(el_mean)

gp_mean = train_df['Guest_Popularity_percentage'].mean()
train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(gp_mean)
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].fillna(gp_mean)

train_df['hpXgp'] = np.sqrt(train_df['Host_Popularity_percentage'] * train_df['Guest_Popularity_percentage'])
test_df['hpXgp'] = np.sqrt(test_df['Host_Popularity_percentage'] * test_df['Guest_Popularity_percentage'])

train_df['Number_of_Ads'] = train_df['Number_of_Ads'].fillna(0)


label_encoder = LabelEncoder()
train_df['Publication_Day'] = label_encoder.fit_transform(train_df['Publication_Day'])
test_df['Publication_Day'] = label_encoder.transform(test_df['Publication_Day'])

train_df = pd.get_dummies(data=train_df, columns=['Publication_Time'], dtype=int)
test_df = pd.get_dummies(data=test_df, columns=['Publication_Time'], dtype=int)


targ_enc = train_df[['Genre','Listening_Time_minutes']].groupby(by='Genre').mean()
train_df['Genre_encoded'] = targ_enc.loc[train_df['Genre']].values.squeeze()
test_df['Genre_encoded'] = targ_enc.loc[test_df['Genre']].values.squeeze()


features = train_df.columns.tolist()
if 'id' in features:
    features.remove('id')
if 'Listening_Time_minutes' in features:
    features.remove('Listening_Time_minutes')
if 'Genre' in features:
    features.remove('Genre')

target = 'Listening_Time_minutes'

X, y = train_df[features], train_df[target]


# # LGBM 

# n_splits = 5
# kf = KFold(n_splits=n_splits)

# scores = list()

# for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     model = LGBMRegressor(random_state=seed, verbose=0)

#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_val)

#     score = mean_squared_error(y_val, y_pred, squared=False)
#     print(f"fold: {i} ==> rmse: {score}")
#     scores.append(score)

# print("-"*40)
# print(f"Mean RMSE: {np.mean(scores)}")

#================================================================
# # catboost

# n_splits = 5
# kf = KFold(n_splits=n_splits)

# scores = list()

# for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     model = CatBoostRegressor(random_state=seed, verbose=False)

#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_val)

#     score = mean_squared_error(y_val, y_pred, squared=False)
#     print(f"fold: {i} ==> rmse: {score}")
#     scores.append(score)

# print("-"*40)
# print(f"Mean RMSE: {np.mean(scores)}")

#================================================================

# # XGBRegressor

# n_splits = 5
# kf = KFold(n_splits=n_splits)

# scores = list()

# for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     model = XGBRegressor(random_state=seed)

#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_val)

#     score = mean_squared_error(y_val, y_pred, squared=False)
#     print(f"fold: {i} ==> rmse: {score}")
#     scores.append(score)

# print("-"*40)
# print(f"Mean RMSE: {np.mean(scores)}")

#================================================================


# # Linear Regression

# n_splits = 5
# kf = KFold(n_splits=n_splits)

# scores = list()

# for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    
#     model = make_pipeline(StandardScaler(), LinearRegression(n_jobs=-1))

#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_val)

#     score = mean_squared_error(y_val, y_pred, squared=False)
#     print(f"fold: {i} ==> rmse: {score}")
#     scores.append(score)

# print("-"*40)
# print(f"Mean RMSE: {np.mean(scores)}")

#================================================================

# # RandomForestRegressor

# n_splits = 5
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

# scores = list()

# for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     model = RandomForestRegressor(n_estimators = 65, min_samples_split=50, n_jobs=-1, random_state=seed)

#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_val)

#     score = mean_squared_error(y_val, y_pred, squared=False)
#     print(f"fold: {i} ==> rmse: {score}")
#     scores.append(score)

# print("-"*40)
# print(f"Mean RMSE: {np.mean(scores)}")


# model_cat = CatBoostRegressor(random_state=seed, verbose=False)
# model_cat.fit(X, y)

model_rfr = RandomForestRegressor(n_estimators = 150, min_samples_split=10, n_jobs=-1, random_state=seed)
model_rfr.fit(X, y)


# preds_cat = model_cat.predict(test_df[features])
preds_rfr = model_rfr.predict(test_df[features])

sample_submission['Listening_Time_minutes'] = preds_rfr
sample_submission.to_csv("submission.csv", index=False)

