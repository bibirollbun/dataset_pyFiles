import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv", index_col='id')
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv", index_col='id')


print("train_df shape :",train_df.shape)
print("test_df shape :",test_df.shape)


train_df.head()


test_df.head()


train_df.info()


train_df.isna().sum()


train_df['Episode_Length_minutes'] = train_df.groupby('Podcast_Name')['Episode_Length_minutes'].transform(
    lambda x: x.fillna(x.median())
)


train_df['Guest_Popularity_missing'] = train_df['Guest_Popularity_percentage'].isna().astype(int)
train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(0)


train_df['Number_of_Ads'] = train_df.groupby('Podcast_Name')['Number_of_Ads'].transform(
    lambda x: x.fillna(x.median())
)


test_df.isna().sum()


test_df['Episode_Length_minutes'] = test_df.groupby('Podcast_Name')['Episode_Length_minutes'].transform(
    lambda x: x.fillna(x.median())
)


test_df['Guest_Popularity_missing'] = test_df['Guest_Popularity_percentage'].isna().astype(int)
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].fillna(0)


# categorical columns to encode
categorical_cols = ['Podcast_Name', 'Genre', 'Episode_Title', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# encode categories
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.fit_transform(test_df[col])


# split features and target
X = train_df.drop(columns=['Listening_Time_minutes'])
y = train_df['Listening_Time_minutes']

# train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=64,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(100)]
)


# Predict
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"Validation RMSE: {rmse:.4f}")


# Final prediction
test_preds = model.predict(test_df)
submission = pd.DataFrame({'id': test_df.index, 'Listening_Time_minutes': test_preds})
submission.to_csv('submission.csv', index=False)

