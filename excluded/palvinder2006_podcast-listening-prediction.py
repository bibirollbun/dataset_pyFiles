import numpy as np
import pandas as pd
import os

# Scikit-learn tools
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import Ridge


for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e4'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train.info()


train.isnull().sum()


test.isnull().sum()


train['is_train'] = 1
test['is_train'] = 0
test['Listening_Time_minutes'] = None


combined = pd.concat([train, test], ignore_index=True)


num_cols = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']
imputer = SimpleImputer(strategy='median')
combined[num_cols] = imputer.fit_transform(combined[num_cols])


cat_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Episode_Sentiment']
for col in cat_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])


time_map = {
    "Morning": 9,
    "Afternoon": 14,
    "Evening": 19,
    "Night": 22,
    "Midnight": 0,
    "Noon": 12
}


combined['Publication_Hour'] = combined['Publication_Time'].map(time_map)
combined.drop(columns=['Publication_Time', 'Episode_Title'], inplace=True)


train_processed = combined[combined['is_train'] == 1].drop(columns=['is_train'])
test_processed = combined[combined['is_train'] == 0].drop(columns=['is_train', 'Listening_Time_minutes'])


sample_train = train_processed.sample(n=200000, random_state=42)
X_sample = sample_train.drop(columns=['Listening_Time_minutes', 'id'])
y_sample = sample_train['Listening_Time_minutes']
X_test = test_processed.drop(columns=['id'])


# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_sample, y_sample)


# Predict and save submission
predictions_lr = lr_model.predict(X_test)
predictions_lr


# Ridge Regression
ridge_model = Ridge(alpha=1.0)


ridge_model.fit(X_sample, y_sample)


predictions_ridge = ridge_model.predict(X_test)


predictions_ridge


# HistGradientBoosting (Full Data)
model = HistGradientBoostingRegressor(
    max_iter=350,
    learning_rate=0.03,
    max_depth=7,
    random_state=42
)
model.fit(X_sample, y_sample)


predictions_hgb = model.predict(X_test)


predictions_hgb


submission_lr = submission.copy()
submission_lr['Listening_Time_minutes'] = predictions_lr
submission_lr.to_csv('submission_lr.csv', index=False)

submission_ridge = submission.copy()
submission_ridge['Listening_Time_minutes'] = predictions_ridge
submission_ridge.to_csv('submission_ridge.csv', index=False)

submission_hgb = submission.copy()
submission_hgb['Listening_Time_minutes'] = predictions_hgb
submission_hgb.to_csv('submission_hgb.csv', index=False)




