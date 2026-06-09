# ðŸš€ Podcast Listening Time Prediction

# 1. Import Libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 2. Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# 3. Feature Engineering
def feature_engineering(df):
    # Is_Night
    df['Is_Night'] = (df['Publication_Time'] == 'Night').astype(int)

    # Is_Weekend
    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

    # Ad_Density
    df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']

    # Host_vs_Guest_Popularity_Diff
    df['Host_vs_Guest_Popularity_Diff'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']

    return df

train = feature_engineering(train)
test = feature_engineering(test)

# 4. Fill Missing Values
for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']:
    median = train[col].median()
    train[col].fillna(median, inplace=True)
    test[col].fillna(median, inplace=True)

# 5. Prepare Features
features = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Number_of_Ads',
    'Is_Night',
    'Is_Weekend',
    'Ad_Density',
    'Host_vs_Guest_Popularity_Diff'
]

X = train[features]
y = train['Listening_Time_minutes']
X_test = test[features]

# 6. Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model
model = lgb.LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=7,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

# 8. Predict
y_pred = model.predict(X_test)

# 9. Submission
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': y_pred
})
submission.to_csv('submission.csv', index=False)

print('âœ… Submission file created!')








