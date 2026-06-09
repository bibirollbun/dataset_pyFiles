import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'])


train = train.dropna(subset=['num_sold'])


print(train.info())
print(train.describe())
print(train.head())


plt.figure(figsize=(14,7))
sns.lineplot(data=train, x='date', y='num_sold', alpha=0.5, label='Daily Sales', color='blue')

train['rolling_mean_90'] = train['num_sold'].rolling(window=90).mean()
sns.lineplot(data=train, x='date', y='rolling_mean_90', label='90-Day Rolling Mean', color='red')

plt.title('Overall Sticker Sales Trends')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout
plt.show()


def create_features(df):
  df['year'] = df['date'].dt.year
  df['month'] = df['date'].dt.month
  df['day'] = df['date'].dt.day
  df['day_of_week'] = df['date'].dt.dayofweek
  df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
  return df

train = create_features(train)
test = create_features(test)


cat_features = ['country', 'store', 'product']
train = pd.get_dummies(train, columns=cat_features, drop_first=True)
test = pd.get_dummies(test, columns=cat_features, drop_first=True)


test = test.reindex(columns=train.columns, fill_value=0)


def create_lag_features(df, lags):
  for lag in lags:
    df[f'lag_{lag}'] = df['num_sold'].shift(lag)
  return df


def create_rolling_features(df, windows):
    for window in windows:
        df[f'roll_mean_{window}'] = df['num_sold'].rolling(window).mean()
        df[f'roll_std_{window}'] = df['num_sold'].rolling(window).std()
    return df


lags = [7, 14, 28]
windows = [7, 14]
train = create_lag_features(train, lags)
train = create_rolling_features(train, windows)


train = train.dropna()


X = train.drop(columns=["num_sold", "date"])
y = train["num_sold"]
X_test = test.drop(columns=["date"])

tscv = TimeSeriesSplit(n_splits=5)


from lightgbm import early_stopping
from lightgbm import log_evaluation


print("Columns in X:", X.columns.tolist())
print("Columns in X_test:", X_test.columns.tolist())


common_columns = X.columns.intersection(X_test.columns)
X = X[common_columns]
X_test = X_test[common_columns]

print("Number of features after alignment:", len(common_columns))
print("Common columns:", common_columns.tolist())


final_predictions = np.zeros(X_test.shape[0])

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold + 1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=100)
        ]
    )

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)
    final_predictions += test_preds / tscv.n_splits

    print(f'Fold {fold + 1} MAE: {mean_absolute_error(y_val, val_preds):.4f}')
    print()

print("Training completed.")


test["num_sold"] = final_predictions
submission = test[["id", "num_sold"]]
submission.to_csv("submission.csv", index=False)
print("Submission file created!")

