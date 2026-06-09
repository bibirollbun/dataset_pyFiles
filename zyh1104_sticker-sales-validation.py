import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import logging

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')
import gc


#load data
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
gc.collect()


#Set id as index

train.set_index('id', inplace=True)
test.set_index('id', inplace=True)


# Delete rows containing NaN in the num_sold column
train = train.dropna(subset=['num_sold'])


def process_date_features(df):
    df['date'] = pd.to_datetime(df['date'])

    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    df['month_name'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.day_name()

    df['week'] = df['date'].dt.isocalendar().week

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    df['group'] = (df['year'] - 2020) * 48 + df['month'] * 4 + df['day'] // 7

    df.drop('date', axis=1, inplace=True)

    df['cos_year'] = np.cos(df['year'] * (2 * np.pi) / 100)
    df['sin_year'] = np.sin(df['year'] * (2 * np.pi) / 100)

    df['Season'] = df['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                              'Spring' if x in [3, 4, 5] else
                                              'Summer' if x in [6, 7, 8] else
                                              'Autumn')

    dummy_prefixes = ['country', 'store', 'product', 'month_name', 'day_of_week', 'Season']
    df = pd.get_dummies(df, columns=dummy_prefixes, drop_first=True)

    return df

train = process_date_features(train)
test = process_date_features(test)


train.head()


test.head()


train.info()


logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

X = train.drop(columns=['num_sold'])
y = np.log1p(train['num_sold'])
test = test[X.columns]

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def cross_val_lgbm_mape(X, y, test, n_splits=5, **params):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    preds = []

    for train_index, valid_index in kf.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        model = lgb.LGBMRegressor(
            **params,
            n_estimators=2000,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid)
        score = mape(np.expm1(y_valid), np.expm1(y_pred))
        mape_scores.append(score)

        preds.append(model.predict(test))

    test_preds_mean = np.mean(preds, axis=0)

    return np.mean(mape_scores), test_preds_mean

model_params = {
    "objective": "regression",
    "metric": "mape",
    "verbose": -1  
}

average_mape, lgb_preds = cross_val_lgbm_mape(X, y, test, n_splits=5, **model_params)
print(f"Average MAPE across folds: {average_mape:.4f}")


test_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
test_preds_final = np.expm1(lgb_preds)
submission = pd.DataFrame({
    'id': test_submission['id'],
    'num_sold': test_preds_final
})
submission.to_csv('submission_lgb.csv', index=False)
print("Submission file created:")
print(submission.head())

