import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

os.chdir('/kaggle/input/playground-series-s5e1') # Setting directory for easier data access


def add_date_features(df):
    df['day_of_week'] = df['date'].dt.day_name().str[:3].astype('category')
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    return df


def obj_to_cat(df):
    # This is useful for using xgboost built-in categorical capabilities
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    return df




df = pd.read_csv('train.csv', parse_dates=['date'], index_col='id')
df = df.dropna() # Dropping NAs for simplicity and to provide a warning, 
                 # you may choose to impute.
df = obj_to_cat(df)
df = add_date_features(df)
display(df)


train_unique = df[['country', 'store', 'product']].drop_duplicates()
print(
    "Number of unique combinations in 'Train.csv':", 
    train_unique.drop_duplicates().shape[0] # Return number of rows
)

test_df = pd.read_csv('test.csv', parse_dates=['date'], index_col='id')
test_unique = test_df[['country', 'store', 'product']].drop_duplicates()

print(
    "Number of unique combinations in 'Test.csv':", 
    test_unique.shape[0] # Return number of rows
)

outer = train_unique.merge(test_unique, how='outer', on=['country', 'store', 'product'], indicator=True)
display(outer.loc[outer._merge!='both'])


horizon = (test_df['date'].max() - test_df['date'].min()).days
print('Test Horizon:', horizon)


# assign parameters here for later use
target = 'num_sold'
horizon = 1094
nb_random_state = 1024 # For reproducibility


def tscv(df, target, horizon, n_folds, xgb_params):
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_length = (max_date - min_date).days
    trim_max = max_date - pd.Timedelta(horizon, unit='D') # We 'trim' the range to account for the horizon
    trim_length = (trim_max - min_date).days

    fold_pct = 1 / n_folds # n_folds is used here to avoid confusion with KFolds
    split_n_days = np.floor(pd.Series(np.arange(0, 1 + fold_pct, fold_pct)) * trim_length)

    mape_scores = []
    weights = np.sqrt(np.arange(1,n_folds+1))
    fold_ct = 1

    for split in split_n_days[1:]: 
        split_date = min_date + pd.Timedelta(split, unit='D')
        train = df[df['date'] <= split_date].copy()
        
        # Dropping category combo for more representative validation
        # See what happens when you skip this test. 
        train = train.loc[~((train['country']=='Canada') 
                            & (train['store']=='Discount Stickers') 
                            & (train['product']=='Kerneler'))]
        val = df[(df['date'] > split_date) & 
                 (df['date'] <= split_date + pd.Timedelta(horizon + 1, unit='D'))].copy()

        #################################################################################
        # Here is where we would drop any time dependent feature engineering functions: #
        # like rolling averages, target encodings, etc.                                 #
        # See: https://unit8co.github.io/darts/userguide/covariates.html#summary-tl-dr  #
        #################################################################################

        X_train = train.drop(columns=[target, 'date'])
        y_train = train[target]
        X_val = val.drop(columns=[target, 'date'])
        y_val = val[target]

        # I use xgboost here but modify this to your specific needs
        model = XGBRegressor(**xgb_params) 
        model.fit(X_train, y_train, verbose=False) # Beware of using early stopping here
        pred = model.predict(X_val)

        mape = mean_absolute_percentage_error(y_val, pred)
        print(f'Fold {fold_ct}/{n_folds}   Split:{split_date.date()}   MAPE:{mape*100:.2f}%')
        fold_ct += 1
        mape_scores.append(mape)

    return np.mean(mape_scores)




xgb_params = {'n_estimators': 119, 
              'max_depth': 11,
              'learning_rate': 0.15,
              'subsample': 0.95,
              'colsample_bytree': 0.95,
              'gamma': 0.8,
              'min_child_weight': 6,
              'reg_lambda': 0.003,
              'reg_alpha': 0.008,
              'enable_categorical': True,
              'random_state': nb_random_state
             }

tscv(df=df, target=target, horizon=horizon, n_folds=5, 
                   xgb_params=xgb_params)


test_df = pd.read_csv('test.csv', parse_dates=['date'], index_col='id')
sample_sub = pd.read_csv('sample_submission.csv')

test_df = obj_to_cat(test_df)
test_df = add_date_features(test_df)

X = df.drop(columns=[target, 'date'])
X_sub = test_df.drop(columns=['date'])
X_sub = X_sub[X.columns]
y = df[target]

xgb_best = XGBRegressor(**xgb_params)
xgb_best.fit(X, y)
sample_sub['num_sold'] = xgb_best.predict(X_sub)
sample_sub.to_csv('/kaggle/working/'+'Submission.csv', index=False)

