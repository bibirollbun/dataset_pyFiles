# importing the libraries and setting up the display 
!pip install lightgbm
#conda install lightgbm

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')



# Quick Data Overview
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.select_dtypes(include='number').quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)



# Loading the datasets


train = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv', parse_dates=['date'])

sample_sub = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv')

# We combine train and test datasets to apply feature engineering and data preprocessing uniformly.
# While this approach has pros (consistency, fewer code repetitions), it also carries the risk of data leakage.
# However, by being cautious and not using target values from the train set during transformation,
# we can safely work on the combined data.
# We'll split them back into train and test as needed.
df = pd.concat([train, test], sort=False)


df["date"].min(), df["date"].max()


check_df(df)


df[["store"]].nunique()



df[["item"]].nunique()


df.groupby(["store"])["item"].nunique()


df.groupby(["store", "item"]).agg({"sales": ["sum"]})


df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})


df.head()


# FEATURE ENGINEERING
df.head()


# Extracting date-based features like month, day, and weekday
 
def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    # df['week_of_year'] = df.date.dt.weekofyear
    df['week_of_year'] = df.date.dt.isocalendar().week
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)

df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})


df.head()


# Generates random noise from a normal distribution to add variability.
# This can help reduce overfitting or simulate natural randomness in features.
# The noise has mean=0 and standard deviation=1.6, matching the length of the dataframe.

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


# Creates lagged sales features by shifting past values for each store-item pair.
# Adds random noise to simulate natural variation and reduce overfitting.

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

df.head()


# Creates rolling mean (moving average) features using past sales values for each store-item pair.
# Uses a triangular window and shifts by 1 to avoid data leakage.
# Adds random noise to introduce variability and prevent overfitting.
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

# this is; the moving average of the period of 1 year ago and 1,5 years ago
df = roll_mean_features(df, [365, 546])

df.head()


# Creates exponentially weighted moving average (EWMA) features for past sales.
# Applies different smoothing levels (alphas) and lags for each store-item pair.
# Higher alpha gives more weight to recent observations, capturing short-term trends.

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)
df.head()


# Applies one-hot encoding to categorical variables to convert them into binary indicator columns.
# Helps machine learning models interpret categorical features like store, item, day_of_week, and month.
# Even though these variables are numerical-looking, they are actually categorical by nature
# Each store and item represents a category or ID, not a quantity.
# They are different labels, not ranked.
# If they are not encoded properly, some models might assume Store 2 > Store 1, which is not true.
# OHE removes this false numerical relationship and treats each store/item as a separate group.
df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])
df.head()



# Converting sales to log(1+sales) transformation to reduce skewness and stabilize variance in the sales data.
df['sales'] = np.log1p(df["sales"].values)
df.head()


# Model

# Custom Cost Function
# We use custom functions to evaluate models with metrics that reflect our real-world goals more accurately than default ones(RMSE or Logloss).
# Custom SMAPE (Symmetric Mean Absolute Percentage Error) metric for LightGBM.
# - `smape()`: Calculates SMAPE between predicted and actual values, ignoring zero-zero pairs.
# - `lgbm_smape()`: LightGBM-compatible wrapper that applies inverse log transformation (expm1) 
#   to get original sales values before computing SMAPE.
#   Returns a tuple in the format LightGBM expects for custom evaluation metrics.

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False




# Time-Based Validation Sets
# until end of 2017
train

# until end of March 2018
test

# Until the end of 2016 is train set
train = df.loc[(df["date"] < "2017-01-01"), :]

# First 3 months of 2017 are validation set
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

#removing the target variable and the other non useful variables
cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales'] # target variable
X_train = train[cols] 

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape


# Time Series Model With LGBM
# !pip install lightgbm
# conda install lightgbm

# LGBM is the most successful tree model for time series, it is a Gradient Boasting based Tree Model
# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: Maximum number of leaves in one tree.
# learning_rate: Also known as shrinkage rate or eta.
# feature_fraction: Similar to Random Forest's random subspace method. 
#   Specifies the fraction of features to be randomly selected at each iteration.
# max_depth: Maximum depth of each tree.
# num_boost_round: Equivalent to n_estimators. Number of boosting iterations. 
#   Should generally be around 10,000–15,000.

# early_stopping_rounds: If the metric on the validation set doesn't improve within the specified number of rounds,
#   stop training early. 
#   Helps reduce training time and prevents overfitting.
# nthread: Number of threads to use (can also be written as num_thread, nthread, nthreads, or n_jobs).

# # Creates LightGBM datasets for training and validation.
# 'lgbtrain' is the main training set, and 'lgbval' is the validation set with a reference to 'lgbtrain' 
# to enable features like early stopping and parameter reuse.

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

# # Trains the LightGBM model using custom SMAPE evaluation.
# Includes early stopping to prevent overfitting and logs progress every 100 rounds.
from lightgbm import early_stopping, log_evaluation

model = lgb.train(
    lgb_params,
    lgbtrain,
    valid_sets=[lgbtrain, lgbval],
    num_boost_round=lgb_params['num_boost_round'],
    feval=lgbm_smape,
    callbacks=[
        early_stopping(lgb_params['early_stopping_rounds']),
        log_evaluation(100)  # logs every 100 rounds
    ]
)


# Predicts validation set using the best iteration from training.
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

# Calculates SMAPE (Symmetric Mean Absolute Percentage Error) on the original sales scale.
# Since sales were log-transformed using log1p during training, we reverse the transformation with expm1.
# SMAPE is preferred in forecasting tasks because it's scale-independent and expresses error as a percentage,
# making it easier to interpret and compare across different time periods or product categories.
smape(np.expm1(y_pred_val), np.expm1(Y_val))

smape(np.expm1(y_pred_val), np.expm1(Y_val))




# Plotting feature importance to see what features actually have an impact on model's success
def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)

# Excluding the features that have no effect on the model (gain is zero)
feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


# Establishing the final model

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]
# if we use imp_feats (from feature importance above), instead of [cols], we would exclude the features that have no impact on the model performace

test = df.loc[df.sales.isna()]
X_test = test[cols]

# we don't use early stopping, beacsue we already know the best iteration amount
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)


# these are not the real sales amounts, they are  log-transformed values
test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)



# Exporting the submission.csv file 
test.head()
# reverting log operation

# Creates a new DataFrame for submission, keeping only the id and sales columns.
submission_df = test.loc[:, ["id", "sales"]]

# Reverses the log1p transformation applied earlier during training.
submission_df['sales'] = np.expm1(test_preds)

# Ensures the id column is of integer type, as required in most submission formats.
submission_df['id'] = submission_df.id.astype(int)

# Exports the submission DataFrame to a CSV file named "submission.csv" without row indices.
submission_df.to_csv("submission.csv", index=False)
submission_df

