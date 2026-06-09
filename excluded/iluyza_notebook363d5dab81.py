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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e1/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e1/test.csv")
sample_submission = pd.read_csv(r"/kaggle/input/playground-series-s5e1/sample_submission.csv")

print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("sample_submission shape :",sample_submission.shape)


train_data.head()


train_data.isna().sum().sort_values(ascending=False)


train_data['country'].value_counts()


train_data = train_data.drop_duplicates()
train_data = train_data.dropna()
print("train_data shape :",train_data.shape)


test_data.head()


test_data.isna().sum().sort_values(ascending=False)


import numpy as np
import pandas as pd

def date_feature_engineering(df):
    """
    Performs feature engineering on a date column in the given DataFrame.
    Adds various date-related features to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'date' column.

    Returns:
        pd.DataFrame: DataFrame with additional date features.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Quarter'] = df['date'].dt.quarter
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.day_name()
    df['week_of_year'] = df['date'].dt.isocalendar().week

    # Cyclical Features
    df['day_sin'] = np.sin(2 * np.pi * df['Day'] / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df['Day'] / 365.0)
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    df['year_sin'] = np.sin(2 * np.pi * df['Year'] / 7.0)
    df['year_cos'] = np.cos(2 * np.pi * df['Year'] / 7.0)

    # Group Calculation
    df['Group'] = (df['Year'] - 2010) * 48 + df['Month'] * 4 + df['Day'] // 7
    
    return df



train_data = date_feature_engineering(train_data)
test_data = date_feature_engineering(test_data)


train_data.drop('date',axis=1,inplace=True)
test_data.drop('date',axis=1,inplace=True)


print("train data shape :", train_data.shape)
print("test data shape :", test_data.shape)


#train_data.head(2)


train_data['num_sold'] = np.log1p(train_data['num_sold'])


train_data['num_sold'].hist()


train_data = train_data.drop('id', axis = 1)
num_cols = list(train_data.select_dtypes(exclude=['object']).columns.difference(['num_sold']))
cat_cols = list(train_data.select_dtypes(include=['object']).columns)

num_cols_test = list(test_data.select_dtypes(exclude=['object']).columns.difference(['id']))
cat_cols_test = list(test_data.select_dtypes(include=['object']).columns)


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# Initialize LabelEncoder
label_encoders = {col: LabelEncoder() for col in cat_cols}

# Apply LabelEncoder to each categorical column
for col in cat_cols:
    train_data[col] = label_encoders[col].fit_transform(train_data[col])
    test_data[col] = label_encoders[col].transform(test_data[col])
    


train_data = train_data.dropna()
train_data.shape


# Calculate the correlation matrix
correlation_matrix = train_data.corr()

# Plot the heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
X = train_data.drop(['num_sold'], axis=1)
y = train_data['num_sold']
test = test_data.drop(['id'],axis=1)

# Split datainto training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


parameters = {'n_estimators': 3946, 'learning_rate': 0.10203344298643195, 'max_depth': 12, 'num_leaves': 20, 'min_child_samples': 39, 'subsample': 0.7786665459484634, 'colsample_bytree': 0.7352055562065795, 'reg_alpha': 0.2840216195298897, 'reg_lambda': 6.583320975256993, "verbosity" : -1}
#value: 0.7870


# Define MAPE metric
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)*100

# Cross-validation for LGBMRegressor
def cross_val_lgbm_mape(X, y, test, n_splits=7, **parameters):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    preds = []

    for train_index, valid_index in kf.split(X):
        # Ensure data types for indexing
        if isinstance(X, pd.DataFrame):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        else:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

        # Initialize and train the model
        model = LGBMRegressor(random_state=42, **parameters)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_valid)
        score = mape(y_valid, y_pred)
        mape_scores.append(score)

        # Predict on the test set
        preds.append(model.predict(test))

    # Average predictions over all folds
    test_preds_mean = np.mean(preds, axis=0)

    return np.mean(mape_scores), test_preds_mean


average_mape, lgb_preds = cross_val_lgbm_mape(X, y, test, n_splits=7, **parameters)

print(f"Average MAPE across folds: {average_mape:.4f}")

# Save predictions for submission
submission = pd.DataFrame({'id': test_data['id'], 'num_sold': np.expm1(lgb_preds).round()})
print(submission.head())
submission.to_csv('submission_lgb.csv', index=False)



paramsxgb = {'n_estimators': 702, 'learning_rate': 0.1316466004260925, 'max_depth': 6, 'min_child_weight': 5, 'subsample': 0.7085976110203339, 'colsample_bytree': 0.9306214290853707, 'gamma': 0.0006666226876864524, 'reg_alpha': 4.783736210532281, 'reg_lambda': 5.162091591648421}
#Best MAPE: 0.00829858896550044

# Define MAPE metric
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

# Cross-validation for XGBRegressor
def cross_val_xgbr_mape(X, y, test, n_splits=5, **paramsxgb):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    preds = []

    for train_index, valid_index in kf.split(X):
        # Ensure data types for indexing
        if isinstance(X, pd.DataFrame):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        else:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

        # Initialize and train the model
        model = XGBRegressor(random_state=42, **paramsxgb)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_valid)
        score = mape(y_valid, y_pred)
        mape_scores.append(score)

        # Predict on the test set
        preds.append(model.predict(test))

    # Average predictions over all folds
    test_preds_mean = np.mean(preds, axis=0)

    return np.mean(mape_scores), test_preds_mean

# Example usage
model_params = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

average_mape, xgb_preds = cross_val_xgbr_mape(X, y, test, n_splits=5, **paramsxgb)

print(f"Average MAPE across folds: {average_mape:.4f}")

# Save predictions for submission
submission = pd.DataFrame({'id': test_data['id'], 'num_sold': np.expm1(xgb_preds).round()})
print(submission.head())
submission.to_csv('submission_xgb.csv', index=False)



params_cat = {'iterations': 746, 'learning_rate': 0.2702240852251232, 'depth': 7, 'l2_leaf_reg': 0.005010078257154434, 'border_count': 250, 'subsample': 0.6439427477473476, 'random_strength': 4.895400575086264}
#Best MAPE: 0.00834734201702038

# Define MAPE metric
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)*100

# Cross-validation for CatBoostRegressor
def cross_val_catboost_mape(X, y, test, n_splits=5, **params_cat):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    preds = []

    for train_index, valid_index in kf.split(X):
        # Ensure data types for indexing
        if isinstance(X, pd.DataFrame):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        else:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

        # Initialize and train the model
        model = CatBoostRegressor(random_state=42, silent=True, **params_cat)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_valid)
        score = mape(y_valid, y_pred)
        mape_scores.append(score)

        # Predict on the test set
        preds.append(model.predict(test))

    # Average predictions over all folds
    test_preds_mean = np.mean(preds, axis=0)

    return np.mean(mape_scores), test_preds_mean

# Example usage
model_params = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "MAPE"
}

average_mape, catboost_preds = cross_val_catboost_mape(X, y, test, n_splits=5, **params_cat)

print(f"Average MAPE across folds: {average_mape:.4f}")

# Save predictions for submission
submission = pd.DataFrame({'id': test_data['id'], 'num_sold': np.expm1(catboost_preds).round()})
print(submission.head())
submission.to_csv('submission_catboost.csv', index=False)


