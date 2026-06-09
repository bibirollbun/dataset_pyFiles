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


sticker_sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
sticker_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
sticker_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


class get_summary:
    def __init__(self, x):
        self.x = x if isinstance(x, pd.DataFrame) else pd.DataFrame()
    def data_set(self):
        #checks for duplicate
        duplicate = self.x.duplicated().any()
        #drop duplicates 
        if duplicate == True:
            self.x.drop_duplicates(inplace=True)
            self.x.reset_index(drop=True)
        #checks for empty values
        null = self.x.isna().sum().any()
        #missing values
        total_missing = self.x.isnull().sum().sum()
        #data types
        data_type = self.x.dtypes
        #shape
        shapes = self.x.shape
        return f"Duplicate: {duplicate}\nNull: {null}\nMissing_value: {total_missing}\nTypes:\n{data_type}\nShape: {shapes}"
    
    def total_missing(self):
        missing_vals = self.x.isnull().sum()
        cols_with_missing = missing_vals[missing_vals > 0]
        return cols_with_missing.to_dict()
print(f"Training dataset:\n{get_summary(sticker_train).data_set()}\nTest dataset:\n{get_summary(sticker_test).data_set()}")
print(f"columns with missing values train\n{get_summary(sticker_train).total_missing()}\ncolumns with missing values test\n{get_summary(sticker_test).total_missing()}")


sticker_train.dropna(subset=['num_sold'], inplace=True)


'''our test data is clean with no missing values but the train data contain
some missing values at the target. The NaN values will be filled and the date
column will be splited for basic analysis'''

def data_eng(df):
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

    df['Group'] = (df['Year'] - 2010) * 48 + df['Month'] * 4 + df['Day'] // 7
    return df.head(3)
    
data_eng(sticker_train)


data_eng(sticker_test)


import seaborn as sns
import matplotlib.pyplot as plt


#this function shows the product distribution globally
def product_distribution(sticker_train):
    plt.pie(sticker_train.value_counts(),
            labels=sticker_train.unique(),
            autopct='%.2f%%',
            shadow=True,
            explode=[0.1, 0.01, 0.1, 0.2, 0.01],
            colors=['green', 'blue', 'brown', 'purple', 'violet'],
            pctdistance=0.5)
    plt.title('Global distribution of product')
    plt.show()
product_distribution(sticker_train['product'])



#distribution of product across countries
plt.figure(figsize=(12, 5))
product_dist_country = sns.barplot(sticker_train, x='country', y='num_sold', hue='product')
plt.legend()

for perc in product_dist_country.patches: 
    height = perc.get_height()
    percentage = f"{height / sum(sticker_train['num_sold']) * 100:.1f}%"
    product_dist_country.text(perc.get_x() + perc.get_width() / 2., 
                 height + 0.5, f'{height:.1f}%', 
                 ha="center", rotation=80)
plt.title('Distribution by country')
plt.show()


plt.figure(figsize=(12, 5))
sns.lineplot(sticker_train, x='Year', y='num_sold')
plt.title('annual sales')
plt.show()


plt.figure(figsize=(12, 5))
sns.barplot(sticker_train, x='Month', y='num_sold')
plt.title('Monthly sales')
plt.show()


from sklearn.preprocessing import OrdinalEncoder 
from sklearn.preprocessing import StandardScaler 


features = sticker_train.drop(['id', 'date', 'num_sold'], axis=1)
target = sticker_train['num_sold']


X_test = sticker_test.drop(['id', 'date'], axis=1)
X_test.head(2)


def encode(x):
    for cols in x.columns:
        enc = OrdinalEncoder()
        if x[cols].dtype == 'object':
            x[[cols]] = enc.fit_transform(x[[cols]])
    return x.head(2)
encode(features)


encode(X_test)


scala = StandardScaler()
X = pd.DataFrame(scala.fit_transform(features))
X.head(2)


X_scaled_test = pd.DataFrame(scala.fit_transform(X_test))
X_scaled_test.head(2)


from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error


X_train, X_val, target_train, target_val = train_test_split(X, 
                                                            target,
                                                            random_state=12,
                                                            test_size=0.2)


params = {'n_estimators': 550, 
          'learning_rate': 0.01316466004260925, 
          'max_depth': 10, 
          'min_child_weight': 5, 
          'subsample': 0.7085976110203339, 
          'colsample_bytree': 0.9306214290853707
         }

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

def cross_val_xgbr_mape(X, target, X_scaled_test, n_splits=5, **params):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=12)
    mape_scores = []
    preds = []

    for train_index, valid_index in kf.split(X):
        # Ensure data types for indexing
        if isinstance(X, pd.DataFrame):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            target_train, target_valid = target.iloc[train_index], target.iloc[valid_index]
        else:
            X_train, X_valid = X[train_index], X[valid_index]
            target_train, target_valid = target[train_index], target[valid_index]

    # Initialize and train the model
        model = xgb.XGBRegressor(random_state=12, **params)
        model.fit(X_train, target_train)

        # Predictions and evaluation
        y_pred = model.predict(X_valid)
        score = mape(target_valid, y_pred)
        mape_scores.append(score)

        # Predict on the test set
        preds.append(model.predict(X_scaled_test))

        # Average predictions over all folds
        test_preds_mean = np.mean(preds, axis=0)

        return np.mean(mape_scores), test_preds_mean

# Example usage
model_params = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

average_mape, xgb_preds = cross_val_xgbr_mape(X, target, X_scaled_test, n_splits=5, **params)

print(f"Average MAPE across folds: {average_mape:.4f}")


# Save predictions for submission
submission = pd.DataFrame({'id': sticker_test['id'], 'num_sold': np.round(xgb_preds)})
print(submission.head(4))
submission.to_csv('submission.csv', index=False)

