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


crypto_submit = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
crypto_trn = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
crypto_txt = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

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
     #missing values
    def total_missing(self):
        missing_vals = self.x.isnull().sum()
        cols_with_missing = missing_vals[missing_vals > 0]
        if not cols_with_missing.empty:
            return cols_with_missing.to_dict()
        else:
            return f"{'No missing values detected'}"
print(f"Training dataset:\n{get_summary(crypto_trn).data_set()}\nTest dataset:\n{get_summary(crypto_txt).data_set()}")
print(f"columns with missing values train\n{get_summary(crypto_trn).total_missing()}\ncolumns with missing values test\n{get_summary(crypto_txt).total_missing()}")


crypto_trn.head(3)


crypto_txt.head(2)


def reduce_mem_usage(dataframe, dataset):    
    print('Reducing memory usage for:', dataset)
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

    return dataframe


trn_crypto = reduce_mem_usage(crypto_trn, "crypto_trn")
txt_crypto = reduce_mem_usage(crypto_txt, "crypto_txt")


import polars as pl
import warnings 
warnings.filterwarnings("ignore", category=RuntimeWarning)

def drop_high_corr_columns(crypto_trn: pd.DataFrame, crypto_txt: pd.DataFrame, threshold: float = 0.99):
  
    # Convert pandas DataFrames to Polars
    train_pl = pl.from_pandas(trn_crypto)
    test_pl = pl.from_pandas(txt_crypto)

    
    corr_df = train_pl.corr()

    
    columns = corr_df.columns
    corr_np = corr_df.to_numpy()

    # Find columns to drop based on upper triangle and threshold
    upper = np.triu(np.ones(corr_np.shape), k=1)
    to_drop = [columns[j] for i in range(corr_np.shape[0])
               for j in range(corr_np.shape[1])
               if upper[i, j] and corr_np[i, j] > threshold]

    to_drop = list(set(to_drop))

    # Drop columns from train and test Polars DataFrames
    train_clean = train_pl.drop(to_drop)
    test_clean = test_pl.drop(to_drop)

    # Convert back to pandas and return
    return train_clean.to_pandas(), test_clean.to_pandas()


trn_crypto, txt_crypto = drop_high_corr_columns(trn_crypto, txt_crypto, threshold=0.99)
trn_crypto.head(3)


txt_crypto.drop('label', axis=1, inplace=True)
txt_crypto.replace([np.inf, -np.inf], np.nan, inplace=True)
txt_crypto.head(2)


from sklearn.model_selection import train_test_split

X = trn_crypto.drop('label', axis=1)
X.replace([np.inf, -np.inf], np.nan, inplace=True)

y = trn_crypto['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=12, test_size=0.30)


from xgboost import XGBRegressor
from scipy.stats import pearsonr


xgb_params = {
    "colsample_bylevel": 0.4778015829774066,
    "colsample_bynode": 0.362764358742407,
    "colsample_bytree": 0.7107423488010493,
    "gamma": 1.7094857725240398,
    "learning_rate": 0.02213323588455387,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "n_jobs": -1,
    "random_state": 12,
    "reg_alpha": 39.352415706891264,
    "reg_lambda": 75.44843704068275,
    "subsample": 0.06566669853471274,
    "verbosity": 0
}


model = XGBRegressor(**xgb_params, missing=np.nan)

model.fit(X_train, y_train)
model_pred = model.predict(X_val)
scr = pearsonr(y_val, model_pred)
print("Final Pearson Correlation = ", scr)


prediction = model.predict(txt_crypto)
prediction


crypto_submit.head(2)


submission = crypto_submit
submission['prediction'] = prediction


submission.to_csv("submission.csv", index=False)

