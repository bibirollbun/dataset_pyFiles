import pandas as pd
import gc
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score
from scipy.stats import pearsonr
from IPython.display import display
import math
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')


train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')
train_df.head(3)


train_df.replace([np.inf, -np.inf], 0, inplace=True)


train_df.shape


train_df = train_df.loc[:,~train_df.columns.duplicated()].copy()


_train_df = train_df.iloc[:500000]
_test_df = train_df.iloc[500000:]


x_train = _train_df.drop(columns = ['label'])
y_train = _train_df['label']
x_test = _test_df.drop(columns = ['label'])
y_test = _test_df['label']


del _train_df
del train_df
gc.collect()


%system free -m


# Random Forest
def get_most_important_features_by_rf(x_train, y_train, top_n=20):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': x_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    return imp_df

# XGBoost
def get_most_important_features_by_xgb(x_train, y_train, top_n=20):
    model = xgb.XGBRegressor()
    model.fit(x_train, y_train)
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': x_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    return imp_df

# Linear Regression
def get_most_important_features_by_lin_reg(x_train, y_train, top_n=20):
    model = LinearRegression()
    model.fit(x_train, y_train)
    coef = np.abs(model.coef_)
    imp_df = pd.DataFrame({
        'Feature': x_train.columns,
        'Coefficient': coef
    }).sort_values(by='Coefficient', ascending=False)
    return imp_df

# Logistic Regression
def get_most_important_features_by_log_reg(x_train, y_train, top_n=20):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    coef = np.abs(model.coef_[0])
    imp_df = pd.DataFrame({
        'Feature': x_train.columns,
        'Coefficient': coef
    }).sort_values(by='Coefficient', ascending=False)
    return imp_df

# getting correlations and ranking them
def get_correlations_and_rank_them(df):
    corr_data = {
        'x': [],
        'y': [],
        'corr': []
    }
    cols_list = df.columns.tolist()
    for i in tqdm(range(len(cols_list)), desc="Calculating correlations..."):
        for j in range(i+1, len(cols_list)):
            x_col = cols_list[i]
            y_col = cols_list[j]
            x = df[x_col]
            y = df[y_col]
            corr = np.corrcoef(x, y)[0, 1]
            corr_data['x'].append(x_col)
            corr_data['y'].append(y_col)
            corr_data['corr'].append(corr)

    corr_df = pd.DataFrame(corr_data)
    corr_df = corr_df.sort_values(by='corr', ascending=False)
    return corr_df

# run all paralally - use all CPUs
with ThreadPoolExecutor() as executor:
    future_rf = executor.submit(get_most_important_features_by_rf, x_train, y_train)
    future_xgb = executor.submit(get_most_important_features_by_xgb, x_train, y_train)
    # future_lin = executor.submit(get_most_important_features_by_lin_reg, x_train, y_train)
    # future_log = executor.submit(get_most_important_features_by_log_reg, x_train, y_train)
    # future_corr = executor.submit(get_correlations_and_rank_them, x_train)
    MIF_rf = future_rf.result()
    MIF_xgb = future_xgb.result()
    # MIF_lin_reg = future_lin.result()
    # MIF_log_reg = future_log.result()
    # MIF_corr = future_corr.result()


MIF_rf.to_csv('MIF_rf.csv')
MIF_xgb.to_csv('MIF_xgb.csv')
# MIF_lin_reg.to_csv('MIF_lin_reg.csv')
# MIF_log_reg.to_csv('MIF_log_reg.csv')
# MIF_corr.to_csv('MIF_corr.csv')

