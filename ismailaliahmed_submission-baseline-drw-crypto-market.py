# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Garbage Collector
import gc 

import pandas as pd
import numpy as np
import os

# Time Modules
import calendar
import time
import datetime
from datetime import datetime, timedelta

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Plots

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(18, 12)})
%matplotlib inline

# Statistics 
from scipy.stats import norm
from scipy.stats import zscore
from scipy import stats

import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import lightgbm as lgb
from xgboost import XGBRegressor


import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


##################################################################
# Installing GPU driver for LightGBM:-
!mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
!sudo apt install nvidia-driver-460 nvidia-cuda-toolkit clinfo
!apt-get update --fix-missing
!pip install -q  lightgbm==4.1.0 \
  --config-settings=cmake.define.USE_GPU=ON \
  --config-settings=cmake.define.OpenCL_INCLUDE_DIR="/usr/local/cuda/include/" \
  --config-settings=cmake.define.OpenCL_LIBRARY="/usr/local/cuda/lib64/libOpenCL.so"


from scipy.stats import pearsonr


main_columns = ['bid_qty','ask_qty','buy_qty','sell_qty','volume','label']


main_features = [
#  'X420','X726','X724','X124','X338','X137','X337','X334','X175',
# 'X184','X721','X86','X39','X580','X371','X415','X325','X125',
# 'X4','X531','X149','X120','X662','X426','X219','X728','X281','X279',
# 'X245','X230','X271','X674','X256','X244','X247','X232','X210','X218',
# 'X729','X119','X123','X136','X156','X161','X162','X167',
# 'X169','X174','X188','X195','X207','X286','X213','X215','X730','X284',
# 'X651','X574','X398','X404','X565','X561','X560','X414','X421', 'X430',
# 'X443','X445','X530','X457','X524','X464','X465','X473','X509','X505',
 'X492','X396','X391','X649','X386','X301','X303','X637','X619','X112',
'bid_qty','ask_qty','buy_qty','sell_qty','volume','label'
]
len(sorted(main_features))


y = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet').label.values
y[0:10]


import pandas as pd
import numpy as np

class Processor:
    def __init__(self, train_path, test_path, main_features):
        self.train = pd.read_parquet(train_path, columns=main_features)
        self.test = pd.read_parquet(test_path, columns=main_features)
        self.indexes = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv').index
        self.target = np.array(self.train.label)

    def reduce_mem_usage(self, dataframe, dataset_name=""):
        """
        Function taken from: https://www.kaggle.com/code/ravaghi/drw-crypto-market-prediction-ensemble
        Reduces memory usage of a DataFrame by downcasting numeric types.
        """
        print('Reducing memory usage for:', dataset_name)
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
            elif str(col_type)[:5] == 'float':
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


processor = Processor(
    train_path='/kaggle/input/drw-crypto-market-prediction/train.parquet',
    test_path='/kaggle/input/drw-crypto-market-prediction/test.parquet',
    main_features = main_features
)


processor.train = processor.reduce_mem_usage(processor.train, "Train Dataset")
processor.test = processor.reduce_mem_usage(processor.test, "Test Dataset")


from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

class FeatureEngineering:
    def __init__(self, main_columns=None, n_components=50, batch_size=100):
        self.main_columns = main_columns
        self.n_components = n_components
        self.batch_size = batch_size
        self.ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        self.scaler = StandardScaler()
        self.fitted = False
        self.scaler_fitted = False

    def transform_columns(self, df):
        df = df.copy()

        for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.replace(0, np.nan, inplace=True)

        # Binary indicators
        df['bid_ask_binary'] = np.where(df['bid_qty'] > df['ask_qty'], 1, 0)
        df['buy_sell_binary'] = np.where(df['buy_qty'] > df['sell_qty'], 1, 0)

        df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
        df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
        df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
        df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
        df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
        df['buy_sell_interaction'] = df['buy_qty'] * df['sell_qty']

        df['spread_indicator'] = (df['ask_qty'] - df['bid_qty']) / (df['ask_qty'] + df['bid_qty'] + 1e-8)

        df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
        df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
        df['volume_weighted_bid'] = df['bid_qty'] * df['volume']
        df['volume_weighted_ask'] = df['ask_qty'] * df['volume']

        df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
        df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)

        df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)

        df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-8)
        df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)

        df['total_liquidity'] = df['bid_qty'] + df['ask_qty']
        df['liquidity_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_liquidity'] + 1e-8)
        df['relative_spread'] = (df['ask_qty'] - df['bid_qty']) / (df['volume'] + 1e-8)

        df['trade_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-8)
        df['avg_trade_size'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
        df['net_trade_flow'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)

        df['depth_ratio'] = df['total_liquidity'] / (df['volume'] + 1e-8)
        df['volume_participation'] = (df['buy_qty'] + df['sell_qty']) / (df['total_liquidity'] + 1e-8)
        df['market_activity'] = df['volume'] * df['total_liquidity']

        df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
        df['realized_volatility_proxy'] = np.abs(df['order_flow_imbalance']) * df['volume']

        df['normalized_buy_volume'] = df['buy_qty'] / (df['bid_qty'] + 1e-8)
        df['normalized_sell_volume'] = df['sell_qty'] / (df['ask_qty'] + 1e-8)

        df['liquidity_adjusted_imbalance'] = df['order_flow_imbalance'] * df['depth_ratio']
        df['pressure_spread_interaction'] = df['buying_pressure'] * df['spread_indicator']

        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

        return df

    def drop_columns(self, df):
        df = df.copy()
        if self.main_columns is not None:
            missing_cols = [col for col in self.main_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: columns {missing_cols} not found in DataFrame.")
            df = df.drop(columns=[col for col in self.main_columns if col in df.columns])
        return df

    def generate_lag_features(self, df, cols, lags=[1,3,5,7,10,20,60,120,180,240,60*24,60*24*2,60*24*3,60*24*4,60*24*5], windows=[60*7, 60*14], dropna=True):
        df = df.copy()

        for col in cols:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

            for window in windows:
                df[f'{col}_roll_mean_{window}'] = df[col].shift(window).ewm(halflife=window).mean()
                df[f'{col}_roll_std_{window}'] = df[col].shift(window).ewm(halflife=window).std()

        if dropna:
            df.dropna(inplace=True)
        else:
            df.replace([np.inf, -np.inf], 0, inplace=True)
            df.fillna(0, inplace=True)

        return df

    def fit_scaler(self, df):
        df = df.copy()
        self.scaler.fit(df)
        self.scaler_fitted = True

    def transform_scaler(self, df):
        if not self.scaler_fitted:
            raise RuntimeError("Scaler is not fitted. Call fit_scaler() on training data first.")
        df_scaled = self.scaler.transform(df)
        return pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

    def fit_ipca(self, df):
        df = df.copy()
        self.ipca.fit(df)
        self.fitted = True

    def transform_ipca(self, df):
        if not self.fitted:
            raise RuntimeError("IPCA model is not fitted. Call fit_ipca() on training data first.")
        df_ipca = self.ipca.transform(df)
        df_ipca = pd.DataFrame(df_ipca, columns=[f'ipca_{i}' for i in range(self.n_components)], index=df.index)
        return df_ipca



# Define main columns to remove later (raw input features)
main_columns = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'label']

fe = FeatureEngineering()

train = fe.transform_columns(processor.train)
test = fe.transform_columns(processor.test)

# Add lag features to e.g. 'volume'
train = fe.generate_lag_features(train, cols=main_columns, lags=[60*7, 60*14, 60*28], windows=[60*7, 60*14], dropna=False)
test = fe.generate_lag_features(test, cols=main_columns, lags=[60*7, 60*14, 60*28], windows=[60*7, 60*14], dropna=False)


fe.fit_scaler(train)
fe.fit_scaler(test)

train = fe.transform_scaler(train)
test = fe.transform_scaler(test)

fe.fit_ipca(train)
fe.fit_ipca(train)

train = fe.transform_ipca(train)
test = fe.transform_ipca(test)


train = fe.drop_columns(train)
test = fe.drop_columns(test)


train.shape


train.head(5)


test.shape


xgb_params = {
    "tree_method": "hist",
    "device": "gpu",
    'metric': ['l1', 'l2'],
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "subsample": 0.06567,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "random_state": 700,
    "verbose": False,
}


params_gdbt = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l1', 'l2'],
    'seed':42,
    'device': 'gpu',
    'learning_rate': 0.06459880558047476,
    'n_estimators': 1196,
    'max_depth': 8, 
    'num_leaves': 483,
    'min_child_samples': 172, 
    'subsample': 0.10298004227879802, 
    'colsample_bytree': 0.9034687230448682, 
    'reg_alpha': 0.7684295974829274, 
    'reg_lambda': 0.49761953142451365,
    'verbose':-1,
    'verbosity':0,
    'predict_disable_shape_check':True,


}


params_goss = {
    'boosting_type': 'goss',
    'objective': 'regression',
    'metric': ['l1', 'l2'],
    'seed':42,
    'device': 'gpu',
    'learning_rate': 0.06459880558047476,
    'n_estimators': 1196,
    'max_depth': 8, 
    'num_leaves': 483,
    'min_child_samples': 172, 
    'subsample': 0.10298004227879802, 
    'colsample_bytree': 0.9034687230448682, 
    'reg_alpha': 0.7684295974829274, 
    'reg_lambda': 0.49761953142451365,
    'verbose':-1,
    'verbosity':0,
    'predict_disable_shape_check':True,

}


goss = lgb.LGBMRegressor(**params_goss)
gbdt = lgb.LGBMRegressor(**params_gdbt)
model_xgb = XGBRegressor(**xgb_params)


import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold, TimeSeriesSplit, GroupKFold
import lightgbm as lgb
from xgboost import XGBClassifier

# Parameters
n_splits = 10
gkf = GroupKFold(n_splits=n_splits)
groups = np.array(train.ipca_5)

# Initialize predictions
oof_gbdt = np.zeros(len(train))
oof_goss = np.zeros(len(train))
oof_xgb  = np.zeros(len(train))

test_preds_gbdt = np.zeros(len(test))
test_preds_goss = np.zeros(len(test))
test_preds_xgb  = np.zeros(len(test))

# CV loop
for fold, (train_idx, val_idx) in enumerate(gkf.split(train, y, groups)):
    print(f"\nðŸŒ€ Fold {fold + 1}/{n_splits}")

    X_train_fold, X_val_fold = train.iloc[train_idx], train.iloc[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    ### LightGBM GBDT
    print("Training LightGBM (GBDT)...")
    gbdt.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=10)
        ]
    )
    oof_gbdt[val_idx] = gbdt.predict(X_val_fold)
    test_preds_gbdt += gbdt.predict(test) / n_splits

    ### LightGBM GOSS
    print("Training LightGBM (GOSS)...")
    goss.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=10)
        ]
    )
    oof_goss[val_idx] = goss.predict(X_val_fold)
    test_preds_goss += goss.predict(test) / n_splits

    ### XGBoost
    print("Training XGBoost...")
    model_xgb.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        early_stopping_rounds=100,
        verbose=False
    )
    oof_xgb[val_idx] = model_xgb.predict(X_val_fold)
    test_preds_xgb += model_xgb.predict(test) / n_splits

# Create submission files
for name, preds in zip(
    ["gbdt", "goss", "xgb"],
    [test_preds_gbdt, test_preds_goss, test_preds_xgb]
):
    submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
    submission["prediction"] = preds
    submission.to_csv(f"submission_{name}.csv", index=False)
    print(f"âœ… Saved: submission_{name}.csv")



# total_weight = 1 / avg_pc_gbdt + 1 / avg_pc_goss + 1 / avg_pc_xgb
# weight_gbdt = (1 / avg_pc_gbdt) / total_weight
# weight_goss = (1 / avg_pc_goss) / total_weight
# weight_xgb = (1 / avg_pc_xgb) / total_weight


# final_test_predictions = (
#     weight_gbdt * test_predictions_gbdt.mean(axis=1) +
#     weight_goss * test_predictions_goss.mean(axis=1) + 
#     weight_xgb * test_predictions_xgb.mean(axis=1)
# )


import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Parameters
n_splits = 10

# Placeholders for OOF and test predictions
oof_preds = {
    "gbdt": np.zeros(len(train)),
    "goss": np.zeros(len(train)),
    "xgb": np.zeros(len(train)),
    "cat": np.zeros(len(train)),
}
test_preds = {
    "gbdt": np.zeros((len(test), n_splits)),
    "goss": np.zeros((len(test), n_splits)),
    "xgb": np.zeros((len(test), n_splits)),
    "cat": np.zeros((len(test), n_splits)),
}
fold_pc = {k: [] for k in oof_preds}

# Model definitions
models = {
    "gbdt": lgb.LGBMRegressor(**params_gdbt),
    "goss": lgb.LGBMRegressor(**params_goss),
    "xgb": xgb.XGBRegressor(**xgb_params),
    "cat": cb.CatBoostRegressor(iterations=1000, learning_rate=0.05, verbose=0)
}

# Time series CV training
for fold, (train_idx, val_idx) in enumerate(gkf.split(train, y, groups)):
    print(f"\nFold {fold + 1}/{n_splits}")
    
    X_train_fold = train.iloc[train_idx]
    X_val_fold = train.iloc[val_idx]
    y_train_fold = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
    y_val_fold = y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]

    for name, model in models.items():
        print(f" Training {name.upper()}...")

        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=10)
            ] if 'lgb' in str(type(model)) else None
        )

        # OOF and test predictions
        oof_preds[name][val_idx] = model.predict(X_val_fold)
        test_preds[name][:, fold] = model.predict(test)
        
        # Pearson correlation
        pc = pearsonr(y_val_fold, oof_preds[name][val_idx])[0]
        fold_pc[name].append(pc)
        print(f"  â†’ Fold Pearson: {pc:.4f}")

# Show average PC
for name in oof_preds:
    avg_pc = np.mean(fold_pc[name])
    print(f"\n{name.upper()} Average Pearson Correlation: {avg_pc:.4f}")



# Combine OOF predictions into stacking features
X_stack_train = np.column_stack([oof_preds[k] for k in oof_preds])
X_stack_test = np.column_stack([test_preds[k].mean(axis=1) for k in test_preds])

# Meta-model: RidgeCV with CV internally
print("\nTraining Meta-Model (RidgeCV)...")
meta_model = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
meta_model.fit(X_stack_train, y)

# Final stacked predictions
final_oof = meta_model.predict(X_stack_train)
final_test = meta_model.predict(X_stack_test)

# Evaluation
final_pc = pearsonr(y, final_oof)[0]

print(f"\n=== FINAL STACKED MODEL ===")
print(f"Stacked Model Pearson Correlation: {final_pc:.4f}")


submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
submission["prediction"] = final_test
submission.to_csv("submission_meta_model.csv", index=False)

