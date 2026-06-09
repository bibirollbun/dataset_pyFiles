#!pip install /kaggle/input/lifelines-new/lifelines-0.30.0-py3-none-any.whl


# !pip install lifelines -q


# !mkdir wheelfactory
# !cd wheelfactory


# !pip wheel lifelines


# !zip -r factory.zip ./wheelfactory


# !mv *.whl wheelfactory


# !unzip /kaggle/input/factory -d ../wheelfactory


# !ls /kaggle/input/wheelfactory


!pip install /kaggle/input/factory/wheelfactory/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/factory/wheelfactory/autograd_gamma-0.5.0-py3-none-any.whl
!pip install /kaggle/input/factory/wheelfactory/contourpy-1.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install /kaggle/input/factory/wheelfactory/cycler-0.12.1-py3-none-any.whl
!pip install /kaggle/input/factory/wheelfactory/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/factory/wheelfactory/fonttools-4.55.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install /kaggle/input/factory/wheelfactory/formulaic-1.1.1-py3-none-any.whl
!pip install /kaggle/input/factory/wheelfactory/kiwisolver-1.4.8-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl
!pip install /kaggle/input/factory/wheelfactory/packaging-24.2-py3-none-any.whl
!pip install /kaggle/input/factory/wheelfactory/pillow-11.1.0-cp310-cp310-manylinux_2_28_x86_64.whl
!pip install /kaggle/input/factory/wheelfactory/pyparsing-3.2.1-py3-none-any.whl
!pip install /kaggle/input/factory/wheelfactory/python_dateutil-2.9.0.post0-py2.py3-none-any.whl
!pip install /kaggle/input/factory/wheelfactory/pytz-2024.2-py2.py3-none-any.whl
!pip install /kaggle/input/factory/wheelfactory/six-1.17.0-py2.py3-none-any.whl
!pip install /kaggle/input/factory/wheelfactory/typing_extensions-4.12.2-py3-none-any.whl
!pip install /kaggle/input/factory/wheelfactory/tzdata-2024.2-py2.py3-none-any.whl
!pip install /kaggle/input/factory/wheelfactory/wrapt-1.17.2-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install /kaggle/input/factory/wheelfactory/lifelines-0.30.0-py3-none-any.whl


# pip install lifelines -q 


import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
import warnings
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import HistGradientBoostingRegressor


warnings.filterwarnings('ignore')

# ==============================
# Data Loading and Preprocessing
# ==============================

# Load datasets
train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv', index_col='ID')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv', index_col='ID')
sub = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv', index_col='ID')

# One-hot encode categorical features
train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

# Align train and test sets to ensure they have the same columns
train, test = train.align(test, join='left', axis=1, fill_value=0)
# Clean column names to remove invalid characters
train.columns = train.columns.str.replace('[\\[\\]<]', '', regex=True)
test.columns = test.columns.str.replace('[\\[\\]<]', '', regex=True)
test = test.drop(['efs', 'efs_time', 'naf_label'], axis=1, errors='ignore')

# Nelson-Aalen Label Transformation
naf = NelsonAalenFitter()
naf.fit(train['efs_time'], train['efs'])
train['naf_label'] = -naf.cumulative_hazard_at_times(train['efs_time']).values
train.loc[train['efs'] == 0, 'naf_label'] -= 0.1
# Define evaluation metric
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Placeholder for results
kf = KFold(n_splits=5, shuffle=True, random_state=42)



catboost_predictions = np.zeros(test.shape[0])
catboost_oof = np.zeros(train.shape[0])

catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'RMSE',
    'verbose': 100,
    'random_seed': 42
}

for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print(f"CatBoost Fold {fold + 1}")
    X_train, X_valid = train.iloc[train_idx].drop(['efs', 'efs_time', 'naf_label'], axis=1), train.iloc[valid_idx].drop(['efs', 'efs_time', 'naf_label'], axis=1)
    y_train, y_valid = train.iloc[train_idx]['naf_label'], train.iloc[valid_idx]['naf_label']

    # Create CatBoost Pools
    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)

    # Train CatBoost Model
    model = CatBoostRegressor(**catboost_params)
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50)

    catboost_oof[valid_idx] = model.predict(X_valid)
    catboost_predictions += model.predict(test) / kf.n_splits



hgb_predictions = np.zeros(test.shape[0])
hgb_oof = np.zeros(train.shape[0])

hgb_params = {
    'max_iter': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'early_stopping': True,
    'random_state': 42
}

for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print(f"HistGradientBoosting Fold {fold + 1}")
    X_train, X_valid = train.iloc[train_idx].drop(['efs', 'efs_time', 'naf_label'], axis=1), train.iloc[valid_idx].drop(['efs', 'efs_time', 'naf_label'], axis=1)
    y_train, y_valid = train.iloc[train_idx]['naf_label'], train.iloc[valid_idx]['naf_label']

    # Train HistGradientBoosting Model
    model = HistGradientBoostingRegressor(**hgb_params)
    model.fit(X_train, y_train)

    hgb_oof[valid_idx] = model.predict(X_valid)
    hgb_predictions += model.predict(test) / kf.n_splits



import xgboost as xgb

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

xgb_predictions = np.zeros(test.shape[0])
xgb_oof = np.zeros(train.shape[0])


for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print(f"XGBoost Fold {fold + 1}")
    X_train, X_valid = train.iloc[train_idx].drop(['efs', 'efs_time', 'naf_label'], axis=1), train.iloc[valid_idx].drop(['efs', 'efs_time', 'naf_label'], axis=1)
    y_train, y_valid = train.iloc[train_idx]['naf_label'], train.iloc[valid_idx]['naf_label']

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(test)

    model = xgb.train(xgb_params, dtrain, num_boost_round=1000,
                      evals=[(dtrain, 'train'), (dvalid, 'valid')],
                      early_stopping_rounds=50, verbose_eval=100)

    xgb_oof[valid_idx] = model.predict(dvalid)
    xgb_predictions += model.predict(dtest) / kf.n_splits




# Compute RMSE for OOF predictions
catboost_rmse = rmse(train['naf_label'], catboost_oof)
hgb_rmse = rmse(train['naf_label'], hgb_oof)
xgb_rmse = rmse(train['naf_label'], xgb_oof)

# Print OOF RMSE
print(f"CatBoost OOF RMSE: {catboost_rmse:.4f}")
print(f"HistGradientBoosting OOF RMSE: {hgb_rmse:.4f}")
print(f"XGBoost OOF RMSE: {xgb_rmse:.4f}")



# Simple average ensemble
ensemble_predictions = (catboost_predictions + hgb_predictions + xgb_predictions) / 3

# Save predictions
sub['prediction'] = ensemble_predictions
sub.to_csv('submission.csv')
print("Submission file saved.")

















































