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


traindf = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
testdf = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
print(traindf.shape, testdf.shape)


!pip install xgboost==3.1.1



import xgboost as xgb
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings("ignore")


traindf = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',)
testdf = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


def add_new_features(df):
    df['road_weather'] = df['road_type'].astype(str) + '_' + df['weather'].astype(str)
    df['road_light'] = df['road_type'].astype(str) + '_' + df['lighting'].astype(str)
    df['weather_light'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
    df['road_time'] = df['road_type'].astype(str) + '_' + df['time_of_day'].astype(str)
    df['weather_time'] = df['weather'].astype(str) + '_' + df['time_of_day'].astype(str)

    return df

traindf = add_new_features(traindf)
testdf = add_new_features(testdf)


num = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
boolean = ['road_signs_present','public_road','holiday','school_season']
cat = ['road_type','lighting','weather','time_of_day', 'road_weather', 'road_light', 'weather_light', 'road_time', 'weather_time']

# convert categorical columns to category type
for col in cat:
    for data in [traindf, testdf]:
        data[col] = data[col].astype("category")


X = traindf.drop(['id','accident_risk'], axis=1)
y = traindf['accident_risk']

testdf = testdf.copy().drop(columns=['id'], axis=1)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    
    "n_estimators": 10_000,
    "max_depth": 7,
    "min_samples_split": 5,
    "random_state": 42,
    'learning_rate': 0.005, 
    
    # 'min_child_weight': 9, 
    'enable_categorical': True,
    'enable_categorical': True,
    'n_jobs': -1,

    # 'gamma': 0.13, 
    'subsample': 0.7, 
    'colsample_bytree': 0.8, 
    'lambda': 2.0, 
    'alpha': 1.0,
    "tree_method": "hist",
    "device": "cuda",
    
    'seed': 42,
    # 'callbacks':[early_stop]
    }


seeds = [42, 128, 256, 510, 3000]

cols = [f"seed_{seed}" for seed in seeds]
oof_xgb_full = []
xgb_preds_full = []

for seed in seeds:
    SEED = seed
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Out-of-fold (OOF) predictions for blending on the training data
    oof_xgb = np.zeros(X.shape[0])
    
    # Predictions on the test set (on the ORIGINAL target scale)
    xgb_preds = np.zeros(testdf.shape[0])

    print(f"\nTraining xgboost with {seed}")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nTraining fold {fold + 1}/{n_splits}")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        early_stop = xgb.callback.EarlyStopping(rounds=300, metric_name='rmse', data_name='validation_0')
        
        # Ensure params uses the newly created callback object
        local_params = params.copy()
        local_params['callbacks'] = [early_stop]
        local_params["seed"] = seed
        
        model = xgb.XGBRegressor(**local_params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)], 
            verbose= 500,
        )
        
        oof_xgb[val_idx] = model.predict(X_val)
        
        test_preds = model.predict(testdf)

        xgb_preds += test_preds / n_splits
        
        fold_mse = mean_squared_error(y_val, oof_xgb[val_idx])
        print(f"----> XGBoost Fold {fold + 1} validation MSE (Original Scale): {fold_mse:.6f}")

    oof_xgb_full.append(oof_xgb)
    xgb_preds_full.append(xgb_preds)


oof_xgb_full_array = np.array(oof_xgb_full)
xgb_preds_full_array = np.array(xgb_preds_full)

# transpose the array to swap rows (seeds) and columns (samples)
oof_xgb_full_df = pd.DataFrame(data=oof_xgb_full_array.T, columns=cols, index=X.index) 
xgb_preds_full_df = pd.DataFrame(data=xgb_preds_full_array.T, columns=cols, index=testdf.index)

# check the performance of ensembling results

for col in oof_xgb_full_df.columns.tolist():
    rmse = np.sqrt(mean_squared_error(y, oof_xgb_full_df[col]))
    mse = mean_squared_error(y, oof_xgb_full_df[col])
    mae = mean_absolute_error(y, oof_xgb_full_df[col])
    
    print(f"Column: {col}")
    print(f"RMSE: {rmse}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}\n")


oof_mean = oof_xgb_full_df.mean(axis=1)
preds_mean = xgb_preds_full_df.mean(axis=1)

print(f"Mean value of all predictions\nRMSE: {np.sqrt(mean_squared_error(y, oof_mean))}\nMSE: {mean_squared_error(y, oof_mean)}\nMAE: {mean_absolute_error(y, oof_mean)}")


submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission['accident_risk'] = preds_mean
submission.to_csv('submission.csv', index=False) 
submission.head()


#submission["Net RR"] = df_ipl["Net RR"].round(3)


