import dask.dataframe as dd
df = dd.read_csv("/kaggle/input/microsoft-malware-prediction/train.csv",
                 dtype="object",  # everything loaded as string
                 assume_missing=True)
sample_df = df.sample(frac=0.168, random_state=42).compute()
sample_df.to_csv("trainsample.csv", index=False)
import pandas as pd
train = pd.read_csv("trainsample.csv")
columnstodrop = ['AutoSampleOptIn',
'Census_InternalBatteryNumberOfCharges',
'Census_InternalBatteryType',
'Census_IsFlightingInternal',
'Census_IsFlightsDisabled',
'Census_IsWIMBootEnabled',
'Census_ProcessorClass',
'Census_ThresholdOptIn',
'DefaultBrowsersIdentifier',
'IsBeta',
'ProductName',
'PuaMode',
'UacLuaenable']
train = train.drop(columns=columnstodrop)


import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import early_stopping, log_evaluation



df = train.copy()

target = "HasDetections"
X = df.drop(columns=[target])
y = df[target]



drop_cols = [
    "MachineIdentifier",
    "AvSigVersion",
    "Census_OSVersion",
    "OsBuildLab",
    "AppVersion",
    "EngineVersion"
]

X = X.drop(columns=drop_cols, errors="ignore")



categorical_cols = [col for col in X.columns if X[col].dtype == "object"]

for col in categorical_cols:
    X[col] = X[col].astype("category")



X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)



train_data = lgb.Dataset(
    X_train, 
    label=y_train,
    categorical_feature='auto'
)

val_data = lgb.Dataset(
    X_val, 
    label=y_val,
    categorical_feature='auto'
)



params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,

    "max_bin": 255,
    "min_data_in_bin": 3,

    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,

    "verbose": -1
}



model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=2000,
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=100)
    ]
)


y_pred = model.predict(X_val)
y_pred_binary = (y_pred > 0.5).astype(int)

acc = accuracy_score(y_val, y_pred_binary)
print("Validation Accuracy:", acc)

