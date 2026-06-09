import pandas as pd
train = pd.read_csv("/kaggle/input/trainsample-csv/trainsample.csv")
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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import time



df = train.copy()
target = "HasDetections"

X = df.drop(columns=[target])
y = df[target]


categorical_cols = [col for col in X.columns if X[col].dtype == "object"]

for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Encode target if needed
if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)


params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "gpu_hist",    # Use GPU
    "gpu_id": 0,
    "max_depth": 10,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "alpha": 1,
    "lambda": 1
}


evals = [(dtrain, "train"), (dval, "valid")]
start = time.time()

model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=evals,
    early_stopping_rounds=100,
    verbose_eval=100
)

xgb_train_time = time.time() - start


print("\nLearning time:", xgb_train_time)


from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
import numpy as np
import time
start=time.time()
y_pred = model.predict(dval)


y_pred_binary = (y_pred > 0.5).astype(int) 

xgb_infer_time = start-time.time()
print("Inference time:",xgb_infer_time)



accuracy = accuracy_score(y_val, y_pred_binary)
print("Validation Accuracy:", accuracy)


auc_score = roc_auc_score(y_val, y_pred)
print("Validation AUC:", auc_score) 


logloss = log_loss(y_val, y_pred)
print("Validation Log Loss:", logloss)


print("\nClassification Report:\n", classification_report(y_val, y_pred_binary))


xgb.plot_importance(model, max_num_features=30, height=0.4)
plt.show()

