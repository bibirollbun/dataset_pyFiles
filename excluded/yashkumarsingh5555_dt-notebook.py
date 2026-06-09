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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
import time



target = "HasDetections"
X = train.drop(columns=[target])
y = train[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


id_cols = ["MachineIdentifier"]  # add more if needed
X_train = X_train.drop(columns=id_cols, errors='ignore')
X_val = X_val.drop(columns=id_cols, errors='ignore')


categorical_cols = X_train.select_dtypes(include="object").columns.tolist()


encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
X_val[categorical_cols]   = encoder.transform(X_val[categorical_cols])


imputer = SimpleImputer(strategy="most_frequent")

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val   = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)


dt = DecisionTreeClassifier(
    max_depth=12,
    min_samples_split=20,
    random_state=42
)
start = time.time()
dt.fit(X_train, y_train)
dt_learn_time = time.time() - start
print("learning time:", dt_learn_time)


start = time.time()
y_pred = dt.predict(X_val)
pred_probs = dt.predict_proba(X_val)[:, 1]
dt_infer_time = time.time() - start
print("inference time:", dt_infer_time)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("AUC:", roc_auc_score(y_val, pred_probs))
print("Log Loss:", log_loss(y_val, pred_probs))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


