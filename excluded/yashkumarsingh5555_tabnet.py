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


!pip install pytorch-tabnet -q

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.impute import SimpleImputer
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

print("PyTorch CUDA Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))

target = "HasDetections"
X = train.drop(columns=[target])
y = train[target]

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print("Number of categorical columns:", len(categorical_cols))

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

num_cols = X.select_dtypes(include=['int64','float64']).columns
num_imputer = SimpleImputer(strategy="median")
X[num_cols] = num_imputer.fit_transform(X[num_cols])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_np = X_train.values
X_val_np = X_val.values
y_train_np = y_train.values
y_val_np = y_val.values

clf = TabNetClassifier(
    n_d=32,
    n_a=32,
    n_steps=5,
    gamma=1.5,
    lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-3),
    device_name='cuda',
    mask_type='sparsemax'
)

clf.fit(
    X_train=X_train_np,
    y_train=y_train_np,
    eval_set=[(X_val_np, y_val_np)],
    eval_name=['val'],
    eval_metric=['logloss'],
    max_epochs=50,
    patience=10,
    batch_size=1024,
    virtual_batch_size=256,
    num_workers=2,
    drop_last=False
)

y_pred_proba = clf.predict_proba(X_val_np)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_val_np, y_pred))
print("Log Loss:", log_loss(y_val_np, y_pred_proba))
print("AUC:", roc_auc_score(y_val_np, y_pred_proba))

import matplotlib.pyplot as plt

feature_importances = clf.feature_importances_
fi_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 12))
plt.barh(fi_df['feature'][:30][::-1], fi_df['importance'][:30][::-1])
plt.title("Top 30 Important Features (TabNet)")
plt.show()

clf.save_model("tabnet_model.zip")
print("Model saved as tabnet_model.zip")


