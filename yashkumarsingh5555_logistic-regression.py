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


missing_percent = (train.isnull().sum() / len(train)) * 100

missing_table = missing_percent[missing_percent > 0].sort_values(ascending=False)

print(missing_table.head(20))


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score, roc_curve, auc
from sklearn.pipeline import Pipeline
import time




target = "HasDetections"
X = train.drop(columns=[target])
y = train[target]


categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)) # Changed to sparse_output
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)


model = LogisticRegression(
    max_iter=500,
    solver="lbfgs",
    n_jobs=-1
)


clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

start = time.time()
clf.fit(X_train, y_train)
log_train_time = time.time() - start


print("\nLearning time:", log_train_time)



start = time.time()
y_pred = clf.predict(X_val)
log_infer_time = time.time() - start
y_pred_proba = clf.predict_proba(X_val)[:, 1]
print("\nInference time:", log_infer_time)

auc_score = roc_auc_score(y_val, y_pred_proba)
logloss = log_loss(y_val, y_pred_proba)

print("\nAccuracy:", accuracy_score(y_val, y_pred))

print(f"\nAUC Score: {auc_score:.4f}")
print(f"Log Loss: {logloss:.4f}")

print("\nClassification Report:\n", classification_report(y_val, y_pred))


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Guess 0.5')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.show()

