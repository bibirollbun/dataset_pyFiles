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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")



test_ids = test_df["id"].copy()



X = train_df.drop(columns=["id", "diagnosed_diabetes"])
y = train_df["diagnosed_diabetes"]

X_test = test_df.drop(columns=["id"])
train_df


cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns



X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=cat_cols, drop_first=True)



X, test_df = X.align(test_df, join="left", axis=1, fill_value=0)



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(test_df)



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)



model.fit(X_train_scaled, y_train)



from sklearn.metrics import roc_auc_score

val_preds = model.predict_proba(X_val_scaled)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)

print("Validation ROC-AUC:", roc_auc)



from sklearn.ensemble import GradientBoostingClassifier

model_boos = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)



model_boos.fit(X_train_scaled, y_train)



val_preds = model_boos.predict_proba(X_val_scaled)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)

print("Validation ROC-AUC:", roc_auc)



test_preds = model_boos.predict_proba(X_test_scaled)[:, 1]


submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission.csv", index=False)



submission


import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

fpr, tpr, thresholds = roc_curve(y_val, val_preds)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], color='red', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()





