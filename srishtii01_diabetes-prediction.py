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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import *
import xgboost as xgb
import lightgbm as lgb

print("All Libraries Imported Successfully âœ… ")


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

print("Data Read Successfully âœ…")

print("Train shape:", train.shape)
print("Test shape:", test.shape)


train.head()


plt.figure(figsize=(6,4))
sns.countplot(x=train["diagnosed_diabetes"])
plt.title("Target Distribution: Diagnosed Diabetes")
plt.xlabel("Diagnosed Diabetes (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()



train["diagnosed_diabetes"].value_counts(normalize=True) * 100


TARGET = "diagnosed_diabetes"

X = train.drop([TARGET, "id"], axis=1)
y = train[TARGET]

X_test = test.drop("id", axis=1)

print(X.shape, y.shape, X_test.shape)



numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

X[numerical_cols].hist(
    figsize=(20,16),
    bins=30,
    edgecolor="black"
)

plt.suptitle("Numerical Feature Distributions", fontsize=16)
plt.show()



for col in numerical_cols[:5]:   
    plt.figure(figsize=(5,3))
    sns.boxplot(x=train["diagnosed_diabetes"], y=train[col])
    plt.title(f"{col} vs Diagnosed Diabetes")
    plt.show()



corr_matrix = train[numerical_cols.tolist() + ["diagnosed_diabetes"]].corr()
corr_matrix



plt.figure(figsize=(4,4))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.show()



target_corr = corr_matrix["diagnosed_diabetes"].sort_values(ascending=False)

plt.figure(figsize=(6,6))
target_corr.drop("diagnosed_diabetes").plot(kind="barh")
plt.title("Feature Correlation with Target")
plt.xlabel("Correlation")
plt.show()



TARGET = "diagnosed_diabetes"

X = train.drop([TARGET, "id"], axis=1)
y = train[TARGET]

X_test = test.drop("id", axis=1)

print(X.shape, y.shape, X_test.shape)


X.head()


# Interaction & transformation features
X["bmi_log"] = np.log1p(X["bmi"])
X_test["bmi_log"] = np.log1p(X_test["bmi"])

X["age_squared"] = X["age"] ** 2
X_test["age_squared"] = X_test["age"] ** 2

X["waist_bmi_ratio"] = X["waist_to_hip_ratio"] / (X["bmi"] + 1e-5)
X_test["waist_bmi_ratio"] = X_test["waist_to_hip_ratio"] / (X_test["bmi"] + 1e-5)


categorical_cols = X.select_dtypes(include=["object"]).columns
categorical_cols



X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)


X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)


scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y), 1):
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc"
    )

    model.fit(X_tr, y_tr)

    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test_scaled)[:, 1]

    oof_preds[val_idx] = val_pred
    test_preds += test_pred / kf.n_splits

    print(f"Fold {fold} ROC-AUC:", roc_auc_score(y_val, val_pred))

print("\nOverall OOF ROC-AUC:", roc_auc_score(y, oof_preds))


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_scaled, y)

lr_test_preds = lr.predict_proba(X_test_scaled)[:, 1]


final_preds = 0.7 * test_preds + 0.3 * lr_test_preds


submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": final_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()


submission.to_csv("submission.csv", index=False)
print("submission.csv created")

