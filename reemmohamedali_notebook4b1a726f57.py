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

# Load the train data
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
train.head()



train.describe()


train.info()


train.drop(columns="id", inplace=True)



train["balance"].min()


print(train.isnull().sum())


print("Total Duplicates:", train.duplicated().sum())



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,8))
corr = train.corr(numeric_only=True) 
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()



cat_cols = train.select_dtypes(include="object").columns
print(cat_cols)



import seaborn as sns
import matplotlib.pyplot as plt

for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(data=train, x=col, hue="y")
    plt.title(f"{col} vs Target y")
    plt.xticks(rotation=45)
    plt.show()



from scipy.stats import chi2_contingency
import pandas as pd

for col in cat_cols:
    contingency = pd.crosstab(train[col], train["y"])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"{col}: p-value = {p}")



plt.figure(figsize=(8,5))
sns.histplot(data=train, x="age", hue="y", kde=True, bins=30)
plt.title("Age distribution by target y")
plt.show()



plt.figure(figsize=(6,4))
sns.boxplot(data=train, x="y", y="balance")
plt.title("Balance vs y")
plt.show()



plt.figure(figsize=(8,5))
sns.histplot(data=train, x="duration", hue="y", kde=True, bins=30)
plt.title("Duration distribution by target y")
plt.show()




train["duration_log"] = np.log1p(train["duration"])
plt.figure(figsize=(8,4))
sns.histplot(data=train, x="duration_log", hue="y", bins=50, kde=True, element="step")
plt.title("Log-Transformed Duration by Target y")
plt.show()



sns.countplot(x="y", data=train, palette="Set2")
plt.title("Target Distribution (y)")
plt.show()



X_raw = train.drop(columns=["y"])
y = train["y"]

X = pd.get_dummies(X_raw, drop_first=True)


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Shape after encoding:", X.shape)
print("Any object dtypes left?", X.dtypes.eq("object").any())



from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_val)
y_proba = log_reg.predict_proba(X_val)[:, 1]

print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))
print("ROC-AUC:", roc_auc_score(y_val, y_proba))


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
y_proba = rf.predict_proba(X_val)[:, 1]

print("\n Random Forest Results")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))
print("ROC-AUC:", roc_auc_score(y_val, y_proba))


from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_val)
y_proba = xgb.predict_proba(X_val)[:, 1]

print("\n XGBoost Results")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))
print("ROC-AUC:", roc_auc_score(y_val, y_proba))


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    device='cuda',               
    early_stopping_rounds=10,
    random_state=42
)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

y_proba = xgb_model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_proba)
print("Validation ROC-AUC:", roc_auc)



test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
X_test = pd.get_dummies(test_data)  
X_test = X_test.reindex(columns=X.columns, fill_value=0)  

submission = pd.DataFrame({
    "id": test_data["id"],
    "y": xgb_model.predict_proba(X_test)[:,1]
})

submission.to_csv("submission.csv", index=False)

