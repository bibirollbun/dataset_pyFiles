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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report,
    f1_score
)


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


categorical_cols=train.select_dtypes(include=["object","category"]).columns
for col in categorical_cols:
    train[col].value_counts().plot(kind="bar")
    plt.show()
    


for col in categorical_cols:
    result=train.groupby([col,"loan_paid_back"]).size().unstack(fill_value=0)
    result.plot(kind='bar')
    plt.title(f"Loan Paid Back by {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()


numeric_col=train.select_dtypes(include=["int64","float64"]).columns
sns.heatmap(train[numeric_col].corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.show()


x=train.drop(["loan_paid_back"],axis=1)
y=train["loan_paid_back"]
combined=pd.concat([x,test],ignore_index=True)
categorical_col=combined.select_dtypes(include=["object","category"]).columns
numerical_col=combined.select_dtypes(include=["int64","float64"]).columns


for col in numerical_col:
    combined[col]=combined[col].fillna(combined[col].median())
le=LabelEncoder()
for col in categorical_col:
    combined[col]=combined[col].fillna(combined[col].mode()[0])
    combined[col] = le.fit_transform(combined[col])


combined["interest_amount"]=combined["loan_amount"]*combined["interest_rate"]/100
combined["income_to_interest_ratio"]=combined["annual_income"]/(combined["interest_amount"]+1)
combined["loan_to_income_ratio"]=combined["loan_amount"]/(combined["annual_income"]+1)
combined["loan_risk"]=combined["credit_score"]/(combined["debt_to_income_ratio"]+1)


for base_col,new_col in [
    ("credit_score","credit_bucket"),
    ("annual_income","annual_bucket"),
    ("debt_to_income_ratio","dti_bucket")
]:
    try:
        combined[new_col]=pd.qcut(
           combined[base_col],
           5,
          labels=False,
          duplicates="drop")
    except Exception as e:
        print(f"Bucket creation failed for {base_col}: {e}")
        combined[new_col] = 0
        


combined["cs_emp_interaction"]=combined["credit_score"]*combined["employment_status"]
combined["risk_dti_ratio"] = combined["loan_risk"] / (combined["dti_bucket"] + 1)
combined["score_grade_combo"] = combined["credit_score"] * combined["grade_subgrade"]


X_processed  = combined.iloc[: len(train)]
X_test_final = combined.iloc[len(train):]


X_train, X_val, y_train, y_val = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = pos / neg
print("neg:", neg, "pos:", pos, "scale_pos_weight:", scale)


best_params_lgb = {
    "n_estimators": 2425,
    "learning_rate": 0.04799514605781845,
    "max_depth": 4,
    "num_leaves": 35,
    "subsample": 0.9906130959928888,
    "colsample_bytree": 0.6077506330189334,
    "min_child_samples": 57,
    "reg_lambda": 4.798855044917759,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 200
}

lgb_best = LGBMClassifier(**best_params_lgb, verbosity=-1)
lgb_best.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc"
)

lgb_val_proba = lgb_best.predict_proba(X_val)[:, 1]
lgb_val_pred  = (lgb_val_proba >= 0.5).astype(int)

print("LGBM V2 ROC-AUC:", roc_auc_score(y_val, lgb_val_proba))
print("\nLGBM V2 Classification Report:\n", classification_report(y_val, lgb_val_pred))



test_proba = lgb_best.predict_proba(X_test_final)[:, 1]


test_ids = test["id"]


submission = pd.DataFrame({
    "id": test_ids,
    "loan_paid_back": test_proba
})


submission.to_csv("submission.csv", index=False)
submission.head()


