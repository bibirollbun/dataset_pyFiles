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
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
#import pytabkit
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from sklearn.metrics import *

from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, early_stopping, log_evaluation,early_stopping
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from lightgbm import early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')
print('Done')


df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")



df_train.head


df_train.shape


df_test.shape


df_test.head()


df_train.head()


df_train.columns


df_train.info()


df_train.describe().T


df_train.isnull().sum().sort_values(ascending=False)



df_train.duplicated().sum()



df_train["loan_paid_back"].value_counts()



df_train["loan_paid_back"].value_counts(normalize=True)



sns.countplot(x="loan_paid_back", data=df_train)
plt.title("Loan Payback Distribution")
plt.show()



cat_cols = df_train.select_dtypes(include="object").columns.tolist()
num_cols = df_train.select_dtypes(exclude="object").columns.tolist()

num_cols.remove("loan_paid_back")

cat_cols, num_cols



for col in num_cols:
    plt.figure(figsize=(5,3))
    sns.kdeplot(
        data=df_train,
        x=col,
        hue="loan_paid_back",
        fill=True,
        common_norm=False,
        alpha=0.4
    )
    plt.title(f"{col} distribution by loan status")
    plt.show()



df_train.groupby("loan_paid_back")[num_cols].mean().T



df_train.groupby("loan_paid_back")[num_cols].median().T



def effect_size(col):
    g0 = df_train[df_train.loan_paid_back == 0][col]
    g1 = df_train[df_train.loan_paid_back == 1][col]
    return (g1.mean() - g0.mean()) / df_train[col].std()

for col in num_cols:
    print(col, round(effect_size(col), 3))



for col in num_cols:
    df_train[f"{col}_bin"] = pd.qcut(df_train[col], q=5, duplicates="drop")
    
    rate = df_train.groupby(f"{col}_bin")["loan_paid_back"].mean()
    
    rate.plot(kind="bar", figsize=(5,3), title=f"Payback Rate vs {col}")
    plt.ylabel("Payback Probability")
    plt.show()



for col in num_cols:
    q99 = df_train[col].quantile(0.99)
    high_risk_rate = df_train[df_train[col] > q99]["loan_paid_back"].mean()
    overall_rate = df_train["loan_paid_back"].mean()
    
    print(col, "High-value payback rate:", round(high_risk_rate,3),
          "| Overall:", round(overall_rate,3))



plt.figure(figsize=(10,6))
sns.heatmap(df_train[num_cols + ["loan_paid_back"]].corr(),
            cmap="coolwarm", annot=False)
plt.title("Correlation Matrix")
plt.show()



df_test_ids = df_test["id"]

df_train.drop(columns=["id"], inplace=True)
df_test.drop(columns=["id"], inplace=True)



num_cols = df_train.select_dtypes(exclude="object").columns.tolist()
num_cols.remove("loan_paid_back")




bin_cols = [col for col in df_train.columns if col.endswith("_bin")]

df_train.drop(columns=bin_cols, inplace=True)



corr_target = df_train[num_cols].corrwith(df_train["loan_paid_back"]) \
                                .sort_values(ascending=False)

corr_target


TARGET = "loan_paid_back"

X = df_train.drop(columns=[TARGET])
y = df_train[TARGET]

print(X.shape, y.shape)



X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


y.value_counts(normalize=True)



cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

cat_cols, num_cols




X_combined = pd.concat([X_tr, X_val], axis=0)

X_combined = pd.get_dummies(X_combined, columns=cat_cols, drop_first=True)

X_tr_enc = X_combined.iloc[:len(X_tr)]
X_val_enc = X_combined.iloc[len(X_tr):]



scaler = StandardScaler()

X_tr_enc[num_cols] = scaler.fit_transform(X_tr_enc[num_cols])
X_val_enc[num_cols] = scaler.transform(X_val_enc[num_cols])



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

lr.fit(X_tr_enc, y_tr)

val_pred_lr = lr.predict(X_val_enc)
val_prob_lr = lr.predict_proba(X_val_enc)[:, 1]

print(classification_report(y_val, val_pred_lr))
print("ROC-AUC:", roc_auc_score(y_val, val_prob_lr))



TARGET = "loan_paid_back"

X = df_train.drop(columns=[TARGET])
y = df_train[TARGET]

X_test_final = df_test.copy()



cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()



X_all = pd.concat([X, X_test_final], axis=0)

X_all_encoded = pd.get_dummies(
    X_all,
    columns=cat_cols,
    drop_first=True
)

X_encoded = X_all_encoded.iloc[:len(X)]
X_test_encoded = X_all_encoded.iloc[len(X):]



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])
X_test_encoded[num_cols] = scaler.transform(X_test_encoded[num_cols])



from sklearn.linear_model import LogisticRegression

final_lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

final_lr.fit(X_encoded, y)



test_prob = final_lr.predict_proba(X_test_encoded)[:, 1]



submission = pd.DataFrame({
    "id": df_test_ids,
    "loan_paid_back": test_prob
})

submission.to_csv("submission.csv", index=False)
submission.head()



from xgboost import XGBClassifier



neg, pos = np.bincount(y)
scale_pos_weight = neg / pos
scale_pos_weight



from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)



xgb = XGBClassifier(
    n_estimators=1200,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)



xgb.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    verbose=False
)



from sklearn.metrics import roc_auc_score

val_prob_xgb = xgb.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, val_prob_xgb)



xgb.best_iteration



xgb = XGBClassifier(
    n_estimators=4000,          
    learning_rate=0.01,         
    max_depth=7,                
    min_child_weight=1,         
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.0,                  
    reg_alpha=0.0,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)



xgb.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=300,
    verbose=False
)



xgb.best_iteration
roc_auc_score(y_val, xgb.predict_proba(X_val)[:,1])



X_encoded["dti_interest"] = (
    X_encoded["debt_to_income_ratio"] * X_encoded["interest_rate"]
)

X_test_encoded["dti_interest"] = (
    X_test_encoded["debt_to_income_ratio"] * X_test_encoded["interest_rate"]
)



import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score



from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)



lgb_model = LGBMClassifier(
    n_estimators=6000,
    learning_rate=0.01,
    num_leaves=96,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight="balanced",
    objective="binary",
    metric="auc",
    random_state=42,
    n_jobs=-1
)



lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=500)],
)





val_prob_lgb = lgb_model.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, val_prob_lgb)



lgb_model.best_iteration_



lgb_final = LGBMClassifier(
    n_estimators=lgb_model.best_iteration_ + 200,
    learning_rate=0.01,
    num_leaves=96,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight="balanced",
    objective="binary",
    metric="auc",
    random_state=42,
    n_jobs=-1
)

lgb_final.fit(X_encoded, y)



test_prob_lgb = lgb_final.predict_proba(X_test_encoded)[:, 1]

submission = pd.DataFrame({
    "id": df_test_ids,
    "loan_paid_back": test_prob_lgb
})

submission.to_csv("submission_lgb.csv", index=False)


