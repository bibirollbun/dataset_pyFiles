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


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


# Remove the id column from train and test

train.drop("id",axis=1,inplace=True)
test.drop("id",axis=1,inplace=True)


train.sample(5)


train.info()


train.describe()


train.isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt


num_cols = list(train.select_dtypes(exclude=['object']).columns.difference(['y']))

def plot_numerical_eda(df,num_cols):
    for col in num_cols:
        skewness = df[col].skew()
        kurt = df[col].kurtosis()
        
        print(f"\nColumn: {col}")
        print(f"  Skewness: {skewness:.3f}")
        print(f"  Kurtosis: {kurt:.3f}")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Histogram with KDE
        sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f'Histogram - {col}')
        
        # Boxplot
        sns.boxplot(x=df[col], ax=axes[1], color='lightgreen')
        axes[1].set_title(f'Boxplot - {col}')
        
        # Violin plot
        sns.violinplot(x=df[col], ax=axes[2], color='lightcoral')
        axes[2].set_title(f'Violin Plot - {col}')
        
        plt.tight_layout()
        plt.show()


def preprocessing(df) : 

    # New feature indicating wheter the person was contacted
    # 1--> Contacted  0 --> Not contacted (-1)
    df["was_contacted"] = (df["pdays"] != -1).astype(int)

    # We need to log-transform balance feature
    # For that we need to get rid of negatve values from the column without losing their significance in the data
    df["balance_negative"] = (df["balance"] < 0).astype(int)

    df["balance"] = df["balance"].abs()  #Retain only absolute value for log tranformation
    df["balance"] = np.log1p(df["balance"]+1) # +1 to deal with 0 if any in the feature

    # Fixing skew of other features
    for col in ["campaign","pdays","previous"] :
        df[col] = np.log1p(df[col]+1)

    # Encoding binary-categorical features
    mapping = {"yes" : 1, "no" : 0}
    for col in ["default","housing","loan"] :
        df[col] = df[col].map(mapping)

    return df


train = preprocessing(train)
test = preprocessing(test)


from sklearn.preprocessing import LabelEncoder, StandardScaler

cat_cols = list(train.select_dtypes(include=['object']).columns)
num_cols = list(train.select_dtypes(exclude=['object']).columns.difference(['y']))

le = LabelEncoder()
for col in cat_cols :
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])


# plt.figure(figsize=(15,10))
# sns.heatmap(train.corr(),annot=True, cmap = 'coolwarm')
# plt.show()


test.sample(5)


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score,roc_auc_score
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold


train = train.fillna(0)
test = test.fillna(0)


X = train.drop("y",axis=1)
y = train["y"]


# Best params for LightGBM using Optuna
lgb_params = {'n_estimators': 1582,
                     'max_depth': 15,
                     'learning_rate': 0.04436352313699452,
                     'num_leaves': 77,
                     'min_child_samples': 81,
                     'subsample': 0.8677563315146003,
                     'colsample_bytree': 0.5261353954090011,
                     'reg_alpha': 0.0631139742323974,
                     'reg_lambda': 6.686183660331108,
                     'verbose': -1,
                     'random_state': 42,
                     'device_type': 'gpu',
                     'gpu_platform_id': 0,
                     'gpu_device_id': 0
                  }

xgb_params = {
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 8,
    'n_estimators': 2000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1
}

cat_params = {
    'task_type': 'GPU',
    'devices': '0',
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.05,
    'depth': 8,
    'iterations': 2000,
    'l2_leaf_reg': 3,
    'border_count': 128,
    'verbose': 100
}


import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ===================
# Storage for predictions
# ===================
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_lgb = np.zeros(test.shape[0])
test_xgb = np.zeros(test.shape[0])
test_cat = np.zeros(test.shape[0])

# ===================
# Cross-validation loop
# ===================
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Ensure numeric and fill NaN
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_val = X_val.apply(pd.to_numeric, errors='coerce').fillna(0)
    test_ = test.apply(pd.to_numeric, errors='coerce').fillna(0)

    # ---- LightGBM ----
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_train, y_train)
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    test_lgb += model_lgb.predict_proba(test_)[:, 1]

    # ---- XGBoost ----
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_train, y_train)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    test_xgb += model_xgb.predict_proba(test_)[:, 1]

    # ---- CatBoost ----
    model_cat = cb.CatBoostClassifier(**cat_params)
    model_cat.fit(X_train, y_train)
    oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
    test_cat += model_cat.predict_proba(test_)[:, 1]

    # Fold scores
    score_lgb = roc_auc_score(y_val, oof_lgb[val_idx])
    score_xgb = roc_auc_score(y_val, oof_xgb[val_idx])
    score_cat = roc_auc_score(y_val, oof_cat[val_idx])
    print(f"Fold {fold}: LGB {score_lgb:.4f} | XGB {score_xgb:.4f} | CAT {score_cat:.4f}")


num_folds = 3

# ===================
# Average Test Predictions
# ===================
test_lgb /= num_folds
test_xgb /= num_folds
test_cat /= num_folds

# ===================
# Ensemble with weighted average
# Adjust weights as needed
# ===================
w_lgb, w_xgb, w_cat = 0.4, 0.3, 0.3  # Example weights
oof_ensemble = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
test_ensemble = w_lgb * test_lgb + w_xgb * test_xgb + w_cat * test_cat

# Ensemble score
ensemble_score = roc_auc_score(y, oof_ensemble)
print(f"\nEnsemble ROC-AUC: {ensemble_score:.5f}")


import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score
)
import numpy as np

# âœ… ROC Curve (probabilities for smooth curve)
fpr, tpr, _ = roc_curve(y, oof_ensemble)
auc_score = roc_auc_score(y, oof_ensemble)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# âœ… Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y, oof_ensemble)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label="Precision-Recall Curve", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# âœ… Confusion Matrix (convert probabilities to hard labels with 0.5 threshold)
oof_preds = (oof_ensemble >= 0.5).astype(int)
cm = confusion_matrix(y, oof_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()



sub['y'] = ensemble_score
sub.to_csv("submission.csv",index=False)

