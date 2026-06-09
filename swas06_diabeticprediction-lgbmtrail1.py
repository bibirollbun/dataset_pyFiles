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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import xgboost as xgb
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency
import plotly.figure_factory as f



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


train.head(3)


train['education_level'] = train['education_level'].map({
    'No formal': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3
})
test['education_level'] = test['education_level'].map({
    'No formal': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3
})

# Income Level (clear progression)
income_order = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}
train['income_level'] = train['income_level'].map(income_order)
test['income_level'] = test['income_level'].map(income_order)

# Smoking Status (health impact order)
smoking_order = {'Never': 0, 'Former': 1, 'Current': 2}
train['smoking_status'] = train['smoking_status'].map(smoking_order)
test['smoking_status'] = test['smoking_status'].map(smoking_order)


from sklearn.preprocessing import LabelEncoder
le_gender = LabelEncoder()
le_ethnicity = LabelEncoder()
le_employment = LabelEncoder()

# Gender - Label Encoding
train['gender'] = le_gender.fit_transform(train['gender'])
test['gender'] = le_gender.transform(test['gender'])

# Ethnicity - Label Encoding
train['ethnicity'] = le_ethnicity.fit_transform(train['ethnicity'])
test['ethnicity'] = le_ethnicity.transform(test['ethnicity'])

# Employment Status - Label Encoding
train['employment_status'] = le_employment.fit_transform(train['employment_status'])
test['employment_status'] = le_employment.transform(test['employment_status'])


train = train.drop(columns=["id"])
test = test.drop(columns=["id"])


X = train.drop(columns=['diagnosed_diabetes'])
y = train['diagnosed_diabetes']

X_test = test[X.columns]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)


train.info()


def objective(trial):
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "is_unbalance": False,
        "class_weight": {0: 1, 1: scale_pos_weight},
        "n_estimators": trial.suggest_int("n_estimators", 300, 5000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "verbose": -1
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc"
        )

        preds = model.predict_proba(X_va)[:, 1]
        auc_scores.append(roc_auc_score(y_va, preds))

    return np.mean(auc_scores)


best_params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "class_weight": {0: 1},
    "device_type": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "n_estimators": 4133,
    "learning_rate": 0.02068171689337402,
    "num_leaves": 28,
    "max_depth": 5,
    "min_child_samples": 24,
    "subsample": 0.7412048932901383,
    "colsample_bytree": 0.965256994834274,
    "reg_alpha": 2.5659237069355467,
    "reg_lambda": 1.028840143860832,
    "verbose": -1
}


from sklearn.metrics import roc_auc_score, roc_curve
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))
fold_aucs = []

plt.figure(figsize=(12, 8))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

    model = LGBMClassifier(**best_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc"
    )

    y_val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred)
    fold_aucs.append(auc)

    print(f"Fold {fold} AUC: {auc:.6f}")

    # accumulate test predictions
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

    # ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_val_pred)
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC={auc:.3f})')

print("\nMean AUC:", np.mean(fold_aucs))
print("Std AUC:", np.std(fold_aucs))

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LGBM ROC Curves per Fold')
plt.legend(loc='lower right')
plt.show()


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


submission = pd.DataFrame({
    'id': submission_df.id,  
    'prediction': test_preds
})

submission.to_csv('submission.csv', index=False)

