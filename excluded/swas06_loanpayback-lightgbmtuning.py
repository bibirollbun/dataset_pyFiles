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



train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
def encode_categorical(df):
    """
    Encodes categorical columns :
    - One-hot encodes nominal features
    - Ordinal encodes ordered features
    - Maps grade_subgrade to numeric order
    Returns: dataframe with numeric columns only
    """
    
    # Make a copy to avoid changing original data
    df = df.copy()
    
    # Define column groups
    nominal_cols = ['gender', 'marital_status', 'employment_status', 'loan_purpose']
    ordinal_cols = ['education_level']
    
    # 1️⃣ Ordinal encoding for education_level
    education_order = [['High School', "Bachelor's", "Master's", 'PhD', 'Other']]
    ord_enc = OrdinalEncoder(categories=education_order)
    df['education_level'] = ord_enc.fit_transform(df[['education_level']])
    
    # 2️⃣ Custom mapping for grade_subgrade
    grade_map = {g: i for i, g in enumerate(
        ['A1','A2','A3','A4','A5',
         'B1','B2','B3','B4','B5',
         'C1','C2','C3','C4','C5',
         'D1','D2','D3','D4','D5',
         'E1','E2','E3','E4','E5',
         'F1','F2','F3','F4','F5'], start=1)}
    df['grade_subgrade'] = df['grade_subgrade'].map(grade_map)
    
    # 3️⃣ One-hot encode nominal columns
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)
    
    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric)
    
    
    return df


train  = encode_categorical(train)
test  = encode_categorical(test)


train.head(3)


train = train.drop(columns=["id"])
test = test.drop(columns=["id"])


X = train.drop(columns=['loan_paid_back'])
y = train['loan_paid_back']

X_test = test[X.columns]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)


scale_pos_weight = y_train.value_counts()[0.0] / y_train.value_counts()[1.0]


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
    "class_weight": {0: 1, 1: scale_pos_weight},
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


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


submission = pd.DataFrame({
    'id': submission_df.id,  
    'prediction': test_preds
})

submission.to_csv('submission.csv', index=False)

