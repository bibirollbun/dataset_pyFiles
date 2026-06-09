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


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error, make_scorer, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier
from sklearn.base import clone
import optuna
import warnings
warnings.filterwarnings("ignore")



train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
orig_df = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')


# Function to display missing values and data types
def check_missing_and_dtypes(df, name="Dataset"):
    print(f"\n{name} Info:")
    print("-" * 50)
    nulls = df.isnull().sum()
    dtypes = df.dtypes
    summary = pd.DataFrame({
        "Data Type": dtypes,
        "Missing Values": nulls,
        "Missing (%)": (nulls / len(df)) * 100
    })
    print(summary)
    return summary


train_info = check_missing_and_dtypes(train, "Train Set")
test_info = check_missing_and_dtypes(test, "Test Set")
orig_info = check_missing_and_dtypes(orig_df, 'Original Set')


def feature_rename(df):
    return df.rename(columns={'Temparature': 'Temperature'})

#replace temparature
train = feature_rename(train)
test = feature_rename(test)
orig_df = feature_rename(orig_df)

#merge orig_df and train set
train = pd.concat([train, orig_df])


cat_cols = ['Crop Type', 'Soil Type']
num_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
target_col = 'Fertilizer Name'

# --- Encode target ---
le = LabelEncoder()
train[target_col] = le.fit_transform(train[target_col])

# --- Train/Valid split ---
x = train[cat_cols + num_cols]
y = train[target_col]


encoder = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
], remainder='passthrough') 


X_enc = encoder.fit_transform(x)
X_test_enc = encoder.transform(test) 


#x_train, x_valid, y_train, y_valid = train_test_split(X_enc, y, stratify=y, test_size=0.2, random_state=42)


def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    for i, p in enumerate(predicted):
        if p == actual:
            score += 1.0 / (i + 1.0)
            break
    return score

def mapk(actuals, predicteds, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predicteds)])


n_classes = len(le.classes_)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_oof = np.zeros((X_enc.shape[0], n_classes))
lgb_oof = np.zeros((X_enc.shape[0], n_classes))
xgb_test_folds = []
lgb_test_folds = []

xgb_scores = []
lgb_scores = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_enc, y)):
    print(f"\nğŸ”� Fold {fold + 1}")

    X_tr, X_val = X_enc[train_idx], X_enc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

    # --- XGBoost ---
    xgb_model = XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        n_estimators=7500,
        learning_rate=0.016899561249968886,
        max_depth=10,
        subsample=0.8682884347090585,
        colsample_bytree=0.6764980670532166,
        reg_alpha=7.384838748996028,
        reg_lambda=6.677980002272297,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=0)
    xgb_val_probs = xgb_model.predict_proba(X_val)
    xgb_oof[valid_idx] = xgb_val_probs
    xgb_test_folds.append(xgb_model.predict_proba(X_test_enc))

    xgb_top3 = np.argsort(xgb_val_probs, axis=1)[:, -3:][:, ::-1]
    flat_top3 = xgb_top3.flatten()
    xgb_preds = le.inverse_transform(flat_top3).reshape(xgb_top3.shape)
    xgb_true = le.inverse_transform(y_val)
    xgb_map3 = mapk(xgb_true, xgb_preds, k=3)
    xgb_scores.append(xgb_map3)
    print(f"ğŸ“˜ XGB Fold {fold + 1} MAP@3: {xgb_map3:.4f}")

    # --- LightGBM ---
    lgb_model = LGBMClassifier(
        objective='multiclass',
        num_class=n_classes,
        n_estimators=8000,
        learning_rate=0.0035485095833419416,
        max_depth=12,
        num_leaves=433,
        min_child_samples=87,
        colsample_bytree=0.9610872245314738,
        subsample=0.935777602645215,
        lambda_l1=7.025689253089766,
        lambda_l2=1.4202823325638203,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  eval_metric="multi_logloss", callbacks=[early_stopping(stopping_rounds=50)])
    lgb_val_probs = lgb_model.predict_proba(X_val)
    lgb_oof[valid_idx] = lgb_val_probs
    lgb_test_folds.append(lgb_model.predict_proba(X_test_enc))

    lgb_top3 = np.argsort(lgb_val_probs, axis=1)[:, -3:][:, ::-1]
    flat_top3 = lgb_top3.flatten()
    lgb_preds = le.inverse_transform(flat_top3).reshape(lgb_top3.shape)
    lgb_true = le.inverse_transform(y_val)
    lgb_map3 = mapk(lgb_true, lgb_preds, k=3)
    lgb_scores.append(lgb_map3)
    print(f"ğŸ“— LGB Fold {fold + 1} MAP@3: {lgb_map3:.4f}")


print("\nğŸ§  Training Meta-Model (Logistic Regression)...")
meta_X = np.hstack([xgb_oof, lgb_oof])

meta_model = LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial',
    C=32.89802104596641,
    tol=0.0029878837974181643,
    max_iter=1000,
    fit_intercept=True,
    random_state=42
)
meta_model.fit(meta_X, y)


# Evaluate on training
meta_val_preds = meta_model.predict_proba(meta_X)
meta_top3 = np.argsort(meta_val_preds, axis=1)[:, -3:][:, ::-1]
flat_top3 = meta_top3.flatten()
meta_preds = le.inverse_transform(flat_top3).reshape(meta_top3.shape)
meta_true = le.inverse_transform(y)
meta_map3 = mapk(meta_true, meta_preds, k=3)
print(f"\nğŸ§  Meta-Model LogisticRegression MAP@3 on train: {meta_map3:.4f}")


xgb_test_avg = np.mean(xgb_test_folds, axis=0)
lgb_test_avg = np.mean(lgb_test_folds, axis=0)
meta_test_X = np.hstack([xgb_test_avg, lgb_test_avg])
final_probs = meta_model.predict_proba(meta_test_X)

top3_preds = np.argsort(final_probs, axis=1)[:, -3:][:, ::-1]
flat_top3 = top3_preds.flatten()
top3_labels = le.inverse_transform(flat_top3).reshape(top3_preds.shape)


submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})
submission.to_csv("stacked_submission.csv", index=False)
print("\nâœ… Submission file 'stacked_submission.csv' saved!")

# ğŸ“Š 8. FINAL SUMMARY
print("\nğŸ“Š Final MAP@3 Scores:")
print(f"ğŸ“˜ XGBoost  Avg MAP@3: {np.mean(xgb_scores):.4f}")
print(f"ğŸ“— LightGBM Avg MAP@3: {np.mean(lgb_scores):.4f}")
print(f"ğŸ§  Meta-Model LogisticRegression MAP@3: {meta_map3:.4f}")




