# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


df_train = train.copy()
df_test = test.copy()

y = df_train['diagnosed_diabetes']
X = df_train.drop(['diagnosed_diabetes'], axis=1)

numeric_col = df_train.select_dtypes(['int', 'float']).drop(['id', 'diagnosed_diabetes'], axis=1)
cat_col = df_train.select_dtypes('object')

df_train = df_train.drop(['id'], axis=1)
df_test = df_test.drop(['id'], axis=1)
df_train.info()

df_train.describe().T

for cat in numeric_col.columns:
    df = df_train.groupby(y)[cat].agg(['count', 'mean']).round(2)
    print(cat, '\n', df, '\n')


for cat in cat_col.columns:
    vc = df_train[cat].value_counts(normalize=True).round(2)
    print('\n', vc)


for cat in numeric_col.columns:
    print(f'{cat}: skew is {df_train[cat].skew().round(2)} and kurtosis is {df_train[cat].kurtosis().round(2)} {'\n'}')


education_level = { 
    'No formal': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3   
}

income_level = {
    'Low': 0,
    'Lower-Middle': 1,
    'Upper-Middle': 2,
    'Middle': 3,
    'High': 4   
}

for df in [df_train, df_test]:
    df['education_level'] = df['education_level'].map(education_level) 
    df['income_level'] = df['income_level'].map(income_level) 


nominal_cat = ['gender', 'ethnicity', 'smoking_status', 'employment_status']

df_train[nominal_cat] = df_train[nominal_cat].astype('category')
df_test[nominal_cat] = df_test[nominal_cat].astype('category')

for col in nominal_cat:
    df_train[col] = df_train[col].cat.codes
    df_test[col] = df_test[col].cat.codes



df_train.info()



# check for imbalnce
pos = y.sum()
neg = len(y) - pos


ratio_pos_over_neg = pos / neg
ratio_neg_over_pos = neg / pos


# Find weights to the majority - label, 1 in order to DECREASE his loss
scale_pos_weight = neg / pos

y = df_train['diagnosed_diabetes']
X = df_train.drop(['diagnosed_diabetes'], axis=1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_scores = []
valid_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )


    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_pred_proba)

    y_pred_train = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_pred_train)
    valid_auc = roc_auc_score(y_valid, y_pred_proba)
    
    train_scores.append(train_auc)
    valid_scores.append(valid_auc)

    print("Train AUC mean:", np.mean(train_scores))
    print("Valid AUC mean:", np.mean(valid_scores))
    print("Gap:", np.mean(train_scores) - np.mean(valid_scores), '\n')

print('*'*10, 'Model mean', '*'*10)
print("\nMean AUC:", np.mean(valid_scores).round(4))
print("Std AUC:", np.std(valid_scores).round(4))


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_auc_scores = []
valid_auc_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = LGBMClassifier(
        n_estimators=400,          # מספר עצים
        learning_rate=0.05,        # שיעור למידה
        max_depth=-1,              # נותן חופש לעומק (LGBM אוהב את זה)
        subsample=0.8,             # כמו XGBoost - דגימה מהשורות
        colsample_bytree=0.8,      # דגימה מהפיצ'רים
        objective="binary",        # החזרת הסתברות
        is_unbalance=True,   # משקל למחלקה הנדירה
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # הסתברויות למחלקה החיובית
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_valid_pred = model.predict_proba(X_valid)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_pred)
    valid_auc = roc_auc_score(y_valid, y_valid_pred)

    train_auc_scores.append(train_auc)
    valid_auc_scores.append(valid_auc)

    print(f"Fold {fold}: Train AUC={train_auc:.4f}, Valid AUC={valid_auc:.4f}")

print("\n******** LightGBM CV Summary ********")
print(f"Train AUC mean: {sum(train_auc_scores)/len(train_auc_scores):.4f}")
print(f"Valid AUC mean: {sum(valid_auc_scores)/len(valid_auc_scores):.4f}")

gap = (sum(train_auc_scores)/len(train_auc_scores)) - (sum(valid_auc_scores)/len(valid_auc_scores))
print(f"Gap: {gap:.4f}")


start_row =  df_train.shape[0]
ids = np.arange(start_row, start_row + df_test.shape[0])

y_test_proba = model.predict_proba(df_test)[:, 1]

submission = pd.DataFrame(
    {
        'id': ids,
        'diagnosed_diabetes': y_test_proba.round(1)
    }
)


submission.to_csv("submission.csv", index=False)
submission.head()

