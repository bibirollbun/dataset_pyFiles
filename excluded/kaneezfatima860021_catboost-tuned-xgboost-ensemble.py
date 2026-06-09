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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings; warnings.filterwarnings('ignore')


# Load data
data_path = "/kaggle/input/playground-series-s5e7"
train = pd.read_csv(f"{data_path}/train.csv")
test  = pd.read_csv(f"{data_path}/test.csv")
print(train.shape, test.shape)


train.head()
print(train['Personality'].value_counts())


missing = train.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0].head())


num_cols = train.select_dtypes(include=['int64','float64']).columns
cat_cols = train.select_dtypes('object').drop(['Personality'], axis=1).columns.tolist()



# Fill NaNs
auto_median = train[num_cols].median()
train[num_cols] = train[num_cols].fillna(auto_median)
# Fill only numeric columns that exist in test (skip 'y' if present)
test_cols_to_fill = [col for col in num_cols if col in test.columns]
train[num_cols] = train[num_cols].fillna(auto_median)
test[test_cols_to_fill] = test[test_cols_to_fill].fillna(auto_median)

for col in cat_cols:
    mode = train[col].mode()[0]
    train[col].fillna(mode, inplace=True)
    test[col].fillna(mode, inplace=True)


# Encode target
target_le = LabelEncoder()
train['y'] = target_le.fit_transform(train['Personality'])
y = train['y']


# Labelâ€‘encode categoricals for XGBoost
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

X      = train.drop(columns=['id','Personality','y'])
X_test = test.drop(columns=['id'])
cat_idx = [X.columns.get_loc(c) for c in cat_cols]


SEED, N_FOLDS = 42, 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# CatBoost params (MultiClass)
cat_params = dict(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=4,
    random_state=SEED,
    eval_metric='Accuracy',
    loss_function='MultiClass',
    early_stopping_rounds=200,
    verbose=False
)


# XGBoost params (MultiClass)
xgb_params = dict(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED,
    objective='multi:softprob',
    num_class=len(np.unique(y)),
    eval_metric='mlogloss',
    tree_method='hist'
)


cat_oof   = np.zeros((len(train), len(np.unique(y))))
cat_probs = np.zeros((len(test), len(np.unique(y))))

xgb_oof   = np.zeros((len(train), len(np.unique(y))))
xgb_probs = np.zeros((len(test), len(np.unique(y))))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}")


    # ---- CatBoost ----
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
                  cat_features=cat_idx,
                  use_best_model=True)
    cat_oof[val_idx] = cat_model.predict_proba(X.iloc[val_idx])
    cat_probs += cat_model.predict_proba(X_test) / N_FOLDS

    # ---- XGBoost ----
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                  early_stopping_rounds=200,
                  verbose=False)
    xgb_oof[val_idx] = xgb_model.predict_proba(X.iloc[val_idx])
    xgb_probs += xgb_model.predict_proba(X_test) / N_FOLDS


blend_weight = 0.6
blended_oof = blend_weight * cat_oof + (1 - blend_weight) * xgb_oof
blended_labels = np.argmax(blended_oof, axis=1)

print('Blended OOF Accuracy:', accuracy_score(y, blended_labels).round(4))
print(classification_report(y, blended_labels, target_names=target_le.classes_))

cm = confusion_matrix(y, blended_labels)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=target_le.classes_,
            yticklabels=target_le.classes_)
plt.title('Blended Confusion Matrix')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout(); plt.show()


blended_test_probs = blend_weight * cat_probs + (1 - blend_weight) * xgb_probs
final_preds = np.argmax(blended_test_probs, axis=1)

submission = pd.DataFrame({
    'id': test['id'],
    'Personality': target_le.inverse_transform(final_preds)
})
submission.to_csv('submission.csv', index=False)
print('ğŸ“„ submission.csv saved')




