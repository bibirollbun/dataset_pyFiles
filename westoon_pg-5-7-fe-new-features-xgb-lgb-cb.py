import numpy as np 
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# Load data

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

add_df = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')

train.head()


# Combining datasets for simultaneous processing

# Merge original dataset
train.drop(columns='id', inplace=True)

train = pd.concat([train.reset_index(drop=True),
                add_df.reset_index(drop=True)],
               ignore_index=True)

# Save train index
train_idx = train.index[:]

# Merge test dataset
df = pd.concat([train.reset_index(drop=True),
                test.reset_index(drop=True)],
               ignore_index=True)


# Data processing

num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
cat_cols = ['Stage_fear', 'Drained_after_socializing']
target = 'Personality'

for col in cat_cols:
    df[col] = df[col].fillna('Unknown')


for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

binary_map = {'Yes': 1, 'No': 0, 'Unknown': -1}

for col in cat_cols:
    df[col] = df[col].map(binary_map)

df.loc[train_idx, target] = df.loc[train_idx, target].map({'Extrovert': 1, 'Introvert': 0})

df.drop(columns=['id'],inplace=True)


# New features

df['social_post_interaction'] = df['Post_frequency'] * df['Social_event_attendance']
df['outside_x_friends'] = df['Going_outside'] * df['Friends_circle_size']


df['social_score'] = (
    df['Social_event_attendance'] + 
    df['Going_outside'] + 
    df['Post_frequency']
)


df['introvert_signals'] = (
    (df['Drained_after_socializing'] == 1).astype(int) + 
    (df['Stage_fear'] == 1).astype(int)
)

df['extrovert_score'] = df['social_score'] - df['introvert_signals']


# Train test split

train_df = df.loc[train_idx].copy()
test_df = df.drop(train_idx).copy()

X_train = train_df.drop(columns=[target])
y_train = train_df[target].astype(int)

X_test = test_df.drop(columns=[target])


# Models parameters

xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
    'early_stopping_rounds': 50,
    'use_label_encoder': False
}


lgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 7,
    'num_leaves': 32,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'objective': 'binary',
    'metric': 'binary_error',
    'device': 'gpu',
    'random_state': 42,
    'verbose':-1
}


cat_params = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 7,
    'l2_leaf_reg': 1,
    'loss_function': 'Logloss',
    'eval_metric': 'Accuracy',
    'task_type': 'GPU',
    'devices': '0',
    'random_seed': 42
}



# XGB model

models_xgb = []
oof_preds_xgb = np.zeros(len(X_train))
test_preds_xgb = np.zeros(len(X_test))
results_xgb = {'accuracy': []}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f'\n Fold {fold + 1}')

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = XGBClassifier(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    models_xgb.append(model)

    # OOF predictions
    oof_preds_xgb[val_idx] = model.predict(X_val)

    # Accuracy
    acc = accuracy_score(y_val, oof_preds_xgb[val_idx])
    results_xgb['accuracy'].append(acc)
    print(f'Accuracy: {acc:.4f}')

    # Predict on test and accumulate
    test_preds_xgb += model.predict_proba(X_test)[:, 1] / kf.n_splits

print(f"\n Mean accuracy: {sum(results_xgb['accuracy']) / kf.n_splits}")


# LGB model

models_lgb = []
oof_preds_lgb = np.zeros(len(X_train))
test_preds_lgb = np.zeros(len(X_test))
results_lgb = {'accuracy': []}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f'\n Fold {fold + 1}')

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr,
        y_tr
    )

    models_lgb.append(model)

    # OOF predictions
    oof_preds_lgb[val_idx] = model.predict(X_val)

    # Accuracy
    acc = accuracy_score(y_val, oof_preds_lgb[val_idx])
    results_lgb['accuracy'].append(acc)
    print(f'Accuracy: {acc:.4f}')

    # Predict on test and accumulate
    test_preds_lgb += model.predict_proba(X_test)[:, 1] / kf.n_splits

print(f"\n Mean accuracy: {sum(results_lgb['accuracy']) / kf.n_splits}")


# CatBoost model

models_cb = []
oof_preds_cb = np.zeros(len(X_train))
test_preds_cb = np.zeros(len(X_test))
results_cb = {'accuracy': []}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f'\n Fold {fold + 1}')

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = CatBoostClassifier(**cat_params)
    model.fit(
        X_tr, y_tr,
        verbose=False
    )

    models_cb.append(model)

    # OOF predictions
    oof_preds_cb[val_idx] = model.predict(X_val)

    # Accuracy
    acc = accuracy_score(y_val, oof_preds_cb[val_idx])
    results_cb['accuracy'].append(acc)
    print(f'Accuracy: {acc:.4f}')

    # Predict on test and accumulate
    test_preds_cb += model.predict_proba(X_test)[:, 1] / kf.n_splits

print(f"\n Mean accuracy: {sum(results_cb['accuracy']) / kf.n_splits}")


# Combining results
final_test_preds = (
    0.3 * test_preds_cb +
    0.3 * test_preds_lgb +
    0.4 * test_preds_xgb
)


# Submission
final_labels = (final_test_preds >= 0.5).astype(int)

submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission['Personality'] = np.where(final_labels == 1, 'Extrovert', 'Introvert')

submission.to_csv('submission_ensemble.csv', index=False)

