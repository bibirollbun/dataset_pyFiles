import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_df = train_df.drop('id', axis=1)


def clean_data(df):
    # Replace missing values with the mean of each column in: 'Time_spent_Alone'
    df = df.fillna({'Time_spent_Alone': df['Time_spent_Alone'].mean()})
    # Round column 'Time_spent_Alone' (Number of decimals: 1)
    df = df.round({'Time_spent_Alone': 0})

    # Replace missing values with the most common value of each column in: 'Stage_fear'
    df = df.fillna({'Stage_fear': df['Stage_fear'].mode()[0]})

    # Replace missing values with the mean of each column in: 'Social_event_attendance'
    df = df.fillna({'Social_event_attendance': df['Social_event_attendance'].mean()})
    df = df.round({'Social_event_attendance': 0})

    # Replace missing values with the mean of each column in: 'Going_outside'
    df = df.fillna({'Going_outside': df['Going_outside'].mean()})
    df = df.round({'Going_outside': 0})

    # Replace missing values with the most common value of each column in: 'Drained_after_socializing'
    df = df.fillna({'Drained_after_socializing': df['Drained_after_socializing'].mode()[0]})

    # Replace missing values with the mean of each column in: 'Friends_circle_size'
    df = df.fillna({'Friends_circle_size': df['Friends_circle_size'].mean()})
    df = df.round({'Friends_circle_size': 0})

    # Replace missing values with the mean of each column in: 'Post_frequency'
    df = df.fillna({'Post_frequency': df['Post_frequency'].mean()})
    df = df.round({'Post_frequency': 0})
    return df




df = clean_data(train_df)
test_df = clean_data(test_df)


def feature_engineering(df):
    df['social_activity_score'] = (
        0.5 * df['Social_event_attendance'].fillna(0) +
        0.3 * df['Going_outside'].fillna(0) +
        0.2 * df['Post_frequency'].fillna(0)
    )
    df['friend_post_density'] = df['Post_frequency'] / (df['Friends_circle_size'] + 1e-5)
    df['introversion_index'] = df['Time_spent_Alone'] / (df['Social_event_attendance'] + 1)
    df['interactiveness'] = (
        df['Friends_circle_size'].fillna(0) +
        df['Social_event_attendance'].fillna(0) +
        df['Post_frequency'].fillna(0)
    )
    
    return df


df = feature_engineering(df)
test_df = feature_engineering(test_df)


df.shape


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np



# Encode categorical and target columns
for col in ['Stage_fear', 'Drained_after_socializing']:
    df[col] = LabelEncoder().fit_transform(df[col])
    
# 1. Fit once during training
target_le = LabelEncoder()
df['Personality'] = target_le.fit_transform(train_df['Personality'])

# Split X, y
X = df.drop(['Personality'], axis=1)
y = df['Personality']


import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# --- Set model parameters ---
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_estimators': 100,      # Adjust this.
    'learning_rate': 0.007,
    'max_depth': 10,
    'subsample': 0.76,
    'colsample_bytree': 0.51,
    'reg_lambda': 6.51,
    'reg_alpha': 0.0,
    'tree_method': 'gpu_hist',  # Use 'hist' for CPU
    'verbosity': 0
}

lgbm_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'random_state': 42,
    'n_estimators': 100,   # Adjust this.
    'learning_rate': 0.007,
    'max_depth': 10,
    'num_leaves': 64,
    'min_child_samples': 5,
    'subsample': 0.76,
    'subsample_freq': 1,
    'colsample_bytree': 0.51,
    'reg_lambda': 6.51,
    'reg_alpha': 0.0,
    'device': 'gpu',  # Use 'cpu' for CPU
    'verbose': -1
}

catboost_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'task_type': 'GPU',  # Use 'CPU' for CPU
    'random_seed': 42,
    'iterations': 100,         # Adjust this.
    'learning_rate': 0.007,
    'depth': 10,
    'border_count': 254,
    'l2_leaf_reg': 6.51,
    'random_strength': 5.56,
    'min_data_in_leaf': 5,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.76,
    # 'colsample_bylevel': 0.51,  # Not supported on GPU for Logloss
    'grow_policy': 'SymmetricTree',
    'verbose': False
}


# --- Prepare cross-validation ---
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_preds_xgb = np.zeros(len(y))
oof_preds_lgbm = np.zeros(len(y))
oof_preds_cat = np.zeros(len(y))
oof_preds_ensemble = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # XGBoost
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        # early_stopping_rounds=100,
        verbose=False
    )
    val_preds_xgb = xgb.predict(X_val)
    oof_preds_xgb[val_idx] = val_preds_xgb

    # LightGBM
    lgbm = LGBMClassifier(**lgbm_params)
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        # early_stopping_rounds=100,
        # verbose=False
    )
    val_preds_lgbm = lgbm.predict(X_val)
    oof_preds_lgbm[val_idx] = val_preds_lgbm

    # CatBoost
    cat = CatBoostClassifier(**catboost_params)
    cat.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        # early_stopping_rounds=100,
        use_best_model=True
    )
    val_preds_cat = cat.predict(X_val)
    oof_preds_cat[val_idx] = val_preds_cat

    # Ensemble: majority voting
    val_preds_ensemble = (
        val_preds_xgb.astype(int) +
        val_preds_lgbm.astype(int) +
        val_preds_cat.astype(int)
    )
    # If sum >= 2, predict 1, else 0
    oof_preds_ensemble[val_idx] = (val_preds_ensemble >= 2).astype(int)

    print(f"Fold {fold} Accuracies: "
          f"XGB={accuracy_score(y_val, val_preds_xgb):.4f}, "
          f"LGBM={accuracy_score(y_val, val_preds_lgbm):.4f}, "
          f"CAT={accuracy_score(y_val, val_preds_cat):.4f}, "
          f"Ensemble={accuracy_score(y_val, oof_preds_ensemble[val_idx]):.4f}")

print("\nFinal OOF Accuracies:")
print(f"XGBoost:  {accuracy_score(y, oof_preds_xgb):.4f}")
print(f"LightGBM: {accuracy_score(y, oof_preds_lgbm):.4f}")
print(f"CatBoost: {accuracy_score(y, oof_preds_cat):.4f}")
print(f"Ensemble: {accuracy_score(y, oof_preds_ensemble):.4f}")



# Prepare X_test
X_test = test_df.drop('id', axis=1)

for col in ['Stage_fear', 'Drained_after_socializing']:
    X_test[col] = LabelEncoder().fit_transform(X_test[col])

# Train on full data
xgb_full = XGBClassifier(**xgb_params)
xgb_full.fit(X, y)
xgb_test_preds = xgb_full.predict(X_test)

lgbm_full = LGBMClassifier(**lgbm_params)
lgbm_full.fit(X, y)
lgbm_test_preds = lgbm_full.predict(X_test)

cat_full = CatBoostClassifier(**catboost_params)
cat_full.fit(X, y)
cat_test_preds = cat_full.predict(X_test)

# Ensemble: majority voting
ensemble_test_preds = (
    xgb_test_preds.astype(int) +
    lgbm_test_preds.astype(int) +
    cat_test_preds.astype(int)
)
ensemble_test_preds = (ensemble_test_preds >= 2).astype(int)

# Inverse transform to original class labels
final_preds = target_le.inverse_transform(ensemble_test_preds)

# Save to CSV
submission = pd.DataFrame({
    'id': test_df['id'],  # Use the original id column from test_df
    'Personality': final_preds
})
submission.to_csv('submission.csv', index=False)


