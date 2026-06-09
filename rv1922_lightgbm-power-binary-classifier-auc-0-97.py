import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train.head()


train.info()


train.describe()


train.isnull().sum()


for df in [train, test]:
    df['age_prime_working'] = ((df['age'] >= 25) & (df['age'] <= 55)).astype(int)
    df['age_retirement_near'] = (df['age'] >= 60).astype(int)
    df['age_squared'] = df['age'] ** 2

    # BALANCE FEATURES
    df['has_debt'] = (df['balance'] < 0).astype(int)
    df['balance_positive'] = np.maximum(df['balance'], 0)
    df['balance_log'] = np.sign(df['balance']) * np.log1p(np.abs(df['balance']))
    df['high_balance'] = (df['balance'] > df['balance'].quantile(0.75)).astype(int)

    # DAY FEATURES
    df['day_start_month'] = (df['day'] <= 10).astype(int)
    df['day_mid_month'] = ((df['day'] > 10) & (df['day'] <= 20)).astype(int)
    df['day_end_month'] = (df['day'] > 20).astype(int)

    # DURATION FEATURES
    df['duration_minutes'] = df['duration'] / 60
    df['quick_hangup'] = (df['duration'] < 60).astype(int)
    df['engaged_call'] = (df['duration'] > 300).astype(int)
    df['duration_log'] = np.log1p(df['duration'])

    # CAMPAIGN FEATURES
    df['low_campaign'] = (df['campaign'] <= 2).astype(int)
    df['high_campaign'] = (df['campaign'] > 3).astype(int)
    df['campaign_log'] = np.log1p(df['campaign'])

    # PDAYS FEATURES
    df['was_contacted_before'] = (df['pdays'] != -1).astype(int)
    df['never_contacted'] = (df['pdays'] == -1).astype(int)
    df['recent_contact'] = ((df['pdays'] > 0) & (df['pdays'] <= 30)).astype(int)

    # PREVIOUS FEATURES
    df['had_previous_contacts'] = (df['previous'] > 0).astype(int)
    df['multiple_previous'] = (df['previous'] > 1).astype(int)
    df['previous_log'] = np.log1p(df['previous'])

    # INTERACTION FEATURES
    df['success_per_campaign'] = df['previous'] / (df['campaign'] + 1)
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)

    # CATEGORICAL FEATURES (ordinal + binary)
    education_order = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
    df['education_level'] = df['education'].map(education_order)

    df['prev_success'] = (df['poutcome'] == 'success').astype(int)
    df['prev_failure'] = (df['poutcome'] == 'failure').astype(int)
    df['has_housing_loan'] = (df['housing'] == 'yes').astype(int)
    df['has_personal_loan'] = (df['loan'] == 'yes').astype(int)
    df['has_default'] = (df['default'] == 'yes').astype(int)
    df['is_married'] = (df['marital'] == 'married').astype(int)
    df['is_single'] = (df['marital'] == 'single').astype(int)

# ONE-HOT ENCODING (same columns for both sets)
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
train = pd.get_dummies(train, columns=categorical_cols)
test = pd.get_dummies(test, columns=categorical_cols)

# Align columns between train and test
train, test = train.align(test, join='left', axis=1, fill_value=0)


test_ids = test['id'].copy()


X = train.drop(columns=['id', 'y'])
y = train['y']
X_test_external = test.drop(columns=['id'])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)


def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'verbosity': -1,
        'random_state': 42,
        'force_col_wise': True
    }

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_names=['train', 'valid'],
        eval_metric='auc'
    )

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    return auc


#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=20, timeout=3600)


#print("Best AUC:", study.best_value)
#print("Best params:", study.best_params)


# ---------- LIGHTGBM PARAMETERS ----------
best_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'verbosity': -1,
    'seed': 42,
    'max_depth': 8,
    'learning_rate': 0.06846044981194582,
    'num_leaves': 101,
    'min_child_samples': 100,
    'subsample': 0.5015317985049209,
    'colsample_bytree': 0.5088750464199135,
    'reg_alpha': 4.412448231575341,
    'reg_lambda': 7.133946853786371,
    'n_estimators': 1989
}

# ---------- CROSS VALIDATION ----------
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_aucs = []
test_preds = np.zeros(len(X_test_external))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    X_tr = X_tr.reindex(columns=X.columns, fill_value=0)
    X_val = X_val.reindex(columns=X.columns, fill_value=0)

    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='auc'
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    fold_aucs.append(auc)
    print(f"Fold {fold} AUC: {auc:.5f}")

    test_input = X_test_external.reindex(columns=X.columns, fill_value=0)
    test_preds += model.predict_proba(test_input)[:, 1]

# ---------- FINAL OUTPUT ----------
test_preds /= kf.n_splits
mean_auc = np.mean(fold_aucs)
print(f"Mean CV ROC AUC: {mean_auc:.5f}")


aucs = np.array(fold_aucs)

print(f"\nOverall Statistics:")
print(f"Mean AUC: {aucs.mean():.5f}")
print(f"Standard Deviation: {aucs.std():.5f}")
print(f"Min AUC: {aucs.min():.5f}")
print(f"Max AUC: {aucs.max():.5f}")


submission = pd.DataFrame({
    'id': test_ids,
    'y': test_preds / kf.n_splits
})
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")


submission.head()


plt.figure(figsize=(8, 5))
plt.hist(submission['y'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Predicted Probabilities (submission["y"])')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

