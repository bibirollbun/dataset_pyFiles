import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ----------------------
# Preprocessing Function
# ----------------------
def preprocess_data(df, encoders=None):
    df = df.copy()
    df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
    df['loan'] = df['loan'].map({'yes': 1, 'no': 0})
    df['default'] = df['default'].map({'yes': 1, 'no': 0})

    cols_to_encode = ['job', 'marital', 'education', 'contact', 'poutcome', 'month']

    if encoders is None:
        encoders = {}
        for col in cols_to_encode:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        df['pdays'] = df['pdays'].replace(-1, 999)
        return df, encoders
    else:
        for col in cols_to_encode:
            le = encoders[col]
            df[col] = df[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
            if '<unknown>' in df[col].values and '<unknown>' not in le.classes_:
                le.classes_ = np.append(le.classes_, '<unknown>')
            df[col] = le.transform(df[col])
        df['pdays'] = df['pdays'].replace(-1, 999)
        return df, None

# ----------------------
# Feature Engineering
# ----------------------
def feature_engineering(df):
    df = df.copy()

    # Log transformations
    df['duration_log'] = np.log1p(df['duration'])
    df['campaign_log'] = np.log1p(df['campaign'])
    df['pdays_log'] = np.log1p(df['pdays'] + 2)
    df['previous_log'] = np.log1p(df['previous'] + 1)

    # Interaction features
    df['duration_balance'] = df['duration'] * df['balance']
    df['duration_age'] = df['duration'] * df['age']

    # Binary flags
    df['is_month_end'] = (df['day'] >= 26).astype(int)
    df['is_month_start'] = (df['day'] <= 5).astype(int)

    # High value behavior flag
    df['high_value_behavior'] = np.where(
        (df['previous'] > 0) &
        (df['poutcome'] == 'success') &
        (df['duration'] > 300),
        1, 0
    )

    # Job and education mismatch feature
    mismatch = (
        ((df['job'] == 'admin.') & (df['education'] == 'primary')) |
        ((df['job'] == 'management') & (df['education'] == 'primary')) |
        ((df['job'] == 'technician') & (df['education'] == 'unknown'))
    )
    df['job_education_mismatch'] = mismatch.astype(int)

    # Marital debt stress feature
    df['marital_debt_stress'] = np.where(
        (df['marital'] == 'single') &
        (df['housing'] == 1) &
        (df['balance'] < 0),
        1, 0
    )

    # Age groups
    df['age_group'] = pd.cut(
        df['age'], bins=[0, 25, 35, 50, 65, 97],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # Duration bins
    df['duration_bin'] = pd.cut(
        df['duration'], bins=[0, 60, 300, 600, float('inf')],
        labels=[0, 1, 2, 3], right=False
    ).astype(int)

    # Additional binary flags
    df['has_prev_contact'] = (df['previous'] > 0).astype(int)
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['was_contacted'] = (df['pdays'] != 999).astype(int)

    df['balance_posi'] = (df['balance'] > 0).astype(int)
    df['has_previous'] = (df['previous'] > 0).astype(int)
    df['long_duration'] = (df['duration'] >= 360).astype(int)
    df['campaign_multi'] = (df['campaign'] >= 2).astype(int)
    df['is_first_contact'] = (df['campaign'] == 1).astype(int)
    df['high_campaign'] = (df['campaign'] >= 3).astype(int)

    # Month sin/cos
    if 'month_num' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    # Numeric transforms
    df['log_duration'] = np.log1p(df['duration'])
    df['sqrt_duration'] = np.sqrt(df['duration'])
    df['log_campaign'] = np.log1p(df['campaign'])
    df['sqrt_age'] = df['age'] ** 2
    df['cubed_age'] = df['age'] ** 3
    df['log_age'] = np.log1p(df['age'])

    return df

# ----------------------
# Load and Process Data
# ----------------------
print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

test_ids = test['id']
train = train.drop(columns='id')
test = test.drop(columns='id')

train_processed, fitted_encoders = preprocess_data(train)
train_final = feature_engineering(train_processed)

test_processed, _ = preprocess_data(test, encoders=fitted_encoders)
test_final = feature_engineering(test_processed)

X = train_final.drop(columns=['y'])
y = train_final['y']
X_test = test_final

# ----------------------
# Best Parameters (from Optuna)
# ----------------------
best_params = {
    'max_depth': 15,
    'num_leaves': 233,
    'learning_rate': 0.022817373724007342,
    'min_child_samples': 90,
    'subsample': 0.971038373471942,
    'subsample_freq': 1,
    'colsample_bytree': 0.9344757879356346,
    'reg_alpha': 2.740557135879951,
    'reg_lambda': 0.17084831541669024,
    'min_split_gain': 0.2764600717378033,
    'max_bin': 4000,
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'seed': 42,
    'n_jobs': -1
}

# ----------------------
# Train Final Model
# ----------------------
print("Training LightGBM model on 100% of the training data...")
final_train_data = lgb.Dataset(X, label=y)
final_model = lgb.train(
    best_params,
    final_train_data,
    num_boost_round=10000,
    valid_sets=[final_train_data],
    callbacks=[lgb.log_evaluation(100)]
)

# ----------------------
# Predict on Test Data
# ----------------------
print("Making predictions...")
test_probas = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# Save submission
submission = pd.DataFrame({'id': test_ids, 'y': test_probas})
submission.to_csv('submission.csv', index=False)

print("✅ Submission saved as 'submission.csv'")



print("changes done to num_boost_round as many notebooks had as high as 20k so I change my 2,5k to 10k , moreover took advice from another kaggler and increased my max_bin from the best params , other params remained same from what optuna gave me ")

