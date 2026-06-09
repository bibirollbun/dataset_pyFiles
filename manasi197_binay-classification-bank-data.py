# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import category_encoders as ce
import gc
import lightgbm as lgb
import warnings
warnings.simplefilter('ignore')

# Configuration
class CFG:
    state = 42
    n_splits = 5
    target = 'y'
    data_path = '/kaggle/input/playground-series-s5e8'
    orig_path = '/kaggle/input/bank-marketing-dataset-full/bank-full.csv'
    output_path = '/kaggle/working'
    cat_feats = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

np.random.seed(CFG.state)



# Data Loading
train = pd.read_csv(f'{CFG.data_path}/train.csv', index_col='id')
test = pd.read_csv(f'{CFG.data_path}/test.csv', index_col='id')
original = pd.read_csv(CFG.orig_path, sep=';')

# Combine original data with training data
if 'y' in original.columns:
    original['y'] = original['y'].map({'yes': 1, 'no': 0})
else:
    raise KeyError("Target column 'y' not found in original dataset")

train = pd.concat([train, original], axis=0, ignore_index=True)




# Micro Feature Engineering
def micro_fe(df):
    df = df.copy()
    
    def f1(x):
        if x['education'] == 'unknown' and x['contact'] == 'unknown' and x['poutcome'] == 'unknown':
            return 21
        if (x['education'] == 'unknown' and x['contact'] == 'unknown') or \
           (x['education'] == 'unknown' and x['poutcome'] == 'unknown') or \
           (x['contact'] == 'unknown' and x['poutcome'] == 'unknown'):
            return 7
        if x['education'] == 'unknown' or x['contact'] == 'unknown' or x['poutcome'] == 'unknown':
            return 3
        return 0
    
    def f2(x):
        if x['default'] == 'no' and x['housing'] == 'no' and x['loan'] == 'no':
            return 21
        if (x['default'] == 'no' and x['housing'] == 'no') or \
           (x['default'] == 'no' and x['loan'] == 'no') or \
           (x['housing'] == 'no' and x['loan'] == 'no'):
            return 7
        if x['default'] == 'no' or x['housing'] == 'no' or x['loan'] == 'no':
            return 3
        return 0
    
    df['unknowns'] = df.apply(lambda x: f1(x), axis=1)
    df['many_no'] = df.apply(lambda x: f2(x), axis=1)
    
    return df


# Additional Feature Engineering
def create_features(df):
    # Interaction features
    df['balance_duration'] = df['balance'] * df['duration']
    df['campaign_pdays'] = df['campaign'] * df['pdays']
    df['duration_poutcome'] = df['duration'] * df['poutcome']
    df['age_duration'] = df['age'] * df['duration']
    
    # Polynomial features
    df['duration_squared'] = df['duration'] ** 2
    
    # Binning features
    df['age_bin'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4], include_lowest=True)
    df['age_bin'] = df['age_bin'].cat.add_categories([-1]).fillna(-1).astype(int)
    df['duration_bin'] = pd.cut(df['duration'], bins=[0, 100, 300, 600, 1000, float('inf')], 
                               labels=[0, 1, 2, 3, 4], include_lowest=True)
    df['duration_bin'] = df['duration_bin'].cat.add_categories([-1]).fillna(-1).astype(int)
    
    # Pdays binary feature
    df['pdays_contacted'] = (df['pdays'] > -1).astype(int)
    
    return df


# Preprocessing
def preprocess_data(df, is_train=True, target_encoder=None):
    # Handle missing values in numerical columns
    num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Target encoding for categorical columns
    cat_cols = CFG.cat_feats
    if is_train:
        target_encoder = ce.TargetEncoder(cols=cat_cols, smoothing=1.0)
        df[cat_cols] = target_encoder.fit_transform(df[cat_cols], df[CFG.target])
    else:
        df[cat_cols] = target_encoder.transform(df[cat_cols])
    
    # Scale numerical columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df, target_encoder

# Apply micro feature engineering
train = micro_fe(train)
test = micro_fe(test)

# Apply additional feature engineering
train = create_features(train)
test = create_features(test)


# Apply preprocessing
train, target_encoder = preprocess_data(train, is_train=True)
test, _ = preprocess_data(test, is_train=False, target_encoder=target_encoder)

# Define features and target
X = train.drop(columns=[CFG.target])
y = train[CFG.target]
X_test = test




# LightGBM Model (First Run)
lgb_params = {
    'n_estimators': 20000,
    'learning_rate': 0.06,
    'num_leaves': 100,
    'max_depth': 10,
    'min_child_samples': 9,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'reg_alpha': 0.79,
    'reg_lambda': 3.0,
    'max_bin': 4523,
    'random_state': CFG.state,
    'verbosity': -1,
    'is_unbalance': True
}


import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import gc

# Modified Feature Engineering to ensure numerical output
def create_features(df):
    # Ensure poutcome is numerical before creating interaction
    if df['poutcome'].dtype == 'object' or df['poutcome'].dtype.name == 'category':
        le = LabelEncoder()
        df['poutcome'] = le.fit_transform(df['poutcome'].astype(str))
    
    # Interaction features
    df['balance_duration'] = df['balance'] * df['duration']
    df['campaign_pdays'] = df['campaign'] * df['pdays']
    df['duration_poutcome'] = df['duration'] * df['poutcome'].astype(float)
    df['age_duration'] = df['age'] * df['duration']
    
    # Polynomial features
    df['duration_squared'] = df['duration'] ** 2
    
    # Binning features
    df['age_bin'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4], include_lowest=True)
    df['age_bin'] = df['age_bin'].cat.add_categories([-1]).fillna(-1).astype(int)
    df['duration_bin'] = pd.cut(df['duration'], bins=[0, 100, 300, 600, 1000, float('inf')], 
                               labels=[0, 1, 2, 3, 4], include_lowest=True)
    df['duration_bin'] = df['duration_bin'].cat.add_categories([-1]).fillna(-1).astype(int)
    
    # Pdays binary feature
    df['pdays_contacted'] = (df['pdays'] > -1).astype(int)
    
    return df

# Re-apply feature engineering
train = create_features(train)
test = create_features(test)

# Ensure all features have numerical dtypes
def ensure_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median() if df[col].notna().any() else 0)
            except:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        elif df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    return df

# Apply dtype correction
X = ensure_numeric(X)
X_test = ensure_numeric(X_test)

# LightGBM Model (First Run)
kf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.state)
oof_preds = np.zeros(len(X))
y_probs_lgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training LightGBM fold {fold + 1}/{CFG.n_splits} >>>")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = LGBMClassifier(**lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Predict OOF for AUC calculation
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
    # Predict test
    y_probs_lgb += model.predict_proba(X_test)[:, 1] / CFG.n_splits
    
    # Clean up
    gc.collect()

# Calculate AUC
auc_score = roc_auc_score(y, oof_preds)
print(f'LightGBM OOF AUC Score: {auc_score:.4f}')

# Feature Importance
if hasattr(model, 'feature_importances_'):
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nLightGBM Feature Importance:\n", importances)

# Save LightGBM submission
submission_lgb = pd.DataFrame({'id': test.index, 'y': y_probs_lgb})
submission_lgb.to_csv('submission_no_data.csv', index=False)

# CatBoost Model (Second Model)
cat_params = {
    'iterations': 2000,
    'depth': 10,
    'learning_rate': 0.06,
    'random_seed': CFG.state,
    'verbose': 100,
    'early_stopping_rounds': 100
}

oof_preds_cat = np.zeros(len(X))
y_probs_cat = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training CatBoost fold {fold + 1}/{CFG.n_splits} >>>")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=100
    )
    
    # Predict OOF for AUC calculation
    oof_preds_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
    
    # Predict test
    y_probs_cat += model_cat.predict_proba(X_test)[:, 1] / CFG.n_splits
    
    # Clean up
    gc.collect()

# Calculate CatBoost AUC
auc_score_cat = roc_auc_score(y, oof_preds_cat)
print(f'CatBoost OOF AUC Score: {auc_score_cat:.4f}')

# Save CatBoost submission
submission_cat = pd.DataFrame({'id': test.index, 'y': y_probs_cat})
submission_cat.to_csv('submission_catboost.csv', index=False)

# Custom Weighted Blend (Replacement for hv_blend)
def custom_blend(submissions, weights):
    blended_y = np.zeros(len(submissions[0]['y']))
    for sub, weight in zip(submissions, weights):
        blended_y += sub['y'] * weight
    return pd.DataFrame({'id': submissions[0]['id'], 'y': blended_y})

# Blend LightGBM and CatBoost submissions
submissions = [submission_lgb, submission_cat]
weights = [0.8, 0.2]  # Inspired by high weight for best submission (0.805 in original hv_blend)
final_submission = custom_blend(submissions, weights)
final_submission.to_csv('submission.csv', index=False)
display(final_submission)




