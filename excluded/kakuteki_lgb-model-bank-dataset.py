# packages
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings; warnings.filterwarnings('ignore')

# data
path        = '/kaggle/input/playground-series-s5e8/'
train       = pd.read_csv(path + 'train.csv',             index_col = 'id')
test        = pd.read_csv(path + 'test.csv',              index_col = 'id')
submission  = pd.read_csv(path + 'sample_submission.csv', index_col = 'id')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Feature Engineering
def feature_engineering(df):
    df = df.copy()
    
    # Age binning
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], 
                            labels=['young', 'adult', 'middle', 'senior', 'elderly'])
    
    # Balance interaction features
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_log'] = np.log1p(df['balance'] + abs(df['balance'].min()) + 1)
    
    # Duration features
    if 'duration' in df.columns:
        df['duration_log'] = np.log1p(df['duration'])
        df['duration_group'] = pd.cut(df['duration'], bins=5, labels=False)
    
    # Campaign features
    df['campaign_log'] = np.log1p(df['campaign'])
    df['high_campaign'] = (df['campaign'] > df['campaign'].median()).astype(int)
    
    # Previous contact features
    df['has_previous'] = (df['previous'] > 0).astype(int)
    df['previous_log'] = np.log1p(df['previous'])
    
    # Day features
    df['day_group'] = pd.cut(df['day'], bins=[0, 10, 20, 31], 
                            labels=['early', 'mid', 'late'])
    
    # Interaction features
    df['job_education'] = df['job'].astype(str) + '_' + df['education'].astype(str)
    df['marital_housing'] = df['marital'].astype(str) + '_' + df['housing'].astype(str)
    
    return df

# Apply feature engineering
train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

# Categorical columns (original + new)
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
           'contact', 'month', 'poutcome', 'age_group', 'day_group',
           'job_education', 'marital_housing']

# Handle categorical features
label_encoders = {}
for feature in cat_cols:
    le = LabelEncoder()
    
    # Fit on combined data to ensure consistency
    combined_data = pd.concat([train_fe[feature].astype(str), test_fe[feature].astype(str)])
    le.fit(combined_data)
    
    train_fe[feature] = le.transform(train_fe[feature].astype(str))
    test_fe[feature] = le.transform(test_fe[feature].astype(str))
    
    label_encoders[feature] = le

# Convert to category type for LightGBM
for feature in cat_cols:
    train_fe[feature] = train_fe[feature].astype("category")
    test_fe[feature] = test_fe[feature].astype("category")

# Features and target
X = train_fe.drop(columns = 'y')
y = train_fe.y

print(f"Final feature count: {X.shape[1]}")

# Improved LightGBM parameters
model_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'max_depth': 6,
    'num_leaves': 63,
    'n_estimators': 5000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'categorical_feature': cat_cols,
    'verbosity': -1,
    'random_state': 42
}

# 10-fold cross-validation for better stability
n_splits = 10
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
lgb_preds = np.zeros(test_fe.shape[0])
oof_preds = np.zeros(len(X))
val_aucs = []
val_losses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold + 1}/{n_splits}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model_lgb = lgb.LGBMClassifier(**model_params)
    
    model_lgb.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(period=500)]
    )
    
    # Predict probabilities on the validation set
    val_preds = model_lgb.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    # Calculate metrics
    val_auc = roc_auc_score(y_val, val_preds)
    val_loss = log_loss(y_val, val_preds)
    val_aucs.append(val_auc)
    val_losses.append(val_loss)
    
    print(f"Fold {fold + 1} - AUC: {val_auc:.6f}, Log Loss: {val_loss:.6f}")
    
    # Predict probabilities on the test set
    test_preds = model_lgb.predict_proba(test_fe)[:, 1]
    lgb_preds += test_preds / n_splits

# Calculate overall metrics
overall_auc = roc_auc_score(y, oof_preds)
overall_loss = log_loss(y, oof_preds)
avg_val_auc = np.mean(val_aucs)
avg_val_loss = np.mean(val_losses)

print(f"\n=== Final Results ===")
print(f"Overall OOF AUC: {overall_auc:.6f}")
print(f"Overall OOF Log Loss: {overall_loss:.6f}")
print(f"Average CV AUC: {avg_val_auc:.6f} (+/- {np.std(val_aucs):.6f})")
print(f"Average CV Log Loss: {avg_val_loss:.6f} (+/- {np.std(val_losses):.6f})")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model_lgb.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Submission
submission['y'] = lgb_preds
submission.to_csv('submission.csv')
print(f"\nSubmission statistics:")
print(f"Min: {lgb_preds.min():.6f}")
print(f"Max: {lgb_preds.max():.6f}")
print(f"Mean: {lgb_preds.mean():.6f}")
print(f"Std: {lgb_preds.std():.6f}")

submission.head(10)

