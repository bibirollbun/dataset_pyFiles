# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

import lightgbm as lgb
from  lightgbm import LGBMRegressor
import xgboost as  xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


missing = df_train.isnull().sum()
if missing.sum() > 0:
    print("\nMissing Values:")
    for col, count in missing[missing > 0].items():
        print(f"{col}: {count} ({count/len(df_train)*100:.2f}%)")
else:
    print("\nNo missing values deteced.")


numerical_cols_train = df_train.select_dtypes(include=[np.number]).columns.tolist()


categorical_cols_train = df_train.select_dtypes(include=['object']).columns.tolist()


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(df_train['accident_risk'], bins=50, density=True, alpha=0.7, 
                 color='steelblue', edgecolor='black')
axes[0].set_xlabel('Accident Risk', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Distribution of Accident Risk (Training Data)', 
                      fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)


stats.probplot(df_train['accident_risk'], dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot: Normality Assessment', 
                      fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

shapiro_stat, shapiro_p = stats.shapiro(df_train['accident_risk'].sample(min(5000, len(df_train))))
print("\nShapiro-Wilk Test for Normality:")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  P-value: {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("  Interpretation: Data is approximately Normal distribution")
else:
    print("  Interpretation: Data is NOT Normal distribution")



palette = sns.color_palette("husl", len(numerical_cols_train))

plt.figure(figsize=(25, 15))
for i, col in enumerate(numerical_cols_train, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df_train[col], kde=True, color=palette[i-1], bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


CATEGORICAL_FEATURES = ['road_type', 'lighting', 'weather', 'time_of_day']
BOOLEAN_FEATURES = ['road_signs_present', 'public_road', 'holiday', 'school_season']
NUMERICAL_FEATURES = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
TARGET = 'accident_risk'
ID_COL = 'id'


def engineer_features(df):
    """
    Create domain-informed feature interactions.
    """
    df_eng = df.copy()
    
    # Core interactions
    df_eng['curv_speed'] = df_eng['curvature'] * df_eng['speed_limit']
    df_eng['lane_speed'] = df_eng['num_lanes'] * df_eng['speed_limit']
    df_eng['accidents_speed'] = df_eng['num_reported_accidents'] * df_eng['speed_limit']
    df_eng['accidents_curv'] = df_eng['num_reported_accidents'] * df_eng['curvature']
    
    # Polynomial features
    df_eng['curvature_sq'] = df_eng['curvature'] ** 2
    df_eng['curvature_cube'] = df_eng['curvature'] ** 3
    df_eng['speed_sq'] = df_eng['speed_limit'] ** 2
    
    # Risk scores
    df_eng['risk_intensity'] = (df_eng['curvature'] * df_eng['speed_limit']) / 50
    df_eng['lane_capacity_risk'] = (5 - df_eng['num_lanes']) * df_eng['speed_limit']
    df_eng['accidents_per_lane'] = df_eng['num_reported_accidents'] / (df_eng['num_lanes'] + 1)
    
    # Binary indicators
    df_eng['high_risk_combo'] = ((df_eng['curvature'] > 0.5) & 
                                  (df_eng['speed_limit'] >= 60)).astype(int)
    
    return df_eng

# Preprocessing
train_processed = df_train.copy()
test_processed = df_test.copy()

# Convert booleans
for col in BOOLEAN_FEATURES:
    train_processed[col] = train_processed[col].astype(int)
    test_processed[col] = test_processed[col].astype(int)

# Label encode categoricals
label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    train_processed[f'{col}_enc'] = le.fit_transform(train_processed[col])
    test_processed[f'{col}_enc'] = le.transform(test_processed[col])
    label_encoders[col] = le

# Apply feature engineering
train_engineered = engineer_features(train_processed)
test_engineered = engineer_features(test_processed)

print(f"Feature engineering complete")
print(f"Original features: {len(CATEGORICAL_FEATURES + BOOLEAN_FEATURES + NUMERICAL_FEATURES)}")
print(f"Engineered features: {train_engineered.shape[1]}")
print(f"New features created: {train_engineered.shape[1] - train_processed.shape[1]}")


exclude_cols = [ID_COL, TARGET] + CATEGORICAL_FEATURES
feature_cols = [col for col in train_engineered.columns if col not in exclude_cols]

X_train = train_engineered[feature_cols].values
y_train = train_engineered[TARGET].values
X_test = test_engineered[feature_cols].values

print(f"Training matrix: {X_train.shape}")
print(f"Test matrix: {X_test.shape}")


N_SPLITS = 5
kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Storage for predictions
oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))

test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

models_xgb, models_lgb, models_cat = [], [], []
scores_xgb, scores_lgb, scores_cat = [], [], []


xgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'max_depth': 8,
    'min_child_weight': 2,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': 42,
    'tree_method': 'hist',
    'gpu_id': 0,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

print("Training XGBoost with GPU acceleration")
print("=" * 60)

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"Fold {fold_idx}/{N_SPLITS}", end=" ")
    
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        early_stopping_rounds=100,
        verbose=False
    )
    
    oof_xgb[val_idx] = model.predict(X_fold_val)
    test_xgb += model.predict(X_test) / N_SPLITS
    
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, oof_xgb[val_idx]))
    scores_xgb.append(fold_rmse)
    models_xgb.append(model)
    
    print(f"RMSE: {fold_rmse:.6f} | Best iter: {model.best_iteration}")

xgb_oof_rmse = np.sqrt(mean_squared_error(y_train, oof_xgb))
print(f"\nXGBoost OOF RMSE: {xgb_oof_rmse:.6f}")
print(f"CV Std: {np.std(scores_xgb):.6f}")
print("=" * 60)


ensemble_oof = oof_xgb
ensemble_test = test_xgb 


ens_oof_rmse = np.sqrt(mean_squared_error(y_train, ensemble_oof))
print(f"Weighted Ensemble OOF RMSE: {ens_oof_rmse:.6f}")
print("=" * 60)


submission = pd.DataFrame({
    'id': df_test['id'],
    'accident_risk': ensemble_test
})


submission.to_csv('/kaggle/working/submission.csv', index=False)




