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
import gc
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
gc.enable()


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

TARGET = 'diagnosed_diabetes'
ID_COL = 'id'
RANDOM_STATE = 42

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Target prevalence: {train[TARGET].mean():.3%}")


cat_cols = ['gender', 'ethnicity', 'education_level', 
            'income_level', 'smoking_status', 'employment_status']


def create_optimized_features(df):
    df = df.copy()
    
    # High-impact activity features
    df['log_physical_activity'] = np.log1p(df['physical_activity_minutes_per_week'])
    df['activity_per_age'] = df['physical_activity_minutes_per_week'] / (df['age'] + 10)
    df['physical_activity_squared'] = df['physical_activity_minutes_per_week'] ** 2
    
    # Strong interactions
    df['age_bmi_interaction'] = df['age'] * df['bmi'] / 100
    df['activity_diet_interaction'] = df['physical_activity_minutes_per_week'] * df['diet_score']
    df['age_squared'] = df['age'] ** 2
    
    # Lipid ratios
    df['chol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-6)
    df['tg_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-6)
    df['non_hdl_chol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    # Blood pressure
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['map'] = df['diastolic_bp'] + df['pulse_pressure'] / 3
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1e-6)
    
    # Risk scores
    df['genetic_risk_score'] = (df['family_history_diabetes'] * 3 +
                                df['hypertension_history'] * 2 +
                                df['cardiovascular_history'] * 4)
    
    df['lifestyle_score'] = (df['physical_activity_minutes_per_week'] / 100 +
                             df['diet_score'] -
                             df['screen_time_hours_per_day'] / 10)
    
    # Heart rate & LDL interactions
    df['heart_rate_age'] = df['heart_rate'] * df['age'] / 100
    df['ldl_bmi'] = df['ldl_cholesterol'] * df['bmi'] / 100
    
    # Metabolic risk flag
    df['metabolic_risk_count'] = (
        (df['bmi'] >= 30).astype(int) +
        (df['tg_hdl_ratio'] > 3.5).astype(int) +
        (df['systolic_bp'] >= 130).astype(int) +
        (df['family_history_diabetes'] == 1).astype(int)
    )
    
    # Log transforms
    for col in ['triglycerides', 'ldl_cholesterol']:
        df[f'log_{col}'] = np.log1p(df[col])
    
    return df

print("\n" + "="*60)
print("CREATING OPTIMIZED FEATURES")
print("="*60)

train = create_optimized_features(train)
test = create_optimized_features(test)


cluster_cols = ['age', 'bmi', 'physical_activity_minutes_per_week', 
                'chol_hdl_ratio', 'tg_hdl_ratio', 'map', 'lifestyle_score']

cluster_cols = [c for c in cluster_cols if c in train.columns]

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[cluster_cols])
test_scaled = scaler.transform(test[cluster_cols])

# Clustering
kmeans = KMeans(n_clusters=7, random_state=RANDOM_STATE, n_init=10)
train['cluster'] = kmeans.fit_predict(train_scaled)
test['cluster'] = kmeans.predict(test_scaled)

# PCA
pca = PCA(n_components=3, random_state=RANDOM_STATE)
train[['pca_1', 'pca_2', 'pca_3']] = pca.fit_transform(train_scaled)
test[['pca_1', 'pca_2', 'pca_3']] = pca.transform(test_scaled)

print(f"Created clustering + PCA features")


all_features = [c for c in train.columns if c not in [TARGET, ID_COL]]

# Correlation-based ranking
corrs = train[all_features + [TARGET]].corr(numeric_only=True)[TARGET].abs().sort_values(ascending=False)
top_corr = corrs.index[1:60].tolist()

# Must-keep from previous top importance
must_keep = [
    'physical_activity_minutes_per_week', 'log_physical_activity',
    'age_bmi_interaction', 'activity_diet_interaction',
    'family_history_diabetes', 'genetic_risk_score',
    'pca_1', 'heart_rate', 'ldl_cholesterol', 'screen_time_hours_per_day'
]

FEATURES = list(set(top_corr + must_keep + cat_cols))
FEATURES = [f for f in FEATURES if f in train.columns]

print(f"Total features selected: {len(FEATURES)}")
print("Top 10 by correlation:")
print(corrs.head(11))


X = train[FEATURES].copy()
y = train[TARGET].copy()
X_test = test[FEATURES].copy()

# Label encode categoricals
for col in cat_cols:
    if col in FEATURES:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))


scale_pos_weight = (1 - y.mean()) / y.mean()  # ~0.604

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 2500,
    'learning_rate': 0.03,
    'num_leaves': 90,
    'max_depth': 9,
    'min_child_samples': 40,
    'subsample': 0.85,
    'colsample_bytree': 0.78,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'scale_pos_weight': scale_pos_weight,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1,
    'device': 'gpu'
}

xgb_params = {
    'n_estimators': 2200,
    'learning_rate': 0.03,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.78,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': scale_pos_weight,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cuda'
}

cat_params = {
    'iterations': 2200,
    'learning_rate': 0.04,
    'depth': 8,
    'l2_leaf_reg': 3.0,
    'random_strength': 0.8,
    'bagging_temperature': 0.6,
    'od_type': 'Iter',
    'od_wait': 150,
    'random_seed': RANDOM_STATE,
    'verbose': False,
    'task_type': 'GPU',
    'devices': '0',
    'class_weights': [1.0, scale_pos_weight] 
}

print(f"Scale pos weight: {scale_pos_weight:.3f}")
print("GPU enabled for all models")


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

cv_scores = []

print("\n" + "="*60)
print("ENSEMBLE TRAINING WITH DYNAMIC WEIGHT OPTIMIZATION")
print("="*60)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}/{n_splits}")
    
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    # LightGBM
    print("  Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    lgb_val = lgb_model.predict_proba(X_val)[:, 1]
    lgb_test = lgb_model.predict_proba(X_test)[:, 1]
    
    # XGBoost
    print("  Training XGBoost...")
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test = xgb_model.predict_proba(X_test)[:, 1]
    
    # CatBoost
    print("  Training CatBoost...")
    cat_model = CatBoostClassifier(**cat_params)
    cat_indices = [i for i, c in enumerate(FEATURES) if c in cat_cols]
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_indices, verbose=False)
    cat_val = cat_model.predict_proba(X_val)[:, 1]
    cat_test = cat_model.predict_proba(X_test)[:, 1]
    
    # Simple grid for best blend
    best_auc = 0
    best_w = (0.5, 0.3, 0.2)
    for w1 in [0.45, 0.5, 0.55]:
        for w2 in [0.25, 0.3, 0.35]:
            w3 = 1 - w1 - w2
            if w3 < 0.1: continue
            blend = w1 * lgb_val + w2 * xgb_val + w3 * cat_val
            auc = roc_auc_score(y_val, blend)
            if auc > best_auc:
                best_auc = auc
                best_w = (w1, w2, w3)
    
    print(f"  Fold AUC: LGB {roc_auc_score(y_val, lgb_val):.6f} | "
          f"XGB {roc_auc_score(y_val, xgb_val):.6f} | "
          f"CAT {roc_auc_score(y_val, cat_val):.6f} | "
          f"Blend {best_auc:.6f} (weights {best_w})")
    
    cv_scores.append(best_auc)
    
    # Apply best weights to test
    lgb_preds += lgb_test * best_w[0] / n_splits
    xgb_preds += xgb_test * best_w[1] / n_splits
    cat_preds += cat_test * best_w[2] / n_splits
    
    gc.collect()

print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS")
print("="*60)
print(f"Mean CV AUC: {np.mean(cv_scores):.6f} (±{np.std(cv_scores):.6f})")


final_preds = lgb_preds + xgb_preds + cat_preds  # Already weighted

# Light mean calibration
target_mean = train[TARGET].mean()
pred_mean = final_preds.mean()
if abs(pred_mean - target_mean) > 0.005:
    final_preds = final_preds * (target_mean / pred_mean)

final_preds = np.clip(final_preds, 0.001, 0.999)


submission[TARGET] = final_preds
submission.to_csv('submission.csv', index=False)

print("\n" + "="*60)
print("SUBMISSION CREATED")
print("="*60)
print(f"Prediction mean: {final_preds.mean():.4f} (target: {target_mean:.4f})")
print(f"Range: [{final_preds.min():.4f}, {final_preds.max():.4f}]")
print("\nFirst 5 predictions:")
print(submission.head())

# Feature importance (last LGB)
importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 important features:")
print(importance.head(10))

