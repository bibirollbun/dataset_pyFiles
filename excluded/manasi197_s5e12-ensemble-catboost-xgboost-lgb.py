pip install -U scikit-learn imbalanced-learn


# Cell 1: Imports
import numpy as np
import pandas as pd
import gc
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
target = 'diagnosed_diabetes'
N_SPLITS = 5

# Set seeds
def seeds(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)

seeds()


# Cell 2: Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')

# Rename orig columns to match train
orig = orig.rename(columns={
    'Diabetes_binary': target,
    'Age': 'age',
    'BMI': 'bmi',
    'Sex': 'gender',
    'Education': 'education_level',
    'Income': 'income_level',
    'Smoker': 'smoking_status',
})

print(f"Train: {train.shape} | Test: {test.shape} | Orig: {orig.shape}")


# Cell 3: Outlier Removal (Optional - comment out if not needed)
def remove_outliers_iqr(df, n_outliers_per_col=50, features=None):
    df_clean = df.copy()
    if features is None:
        features = df_clean.select_dtypes(include=["int64", "float64"]).columns.drop([target, 'id'], errors='ignore').tolist()
    
    total_removed = 0
    for col in features:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)].index.tolist()
        outliers_to_drop = outliers[:n_outliers_per_col]
        df_clean = df_clean.drop(outliers_to_drop, errors='ignore')
        total_removed += len(outliers_to_drop)
    
    print(f"Removed {total_removed} total outliers")
    return df_clean

train = remove_outliers_iqr(train)


# Cell 4: ORIG Features (Mean and Count Encoding - only for overlapping columns)
# Get overlapping columns between train BASE and orig
overlapping_cols = [col for col in BASE if col in orig.columns and orig[col].dtype == train[col].dtype]

new_cols = []
orig_mean = orig[target].mean()

for col in overlapping_cols:
    # Mean encoding
    mean_map = orig.groupby(col)[target].mean()
    new_mean_col_name = f"orig_mean_{col}"
    train[new_mean_col_name] = train[col].map(mean_map)
    test[new_mean_col_name] = test[col].map(mean_map)
    train[new_mean_col_name] = train[new_mean_col_name].fillna(orig_mean)
    test[new_mean_col_name] = test[new_mean_col_name].fillna(orig_mean)
    new_cols.append(new_mean_col_name)

    # Count encoding
    new_cnt_col_name = f"orig_cnt_{col}"
    cnt_map = orig.groupby(col).size()
    train[new_cnt_col_name] = train[col].map(cnt_map)
    test[new_cnt_col_name] = test[col].map(cnt_map)
    train[new_cnt_col_name] = train[new_cnt_col_name].fillna(0)
    test[new_cnt_col_name] = test[new_cnt_col_name].fillna(0)
    new_cols.append(new_cnt_col_name)

print(f"Created {len(new_cols)} ORIG features from {len(overlapping_cols)} overlapping columns")


# Cell 5: Medical Features
def create_medical_features(df):
    df = df.copy()
    df['lipid_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
    df['tg_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-5)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['map_pressure'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
    df['bmi_waist_interaction'] = df['bmi'] * df['waist_to_hip_ratio']
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    max_alc, max_screen, max_pa, max_sleep = 14, 24, 600, 12
    df['lifestyle_risk_score'] = (
        (df['alcohol_consumption_per_week'] / max_alc) + 
        (df['screen_time_hours_per_day'] / max_screen) -  
        (df['physical_activity_minutes_per_week'] / max_pa) - 
        (df['sleep_hours_per_day'] / max_sleep)
    )
    df['high_bp_flag'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 85)).astype(int)
    df['high_tg_flag'] = (df['triglycerides'] >= 150).astype(int)
    df['low_hdl_flag'] = (df['hdl_cholesterol'] < 45).astype(int)
    df['obesity_flag'] = (df['bmi'] >= 30).astype(int)
    df['metabolic_risk_count'] = (
        df['high_bp_flag'] + df['high_tg_flag'] + df['low_hdl_flag'] + 
        df['obesity_flag'] + df['hypertension_history'] + df['family_history_diabetes']
    )
    df['age_group'] = pd.cut(df['age'], bins=[0, 35, 50, 65, 100], labels=[0, 1, 2, 3]).astype(int)
    return df

train = create_medical_features(train)
test = create_medical_features(test)


# Cell 6: Label Encoding for Categorical Features
CATS = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']

label_encoders = {}
for col in CATS:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]]).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le


# Cell 7: Memory Reduction
def reduce_mem_usage(df):
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
            else:
                df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")
    return df

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
gc.collect()


# Cell 8: Prepare Data
y = train[target]
X = train.drop(columns=[target, 'id'])
X_test = test.drop(columns=['id'])

# Cat features indices for CatBoost (recalculate after all features)
cat_features_indices = [X.columns.get_loc(col) for col in CATS if col in X.columns]

# SMOTE for imbalance
imbalance_ratio = y.mean()
if imbalance_ratio < 0.3:
    smote = SMOTE(random_state=RANDOM_STATE)
    X, y = smote.fit_resample(X, y)
    print(f"Applied SMOTE: {X.shape}")

X_np = X.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_np = y.values

print(f"X: {X_np.shape}, Test: {X_test_np.shape}")


# Cell 9: Define Models
models = [
    # CatBoost Main
    {
        'name': 'cb_main',
        'type': 'cb',
        'params': {
            'iterations': 1000, 'learning_rate': 0.1, 'depth': 6, 'l2_leaf_reg': 3,
            'eval_metric': 'AUC', 'random_seed': RANDOM_STATE, 'verbose': False,
            'early_stopping_rounds': 100
        }
    },
    # XGBoost
    {
        'name': 'xgb',
        'type': 'xgb',
        'params': {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'learning_rate': 0.05,
            'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 0.1, 'tree_method': 'hist', 'device': 'cuda',
            'seed': RANDOM_STATE, 'verbosity': 0
        },
        'n_estimators': 1000
    },
    # LightGBM
    {
        'name': 'lgb',
        'type': 'lgb',
        'params': {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'learning_rate': 0.05,
            'num_leaves': 31, 'max_depth': 6, 'min_child_samples': 20,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'random_state': RANDOM_STATE, 'verbosity': -1, 'device': 'gpu'
        },
        'n_estimators': 1000
    }
]


# Cell 10: Train Models & Collect OOF/Test Preds 
n_models = len(models)
oof_matrix = np.zeros((len(X_np), n_models))
test_matrix = np.zeros((len(X_test_np), n_models))
roc_scores = []  # Per model

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for idx, config in enumerate(models):
    print(f"\n[{idx+1}/{n_models}] Training {config['name']}...")
    fold_oof = np.zeros(len(X_np))
    fold_test = np.zeros(len(X_test_np))
    model_scores = []
    
    for fold_num, (tr_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
        print(f"  Fold {fold_num+1}/{N_SPLITS}")
        X_tr, X_val = X_np[tr_idx], X_np[val_idx]
        y_tr, y_val = y_np[tr_idx], y_np[val_idx]
        
        if config['type'] == 'cb':
            # Use DataFrame for CatBoost to handle cat_features
            X_tr_df = X.iloc[tr_idx].reset_index(drop=True)
            X_val_df = X.iloc[val_idx].reset_index(drop=True)
            train_pool = Pool(X_tr_df, y_tr, cat_features=cat_features_indices)
            valid_pool = Pool(X_val_df, y_val, cat_features=cat_features_indices)
            model = CatBoostClassifier(**config['params'])
            model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=100, verbose=100 if fold_num==0 and idx==0 else False)
            val_pred = model.predict_proba(valid_pool)[:, 1]
            test_df_cb = X_test.reset_index(drop=True)  # Ensure same structure
            test_pool = Pool(test_df_cb, cat_features=cat_features_indices)
            test_pred = model.predict_proba(test_pool)[:, 1]
            
        elif config['type'] == 'xgb':
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dvalid = xgb.DMatrix(X_val, label=y_val)
            dtest = xgb.DMatrix(X_test_np)
            evals = [(dtrain, 'train'), (dvalid, 'val')]
            model = xgb.train(config['params'], dtrain, num_boost_round=config['n_estimators'],
                              evals=evals, early_stopping_rounds=100, verbose_eval=False)
            best_iter = model.best_iteration
            val_pred = model.predict(dvalid, iteration_range=(0, best_iter))
            test_pred = model.predict(dtest, iteration_range=(0, best_iter))
            
        elif config['type'] == 'lgb':
            train_data = lgb.Dataset(X_tr, label=y_tr)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            callbacks = [lgb.early_stopping(100), lgb.log_evaluation(0)]
            model = lgb.train(config['params'], train_data, valid_sets=[valid_data],
                              num_boost_round=config['n_estimators'], callbacks=callbacks)
            best_iter = model.best_iteration
            val_pred = model.predict(X_val, num_iteration=best_iter)
            test_pred = model.predict(X_test_np, num_iteration=best_iter)
        
        fold_oof[val_idx] = val_pred
        fold_test += test_pred / N_SPLITS
        score = roc_auc_score(y_val, val_pred)
        model_scores.append(score)
        print(f"    Fold AUC: {score:.5f}")
    
    oof_matrix[:, idx] = fold_oof
    test_matrix[:, idx] = fold_test
    avg_score = np.mean(model_scores)
    roc_scores.append(avg_score)
    print(f"  → Avg OOF AUC: {avg_score:.5f} (+/- {np.std(model_scores):.5f})")

print(f"\nOverall Avg AUC: {np.mean(roc_scores):.5f}")


# Cell 11: Ridge Stacking
print("Training Ridge stacker...")
ridge = Ridge(alpha=1.0, positive=True, random_state=RANDOM_STATE)
ridge.fit(oof_matrix, y_np)

stacked_oof = ridge.predict(oof_matrix)
stacked_test = ridge.predict(test_matrix)

print(f"FINAL STACKED OOF AUC: {roc_auc_score(y_np, stacked_oof):.6f}")
print("Ridge coefficients:")
for i, coef in enumerate(ridge.coef_):
    print(f"  {models[i]['name']}: {coef:.4f}")


# Cell 12: Submission
submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': stacked_test.clip(0, 1)
})
submission.to_csv('submission_catboost_ensemble.csv', index=False)
print("Submission saved!")
print(submission.head())


# Cell 13: Save OOF for Analysis
oof_df = pd.DataFrame(oof_matrix, columns=[m['name'] for m in models])
oof_df['stacked'] = stacked_oof
oof_df['target'] = y_np
oof_df.to_csv('oof_catboost_ensemble.csv', index=False)
print("OOF saved!")


# Individual Submission Files 

# CatBoost Submission
cb_test = test_matrix[:, 0]
submission_cb = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': cb_test.clip(0, 1)
})
submission_cb.to_csv('submission_catboost.csv', index=False)
print("CatBoost submission saved!")
print(submission_cb.head())


# XGBoost Submission
xgb_test = test_matrix[:, 1]
submission_xgb = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': xgb_test.clip(0, 1)
})
submission_xgb.to_csv('submission_xgboost.csv', index=False)
print("XGBoost submission saved!")
print(submission_xgb.head())


# LightGBM Submission
lgb_test = test_matrix[:, 2]
submission_lgb = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': lgb_test.clip(0, 1)
})
submission_lgb.to_csv('submission_lightgbm.csv', index=False)
print("LightGBM submission saved!")
print(submission_lgb.head())


# Weighted Blending
weights = np.array(roc_scores) / np.sum(roc_scores)  # Normalize CV scores as weights
print(f"Weights: CB={weights[0]:.3f}, XGB={weights[1]:.3f}, LGBM={weights[2]:.3f}")

weighted_test = np.average(test_matrix, axis=1, weights=weights)
weighted_oof = np.average(oof_matrix, axis=1, weights=weights)

print(f"Weighted OOF AUC: {roc_auc_score(y_np, weighted_oof):.6f}")

# Weighted Submission
submission_weighted = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': weighted_test.clip(0, 1)
})
submission_weighted.to_csv('submission_weighted_blend.csv', index=False)
print("Weighted blend submission saved!")
print(submission_weighted.head())




