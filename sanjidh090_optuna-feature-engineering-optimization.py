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


# Optimized Road Accident Risk Prediction with Optuna & Adaptive Feature Engineering
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import scipy.stats
import optuna
from optuna.samplers import TPESampler

# ======================== DATA LOADING ========================
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

# Load original data for target encoding
orig_dfs = []
for k in [2, 10, 100]:
    df = pd.read_csv(f"/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_{k}k.csv")
    orig_dfs.append(df)
orig = pd.concat(orig_dfs, axis=0, ignore_index=True)
orig['id'] = np.arange(len(orig)) + test['id'].max() + 1
orig = orig[train.columns]

TARGET = 'accident_risk'
print(f"Train: {train.shape}, Test: {test.shape}, Original: {orig.shape}")

# ======================== ADAPTIVE FEATURE ENGINEERING ========================
def engineer_features(df, trial=None):
    """Adaptive feature engineering controlled by Optuna trial"""
    df = df.copy()
    
    # Default values if no trial
    use_interaction = trial.suggest_categorical('use_interaction', [True, False]) if trial else True
    use_polynomial = trial.suggest_categorical('use_polynomial', [True, False]) if trial else True
    use_binning = trial.suggest_categorical('use_binning', [True, False]) if trial else True
    use_risk_scores = trial.suggest_categorical('use_risk_scores', [True, False]) if trial else True
    use_complex_interactions = trial.suggest_categorical('use_complex_interactions', [True, False]) if trial else True
    poly_degree = trial.suggest_int('poly_degree', 2, 3) if trial else 2
    n_bins = trial.suggest_int('n_bins', 3, 5) if trial else 4
    
    if use_risk_scores:
        # High-risk combinations
        df['night_bad_weather'] = ((df['lighting'] == 'night') & (df['weather'] != 'clear')).astype(int)
        df['high_speed_curve'] = ((df['speed_limit'] >= 60) & (df['curvature'] > 0.5)).astype(int)
        df['freq_accidents_high_speed'] = ((df['num_reported_accidents'] > 2) & (df['speed_limit'] >= 60)).astype(int)
        
        # Risk score components
        df['speed_risk'] = (df['speed_limit'] >= 60).astype(int) * df['speed_limit'] / 100
        df['accident_history_risk'] = np.clip(df['num_reported_accidents'] / 5, 0, 1)
        df['curvature_risk'] = np.clip(df['curvature'], 0, 1)
        
        # Weather severity encoding
        weather_severity = {'clear': 0, 'fog': 0.3, 'rain': 0.5, 'snow': 0.7}
        df['weather_severity'] = df['weather'].map(weather_severity).fillna(0.5)
        
        # Lighting risk
        lighting_risk = {'day': 0, 'dusk': 0.3, 'night': 0.5}
        df['lighting_risk'] = df['lighting'].map(lighting_risk).fillna(0.25)
    
    if use_interaction:
        # Basic interactions
        df['curvature_x_speed'] = df['curvature'] * df['speed_limit'] / 100
        df['accidents_x_curvature'] = df['num_reported_accidents'] * df['curvature']
        df['speed_x_accidents'] = (df['speed_limit'] / 100) * df['num_reported_accidents']
        
        if use_risk_scores:
            df['weather_x_lighting'] = df['weather_severity'] * df['lighting_risk']
    
    if use_complex_interactions:
        # Three-way interactions
        df['speed_curve_accidents'] = (df['speed_limit'] / 100) * df['curvature'] * df['num_reported_accidents']
        if use_risk_scores:
            df['risk_composite'] = df['speed_risk'] * df['curvature_risk'] * df['accident_history_risk']
    
    if use_polynomial:
        # Polynomial features
        df['curvature_pow'] = df['curvature'] ** poly_degree
        df['speed_pow'] = (df['speed_limit'] / 100) ** poly_degree
        df['accidents_pow'] = df['num_reported_accidents'] ** poly_degree
        
        # Log transforms
        df['log_speed'] = np.log1p(df['speed_limit'])
        df['log_accidents'] = np.log1p(df['num_reported_accidents'])
        df['sqrt_curvature'] = np.sqrt(df['curvature'])
    
    if use_binning:
        # Binned features
        df['speed_bin'] = pd.cut(df['speed_limit'], bins=n_bins, labels=False).astype(int)
        df['curvature_bin'] = pd.cut(df['curvature'], bins=n_bins, labels=False).astype(int)
        df['accidents_bin'] = pd.cut(df['num_reported_accidents'], bins=n_bins, labels=False).astype(int)
    
    return df

# ======================== SYNTHETIC TARGET (Y) ========================
def compute_synthetic_target(X):
    return (0.4 * X["curvature"] +
            0.1 * (X["lighting"] == "night").astype(int) +
            0.1 * (X["weather"] != "clear").astype(int) +
            0.1 * (X["speed_limit"] >= 60).astype(int) +
            0.2 * (X["num_reported_accidents"] > 2).astype(int))

def clip_target(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a) + sigma*(phi_a-phi_b) + 1 - Phi_b
    return clip_f

# ======================== OPTUNA OBJECTIVE ========================
def objective(trial):
    """Optuna objective function for hyperparameter tuning"""
    
    # Apply feature engineering with trial
    train_fe = engineer_features(train, trial)
    test_fe = engineer_features(test, trial)
    orig_fe = engineer_features(orig, trial)
    
    # Combine all data
    combine = pd.concat([train_fe, test_fe, orig_fe], axis=0, ignore_index=True)
    combine["y"] = clip_target(compute_synthetic_target)(combine).values
    
    # Identify all categorical columns
    CATS = combine.select_dtypes(include=['object']).columns.tolist()
    CATS = [c for c in CATS if c not in ['id', TARGET]]
    
    # Label encode ALL categoricals
    for c in CATS:
        le = LabelEncoder()
        combine[c] = le.fit_transform(combine[c].astype(str))
        combine[c] = combine[c].astype('int32')
    
    # Split back
    train_split = combine.iloc[:len(train)].copy()
    test_split = combine.iloc[len(train):len(train)+len(test)].copy()
    orig_split = combine.iloc[-len(orig):].copy()
    
    # Target encoding settings
    use_te_mean = trial.suggest_categorical('use_te_mean', [True, False])
    use_te_std = trial.suggest_categorical('use_te_std', [True, False])
    use_te_interactions = trial.suggest_categorical('use_te_interactions', [True, False])
    
    TE_FEATURES = []
    base_features = ['lighting', 'weather', 'speed_limit', 'curvature', 'num_reported_accidents']
    if 'speed_bin' in train_split.columns:
        base_features.extend(['speed_bin', 'curvature_bin', 'accidents_bin'])
    
    for c in base_features:
        if c not in orig_split.columns:
            continue
        
        if use_te_mean:
            te_mean = orig_split.groupby(c)[TARGET].mean()
            te_col = f"TE_mean_{c}"
            train_split[te_col] = train_split[c].map(te_mean).fillna(orig_split[TARGET].mean())
            test_split[te_col] = test_split[c].map(te_mean).fillna(orig_split[TARGET].mean())
            TE_FEATURES.append(te_col)
        
        if use_te_std:
            te_std = orig_split.groupby(c)[TARGET].std()
            te_col_std = f"TE_std_{c}"
            train_split[te_col_std] = train_split[c].map(te_std).fillna(orig_split[TARGET].std())
            test_split[te_col_std] = test_split[c].map(te_std).fillna(orig_split[TARGET].std())
            TE_FEATURES.append(te_col_std)
    
    if use_te_interactions:
        interaction_pairs = [('lighting', 'weather')]
        if 'speed_bin' in train_split.columns:
            interaction_pairs.extend([('speed_bin', 'curvature_bin'), ('weather', 'speed_bin')])
        
        for c1, c2 in interaction_pairs:
            if c1 not in orig_split.columns or c2 not in orig_split.columns:
                continue
            # Create interaction column as numeric hash
            orig_split[f'{c1}_{c2}'] = orig_split[c1] * 1000 + orig_split[c2]
            train_split[f'{c1}_{c2}'] = train_split[c1] * 1000 + train_split[c2]
            test_split[f'{c1}_{c2}'] = test_split[c1] * 1000 + test_split[c2]
            
            te_inter = orig_split.groupby(f'{c1}_{c2}')[TARGET].mean()
            te_col = f"TE_{c1}_{c2}"
            train_split[te_col] = train_split[f'{c1}_{c2}'].map(te_inter).fillna(orig_split[TARGET].mean())
            test_split[te_col] = test_split[f'{c1}_{c2}'].map(te_inter).fillna(orig_split[TARGET].mean())
            TE_FEATURES.append(te_col)
            # Drop the interaction column as we only need the TE
            train_split.drop(f'{c1}_{c2}', axis=1, inplace=True)
            test_split.drop(f'{c1}_{c2}', axis=1, inplace=True)
    
    NUMS = [c for c in train_split.columns if c not in CATS + ['id', TARGET, 'y'] and c not in TE_FEATURES]
    FEATURES = NUMS + CATS + TE_FEATURES + ['y']
    
    # XGBoost parameters
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "subsample": trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "seed": 42,
        "device": "cuda",
    }
    
    # Quick 3-fold CV for optimization
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_split)):
        X_train = train_split.iloc[train_idx][FEATURES].copy()
        y_train = train_split.iloc[train_idx][TARGET] - train_split.iloc[train_idx]['y']
        
        X_valid = train_split.iloc[val_idx][FEATURES].copy()
        y_valid = train_split.iloc[val_idx][TARGET] - train_split.iloc[val_idx]['y']
        y_valid_synthetic = train_split.iloc[val_idx]['y'].values
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_valid, label=y_valid)
        
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=5000,
            evals=[(dval, "valid")],
            early_stopping_rounds=100,
            verbose_eval=False
        )
        
        preds = model.predict(dval, iteration_range=(0, model.best_iteration + 1)) + y_valid_synthetic
        fold_rmse = np.sqrt(np.mean((preds - train_split.iloc[val_idx][TARGET].values) ** 2))
        cv_scores.append(fold_rmse)
    
    return np.mean(cv_scores)

# ======================== OPTUNA OPTIMIZATION ========================
print("\n" + "="*60)
print("Starting Optuna Hyperparameter Optimization")
print("="*60)

study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42),
    study_name="road_accident_optimization"
)

study.optimize(objective, n_trials=30, timeout=3600, show_progress_bar=True)

print("\n" + "="*60)
print("Optimization Complete!")
print("="*60)
print(f"Best RMSE: {study.best_value:.6f}")
print("\nBest Parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# ======================== FINAL TRAINING WITH BEST PARAMS ========================
print("\n" + "="*60)
print("Training Final Model with Best Parameters")
print("="*60)

# Reconstruct best trial
best_trial = study.best_trial
train_final = engineer_features(train, best_trial)
test_final = engineer_features(test, best_trial)
orig_final = engineer_features(orig, best_trial)

combine = pd.concat([train_final, test_final, orig_final], axis=0, ignore_index=True)
combine["y"] = clip_target(compute_synthetic_target)(combine).values

# Encode categoricals
CATS = combine.select_dtypes(include=['object']).columns.tolist()
CATS = [c for c in CATS if c not in ['id', TARGET]]

for c in CATS:
    le = LabelEncoder()
    combine[c] = le.fit_transform(combine[c].astype(str))
    combine[c] = combine[c].astype('int32')

train_final = combine.iloc[:len(train)].copy()
test_final = combine.iloc[len(train):len(train)+len(test)].copy()
orig_final = combine.iloc[-len(orig):].copy()

# Target encoding with best params
TE_FEATURES = []
base_features = ['lighting', 'weather', 'speed_limit', 'curvature', 'num_reported_accidents']
if 'speed_bin' in train_final.columns:
    base_features.extend(['speed_bin', 'curvature_bin', 'accidents_bin'])

for c in base_features:
    if c not in orig_final.columns:
        continue
    
    if best_trial.params.get('use_te_mean', True):
        te_mean = orig_final.groupby(c)[TARGET].mean()
        te_col = f"TE_mean_{c}"
        train_final[te_col] = train_final[c].map(te_mean).fillna(orig_final[TARGET].mean())
        test_final[te_col] = test_final[c].map(te_mean).fillna(orig_final[TARGET].mean())
        TE_FEATURES.append(te_col)
    
    if best_trial.params.get('use_te_std', False):
        te_std = orig_final.groupby(c)[TARGET].std()
        te_col_std = f"TE_std_{c}"
        train_final[te_col_std] = train_final[c].map(te_std).fillna(orig_final[TARGET].std())
        test_final[te_col_std] = test_final[c].map(te_std).fillna(orig_final[TARGET].std())
        TE_FEATURES.append(te_col_std)

if best_trial.params.get('use_te_interactions', True):
    interaction_pairs = [('lighting', 'weather')]
    if 'speed_bin' in train_final.columns:
        interaction_pairs.extend([('speed_bin', 'curvature_bin'), ('weather', 'speed_bin')])
    
    for c1, c2 in interaction_pairs:
        if c1 not in orig_final.columns or c2 not in orig_final.columns:
            continue
        orig_final[f'{c1}_{c2}'] = orig_final[c1].astype(str) + '_' + orig_final[c2].astype(str)
        train_final[f'{c1}_{c2}'] = train_final[c1].astype(str) + '_' + train_final[c2].astype(str)
        test_final[f'{c1}_{c2}'] = test_final[c1].astype(str) + '_' + test_final[c2].astype(str)
        
        te_inter = orig_final.groupby(f'{c1}_{c2}')[TARGET].mean()
        te_col = f"TE_{c1}_{c2}"
        train_final[te_col] = train_final[f'{c1}_{c2}'].map(te_inter).fillna(orig_final[TARGET].mean())
        test_final[te_col] = test_final[f'{c1}_{c2}'].map(te_inter).fillna(orig_final[TARGET].mean())
        TE_FEATURES.append(te_col)

NUMS = [c for c in train_final.columns if c not in CATS + ['id', TARGET, 'y'] and c not in TE_FEATURES]
FEATURES = NUMS + CATS + TE_FEATURES + ['y']

print(f"Total features: {len(FEATURES)}")

# Extract best XGBoost params
best_xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "seed": 42,
    "device": "cuda",
}
for key in ['learning_rate', 'max_depth', 'min_child_weight', 'subsample', 
            'colsample_bytree', 'colsample_bylevel', 'reg_alpha', 'reg_lambda', 'gamma']:
    best_xgb_params[key] = best_trial.params[key]

# Full training with 11 folds
FOLDS = 11
oof_preds = np.zeros(len(train_final))
test_preds = np.zeros(len(test_final))
feature_importance = np.zeros(len(FEATURES))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_final)):
    print(f"\nFold {fold+1}/{FOLDS}")
    
    X_train = train_final.iloc[train_idx][FEATURES].copy()
    y_train = train_final.iloc[train_idx][TARGET] - train_final.iloc[train_idx]['y']
    
    X_valid = train_final.iloc[val_idx][FEATURES].copy()
    y_valid = train_final.iloc[val_idx][TARGET] - train_final.iloc[val_idx]['y']
    y_valid_synthetic = train_final.iloc[val_idx]['y'].values
    
    X_test = test_final[FEATURES].copy()
    y_test_synthetic = test_final['y'].values
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test)
    
    model = xgb.train(
        params=best_xgb_params,
        dtrain=dtrain,
        num_boost_round=100_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=300,
        verbose_eval=500
    )
    
    oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1)) + y_valid_synthetic
    test_preds += (model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) + y_test_synthetic) / FOLDS
    
    importance = model.get_score(importance_type='gain')
    for i, feat in enumerate(FEATURES):
        feature_importance[i] += importance.get(f'f{i}', 0) / FOLDS
    
    fold_rmse = np.sqrt(np.mean((oof_preds[val_idx] - train_final.iloc[val_idx][TARGET].values) ** 2))
    print(f"Fold {fold+1} RMSE: {fold_rmse:.6f}")

# ======================== RESULTS ========================
cv_rmse = np.sqrt(np.mean((oof_preds - train_final[TARGET].values) ** 2))
baseline_rmse = np.sqrt(np.mean((train_final['y'].values - train_final[TARGET].values) ** 2))

print(f"\n{'='*60}")
print(f"Final CV RMSE: {cv_rmse:.6f}")
print(f"Baseline RMSE: {baseline_rmse:.6f}")
print(f"Improvement: {baseline_rmse - cv_rmse:.6f}")
print(f"{'='*60}")

# Top features
feat_df = pd.DataFrame({'feature': FEATURES, 'importance': feature_importance})
feat_df = feat_df.sort_values('importance', ascending=False).head(20)
print("\nTop 20 Features:")
print(feat_df.to_string(index=False))

# ======================== SUBMISSION ========================
sub[TARGET] = test_preds
sub.to_csv("submission.csv", index=False)
print(f"\nSubmission saved! Predicted range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
print(sub.head(10))

