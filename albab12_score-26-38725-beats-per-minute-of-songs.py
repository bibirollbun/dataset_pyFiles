import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from scipy.stats import skew
from scipy.optimize import minimize

# Load the competition data
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

# Load and concatenate original data
original = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')
train = pd.concat([train, original], ignore_index=True)

# Feature engineering
def create_features(df):
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Interaction features
    df['Rhythm_Audio_Interaction'] = df['RhythmScore'] * df['AudioLoudness']
    df['Vocal_Acoustic_Ratio'] = df['VocalContent'] / (df['AcousticQuality'] + 1e-6)
    df['Energy_Mood_Product'] = df['Energy'] * df['MoodScore']
    df['Instrumental_Live_Interaction'] = df['InstrumentalScore'] * df['LivePerformanceLikelihood']
    df['Mood_Acoustic_Product'] = df['MoodScore'] * df['AcousticQuality']
    df['Vocal_Energy_Product'] = df['VocalContent'] * df['Energy']
    
    # Polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = poly.fit_transform(df[['RhythmScore', 'AudioLoudness', 'Energy', 'VocalContent', 'AcousticQuality']])
    poly_cols = [f'poly_{i}' for i in range(poly_features.shape[1])]
    df[poly_cols] = poly_features
    
    # Log transformation for skewed features
    skew_cols = ['TrackDurationMs', 'AudioLoudness', 'VocalContent', 'Energy', 'RhythmScore', 
                 'MoodScore', 'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood']
    for col in skew_cols:
        if col in df.columns and skew(df[col].dropna()) > 0.5:
            if df[col].min() < 0:
                shift = abs(df[col].min()) + 1
                df[f'log_{col}'] = np.log1p(df[col] + shift)
            else:
                df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
    
    # Binning features
    df['Duration_Bin'] = pd.qcut(df['TrackDurationMs'], q=10, labels=False, duplicates='drop')
    df['Energy_Bin'] = pd.qcut(df['Energy'], q=5, labels=False, duplicates='drop')
    df['Loudness_Bin'] = pd.qcut(df['AudioLoudness'], q=10, labels=False, duplicates='drop')
    df['Vocal_Bin'] = pd.qcut(df['VocalContent'], q=5, labels=False, duplicates='drop')
    
    return df

# Apply feature engineering
train = create_features(train)
test = create_features(test)

# Features list, excluding target and constant features
features = [col for col in train.columns if col not in ['id', 'BeatsPerMinute'] and train[col].nunique() > 1]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(train[features])
X_test = scaler.transform(test[features])
y = train['BeatsPerMinute']

# Model parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.02,
    'num_leaves': 50,
    'max_depth': 8,
    'min_data_in_leaf': 40,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'seed': 42
}

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.02,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'seed': 42,
    'n_jobs': -1
}

cat_params = {
    'loss_function': 'RMSE',
    'learning_rate': 0.03,
    'depth': 7,
    'min_data_in_leaf': 40,
    'l2_leaf_reg': 3.0,
    'iterations': 3000,
    'random_seed': 42,
    'verbose': 0
}

# K-Fold for training
n_splits = 15
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(test))
xgb_preds = np.zeros(len(test))
cat_preds = np.zeros(len(test))

lgb_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))
cat_oof = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # LightGBM
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    lgb_oof[val_idx] = lgb_model.predict(X_val)
    lgb_preds += lgb_model.predict(X_test) / n_splits
    
    # XGBoost
    xgb_train = xgb.DMatrix(X_train, y_train)
    xgb_val = xgb.DMatrix(X_val, y_val)
    xgb_model = xgb.train(
        xgb_params,
        xgb_train,
        num_boost_round=3000,
        evals=[(xgb_val, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    xgb_oof[val_idx] = xgb_model.predict(xgb_val)
    xgb_preds += xgb_model.predict(xgb.DMatrix(X_test)) / n_splits
    
    # CatBoost
    cat_train = cb.Pool(X_train, y_train)
    cat_val = cb.Pool(X_val, y_val)
    cat_model = cb.CatBoostRegressor(**cat_params)
    cat_model.fit(cat_train, eval_set=cat_val, early_stopping_rounds=100, verbose=False)
    cat_oof[val_idx] = cat_model.predict(X_val)
    cat_preds += cat_model.predict(X_test) / n_splits
    
    # Print fold RMSE for blend
    blend_val_pred = 0.5 * lgb_oof[val_idx] + 0.3 * xgb_oof[val_idx] + 0.2 * cat_oof[val_idx]
    rmse = np.sqrt(mean_squared_error(y_val, blend_val_pred))
    print(f'Fold {fold+1} RMSE: {rmse}')

# Optimize blending weights using OOF predictions
def rmse_func(weights):
    blend_oof = weights[0] * lgb_oof + weights[1] * xgb_oof + weights[2] * cat_oof
    return np.sqrt(mean_squared_error(y, blend_oof))

cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bnds = [(0, 1)] * 3
init_guess = [0.5, 0.3, 0.2]

opt_res = minimize(rmse_func, init_guess, bounds=bnds, constraints=cons, method='SLSQP')
print(f'\nOptimized Weights: {opt_res.x}')
print(f'Optimized RMSE: {opt_res.fun}')

# Final blended predictions with optimized weights
final_preds = opt_res.x[0] * lgb_preds + opt_res.x[1] * xgb_preds + opt_res.x[2] * cat_preds

# Clip predictions to reasonable BPM range (e.g., 40-200 BPM)
final_preds = np.clip(final_preds, 40, 200)

# Create submission
submission['BeatsPerMinute'] = final_preds
submission.to_csv('submission.csv', index=False)
print('Optimized submission file created!')

