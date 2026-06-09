!ls /kaggle/input/playground-series-s5e9


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


import warnings

# Suppress simple warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train.head()


train.columns


sns.set(style="whitegrid")
train.info()


train.replace([np.inf, -np.inf], np.nan, inplace=True)



#vis
plt.figure(figsize=(8,5))
sns.histplot(train['BeatsPerMinute'],bins=1000,kde=True,color='skyblue')
plt.title("BPM distribution")
plt.xlabel("BPM")
plt.ylabel("Frequency")
plt.show()


# correlation matrix for features
plt.figure(figsize=(10,8))
corr = train.drop('id', axis=1).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# config for k - fold crossvalidation
RANDOM_SEED = 42
N_FOLDS = 5


# creating new features using feature engineering
def engineer_features(df):
    df_new = df.copy()
    
    # 1. Energy and Loudness
    df_new['Energy_per_Loudness'] = df_new['Energy'] / (df_new['AudioLoudness'] + 1e-6)
    
    # 2. Quality and Energy interaction
    df_new['Quality_x_Energy'] = df_new['AcousticQuality'] * df_new['Energy']
    
    # 3. Total "Content" score
    df_new['TotalContentScore'] = df_new['VocalContent'] + df_new['InstrumentalScore']
    
    # 4. Mood and Rhythm
    df_new['Mood_x_Rhythm'] = df_new['MoodScore'] * df_new['RhythmScore']
    
    return df_new


# for calculating rmse
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


'''Model 1: LGBM'''
def train_lightgbm(X_train, y_train, X_val, y_val):
    """Trains a LightGBM model with early stopping."""
    model = lgb.LGBMRegressor(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        n_estimators=1000, # High number, early stopping will find the best
        learning_rate=0.05,
        num_leaves=31
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model



'''Model 2: XgBoost'''
def train_xgboost(X_train, y_train, X_val, y_val):
    """Trains an XGBoost model with early stopping."""
    model = xgb.XGBRegressor(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=100,
              verbose=False)
    return model


'''Model 3: catBoost'''
def train_catboost(X_train, y_train, X_val, y_val):
    """Trains a CatBoost model with early stopping."""
    model = cb.CatBoostRegressor(
        random_state=RANDOM_SEED,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        loss_function='RMSE'
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=100,
              verbose=False)
    return model


train = engineer_features(train)
test= engineer_features(test)


train.head()


'''Organising Features'''
TARGET_COL = 'BeatsPerMinute'
ORIGINAL_FEATURES = [
    'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
    'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
    'TrackDurationMs', 'Energy'
]
ENGINEERED_FEATURES = [
    'Energy_per_Loudness', 'Quality_x_Energy', 'TotalContentScore', 'Mood_x_Rhythm'
]
ALL_FEATURES = ORIGINAL_FEATURES + ENGINEERED_FEATURES


X_train = train[ALL_FEATURES]
y_train = train[TARGET_COL]
X_test = test[ALL_FEATURES]
X_test = X_test[X_train.columns]
print(f"Training with {len(ALL_FEATURES)} features.")


'''K-fold'''
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)


'''oof - out-of-fold'''
models = []
oof_predictions = np.zeros(len(X_train)) 
test_predictions = np.zeros(len(X_test)) 


for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"\nTraining fold {fold_idx + 1}/{N_FOLDS}")
    
    # Split data for this fold
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train models
    print("Training LightGBM...")
    lgb_model = train_lightgbm(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
    
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
    
    print("Training CatBoost...")
    cb_model = train_catboost(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
    
    # Random Forest as an additional diverse model
    print("Training Random Forest...")
    '''Model 4: Random Forest'''
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    rf_model.fit(X_fold_train, y_fold_train)
    
    # predictions on validation fold 
    lgb_preds = lgb_model.predict(X_fold_val)
    xgb_preds = xgb_model.predict(X_fold_val)
    cb_preds = cb_model.predict(X_fold_val)
    rf_preds = rf_model.predict(X_fold_val)
    
    # Creating a weighted average of predictions
    blend_preds = 0.35 * lgb_preds + 0.35 * xgb_preds + 0.2 * cb_preds + 0.1 * rf_preds
    
    # Store out-of-fold predictions
    oof_predictions[val_idx] = blend_preds
    
    # Make predictions on test set 
    lgb_test_preds = lgb_model.predict(X_test)
    xgb_test_preds = xgb_model.predict(X_test)
    cb_test_preds = cb_model.predict(X_test)
    rf_test_preds = rf_model.predict(X_test)
    
    # Average test predictions from this fold
    fold_test_preds = 0.35 * lgb_test_preds + 0.35 * xgb_test_preds + 0.2 * cb_test_preds + 0.1 * rf_test_preds
    
    # Add this fold's test predictions (we will average them at the end)
    test_predictions += fold_test_preds / N_FOLDS
    
    # Calculate and display fold metrics 
    lgb_rmse = rmse(y_fold_val, lgb_preds)
    xgb_rmse = rmse(y_fold_val, xgb_preds)
    cb_rmse = rmse(y_fold_val, cb_preds)
    rf_rmse = rmse(y_fold_val, rf_preds)
    blend_rmse = rmse(y_fold_val, blend_preds)
    
    print(f"--- Fold {fold_idx + 1} Results ---")
    print(f"LightGBM RMSE: {lgb_rmse:.5f}")
    print(f"XGBoost RMSE: {xgb_rmse:.5f}")
    print(f"CatBoost RMSE: {cb_rmse:.5f}")
    print(f"Random Forest RMSE: {rf_rmse:.5f}")
    print(f"Blended RMSE: {blend_rmse:.5f}")
    
    # Store models for this fold 
    models.append({
        'fold': fold_idx,
        'lgb_model': lgb_model,
        'xgb_model': xgb_model,
        'cb_model': cb_model,
        'rf_model': rf_model
    })




