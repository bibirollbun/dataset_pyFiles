import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

# --- 0. Setup ---
warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. Load Data with Pandas ---
print("--- Loading Data with Pandas ---")
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
except FileNotFoundError:
    print("Could not find Kaggle dataset files. Exiting.")
    exit()

# Prepare dataframes
test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)
y = train_df['accident_risk']

# --- 2. Feature Engineering with Pandas ---
print("--- Starting Feature Engineering with Pandas ---")
full_df = pd.concat([train_df.drop('accident_risk', axis=1), test_df], axis=0).reset_index(drop=True)

# Interaction and Ratio Features
full_df['weather_lighting'] = full_df['weather'] + '_' + full_df['lighting']
full_df['road_time'] = full_df['road_type'] + '_' + full_df['time_of_day']
epsilon = 1e-6
full_df['speed_per_lane'] = full_df['speed_limit'] / (full_df['num_lanes'] + epsilon)
full_df['curvature_per_speed'] = full_df['curvature'] / (full_df['speed_limit'] + epsilon)
full_df['accidents_per_lane'] = full_df['num_reported_accidents'] / (full_df['num_lanes'] + epsilon)
full_df['curvature_sq'] = full_df['curvature']**2
full_df['speed_limit_sq'] = full_df['speed_limit']**2

# Encoding
bool_cols = full_df.select_dtypes(include='bool').columns
for col in bool_cols:
    full_df[col] = full_df[col].astype(int)
categorical_cols = full_df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    full_df[col] = pd.Categorical(full_df[col]).codes

X = full_df[:len(train_df)]
X_test = full_df[len(train_df):]
print(f"Feature engineering complete. Total features: {X.shape[1]}")

# --- 3. K-Fold Modeling and Ensembling ---
print("\n--- Starting 4-Model K-Fold Training ---")
NFOLDS = 10
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Placeholders for OOF and test predictions for all 4 models
oof_preds_lgb = np.zeros(X.shape[0])
sub_preds_lgb = np.zeros(X_test.shape[0])
oof_preds_xgb = np.zeros(X.shape[0])
sub_preds_xgb = np.zeros(X_test.shape[0])
oof_preds_nn = np.zeros(X.shape[0])
sub_preds_nn = np.zeros(X_test.shape[0])
oof_preds_rf = np.zeros(X.shape[0])
sub_preds_rf = np.zeros(X_test.shape[0])

# --- Model Parameters ---
lgb_params = {'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2500, 'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'num_leaves': 40, 'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt', 'device': 'gpu'}
xgb_params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 2500, 'learning_rate': 0.01, 'max_depth': 8, 'subsample': 0.7, 'colsample_bytree': 0.7, 'random_state': 42, 'n_jobs': -1, 'tree_method': 'gpu_hist', 'early_stopping_rounds': 150}
rf_params = {'n_estimators': 200, 'max_depth': 12, 'min_samples_leaf': 10, 'random_state': 42, 'n_jobs': -1}

# --- K-Fold Training Loop ---
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    print(f"--- Fold {n_fold+1}/{NFOLDS} ---")
    
    # --- Data Splitting for this Fold ---
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_valid_fold, y_valid_fold = X.iloc[valid_idx], y.iloc[valid_idx]

    # --- 1. LightGBM (GPU) ---
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_valid_fold, y_valid_fold)], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof_preds_lgb[valid_idx] = lgb_model.predict(X_valid_fold)
    sub_preds_lgb += lgb_model.predict(X_test) / folds.n_splits

    # --- 2. XGBoost (GPU) ---
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_valid_fold, y_valid_fold)], verbose=False)
    oof_preds_xgb[valid_idx] = xgb_model.predict(X_valid_fold)
    sub_preds_xgb += xgb_model.predict(X_test) / folds.n_splits
    
    # --- 3. Random Forest (CPU) ---
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train_fold, y_train_fold)
    oof_preds_rf[valid_idx] = rf_model.predict(X_valid_fold)
    sub_preds_rf += rf_model.predict(X_test) / folds.n_splits

    # --- 4. Neural Network (GPU) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_valid_scaled = scaler.transform(X_valid_fold)
    X_test_scaled = scaler.transform(X_test)

    def create_nn_model(input_shape):
        model = Sequential([Dense(256, activation='relu', input_shape=[input_shape]), BatchNormalization(), Dropout(0.3), Dense(128, activation='relu'), BatchNormalization(), Dropout(0.3), Dense(64, activation='relu'), BatchNormalization(), Dropout(0.3), Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    nn_model = create_nn_model(X_train_scaled.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    nn_model.fit(X_train_scaled, y_train_fold, validation_data=(X_valid_scaled, y_valid_fold), epochs=200, batch_size=512, callbacks=[early_stopping], verbose=0)
    
    oof_preds_nn[valid_idx] = nn_model.predict(X_valid_scaled).flatten()
    sub_preds_nn += nn_model.predict(X_test_scaled).flatten() / folds.n_splits

    # Report fold scores
    lgb_rmse = np.sqrt(mean_squared_error(y_valid_fold, oof_preds_lgb[valid_idx]))
    xgb_rmse = np.sqrt(mean_squared_error(y_valid_fold, oof_preds_xgb[valid_idx]))
    rf_rmse = np.sqrt(mean_squared_error(y_valid_fold, oof_preds_rf[valid_idx]))
    nn_rmse = np.sqrt(mean_squared_error(y_valid_fold, oof_preds_nn[valid_idx]))
    print(f"LGB: {lgb_rmse:.6f} | XGB: {xgb_rmse:.6f} | RF: {rf_rmse:.6f} | NN: {nn_rmse:.6f}")

# --- 4. Final Ensembling ---
print("\n--- Averaging Predictions of 4 Models ---")
final_predictions = (sub_preds_xgb + sub_preds_lgb + sub_preds_nn + sub_preds_rf) / 4

# --- 5. Create Submission File ---
print("\n--- Creating Final Submission File ---")
submission_df = pd.DataFrame({'id': test_ids, 'prediction': final_predictions})
submission_df.to_csv('submission_4_model_ensemble.csv', index=False)
print("\n✅ Submission file 'submission_4_model_ensemble.csv' created successfully!")

