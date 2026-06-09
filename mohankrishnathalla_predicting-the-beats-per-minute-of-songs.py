import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

print(train.head())
print(test.head())
print(submission.head())


features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 
            'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore', 
            'TrackDurationMs', 'Energy']
X = train[features]
y = train['BeatsPerMinute']
X_test = test[features]


X['AudioLoudness'] = np.log1p(-X['AudioLoudness'])
X_test['AudioLoudness'] = np.log1p(-X_test['AudioLoudness'])
X['TrackDurationMs'] = np.log1p(X['TrackDurationMs'])
X_test['TrackDurationMs'] = np.log1p(X_test['TrackDurationMs'])


X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)
X_test_scaled = scaler.transform(X_test)


lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_holdout_pred = lr.predict(X_holdout_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_holdout, lr_holdout_pred))
print(f"Linear RMSE: {lr_rmse:.4f}")


rf = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_holdout_pred = rf.predict(X_holdout_scaled)
rf_rmse = np.sqrt(mean_squared_error(y_holdout, rf_holdout_pred))
print(f"RF RMSE: {rf_rmse:.4f}")


xgb_model = xgb.XGBRegressor(n_estimators=130, learning_rate=0.06, max_depth=6, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_holdout_pred = xgb_model.predict(X_holdout_scaled)
xgb_rmse = np.sqrt(mean_squared_error(y_holdout, xgb_holdout_pred))
print(f"XGBoost RMSE: {xgb_rmse:.4f}")


lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, random_state=42, n_jobs=-1)
lgb_model.fit(X_train_scaled, y_train)
lgb_holdout_pred = lgb_model.predict(X_holdout_scaled)
lgb_rmse = np.sqrt(mean_squared_error(y_holdout, lgb_holdout_pred))
print(f"LightGBM RMSE: {lgb_rmse:.4f}")


kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_rmse = float('inf')
best_weights = None
for train_idx, val_idx in kf.split(X_train_scaled):
    X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    lr_fold_pred = lr.predict(X_val_fold)
    rf_fold_pred = rf.predict(X_val_fold)
    xgb_fold_pred = xgb_model.predict(X_val_fold)
    lgb_fold_pred = lgb_model.predict(X_val_fold)
    
    weight_combos = [
        (0.0985, 0.6010, 0.2990, 0.0015), (0.0990, 0.6005, 0.2985, 0.0020),
        (0.0995, 0.6000, 0.2980, 0.0025), (0.1000, 0.5995, 0.2975, 0.0030),
        (0.0985, 0.6005, 0.2985, 0.0025), (0.0990, 0.6000, 0.2980, 0.0030)
    ]  # (LR, RF, XGB, LGBM) summing to 1
    for w_lr, w_rf, w_xgb, w_lgb in weight_combos:
        ensemble_fold_pred = (w_lr * lr_fold_pred + w_rf * rf_fold_pred + 
                             w_xgb * xgb_fold_pred + w_lgb * lgb_fold_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, ensemble_fold_pred))
        if fold_rmse < best_rmse:
            best_rmse = fold_rmse
            best_weights = (w_lr, w_rf, w_xgb, w_lgb)

print(f"Best Weights (5-fold CV-averaged): {best_weights}, Best CV RMSE: {best_rmse:.4f}")


lr_test_pred = lr.predict(X_test_scaled)
rf_test_pred = rf.predict(X_test_scaled)
xgb_test_pred = xgb_model.predict(X_test_scaled)
lgb_test_pred = lgb_model.predict(X_test_scaled)
test_pred = (best_weights[0] * lr_test_pred + best_weights[1] * rf_test_pred + 
             best_weights[2] * xgb_test_pred + best_weights[3] * lgb_test_pred)
submission['BeatsPerMinute'] = test_pred
submission.to_csv('submission_ensemble_adjusted.csv', index=False)
print("Submission file created: submission_ensemble_adjusted.csv")

