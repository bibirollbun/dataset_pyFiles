import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
np.random.seed(42)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))



train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print("Target distribution:")
print(train['BeatsPerMinute'].describe())


def create_features(df):
    df_new = df.copy()
    df_new['rhythm_energy'] = df_new['RhythmScore'] * df_new['Energy']
    df_new['vocal_energy'] = df_new['VocalContent'] * df_new['Energy']
    df_new['mood_rhythm'] = df_new['MoodScore'] * df_new['RhythmScore']
    df_new['duration_min'] = df_new['TrackDurationMs'] / 60000
    df_new['log_duration'] = np.log1p(df_new['TrackDurationMs'])
    df_new['energy_squared'] = df_new['Energy'] ** 2
    df_new['rhythm_squared'] = df_new['RhythmScore'] ** 2
    df_new['acoustic_ratio'] = df_new['AcousticQuality'] / (df_new['InstrumentalScore'] + 0.01)
    df_new['live_mood'] = df_new['LivePerformanceLikelihood'] * df_new['MoodScore']
    df_new['loudness_energy'] = df_new['Energy'] / (np.abs(df_new['AudioLoudness']) + 0.01)
    return df_new

train_fe = create_features(train)
test_fe = create_features(test)



features = [col for col in train_fe.columns if col not in ['id', 'BeatsPerMinute']]
X = train_fe[features]
y = train_fe['BeatsPerMinute']
X_test = test_fe[features]

scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)



kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_scaled))
test_preds = np.zeros(len(X_test_scaled))
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"Training fold {fold + 1}/5")
    
    X_train_fold = X_scaled.iloc[train_idx]
    X_val_fold = X_scaled.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]
    
    # LightGBM
    lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
    lgb_val = lgb.Dataset(X_val_fold, y_val_fold)
    lgb_model = lgb.train(
        {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'verbose': -1, 'random_state': 42},
        lgb_train, valid_sets=[lgb_val], num_boost_round=1000, callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, n_estimators=1000, random_state=42, verbosity=0
    )
    xgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], early_stopping_rounds=100, verbose=False)
    
    # CatBoost
    cb_model = cb.CatBoostRegressor(
        loss_function='RMSE', learning_rate=0.05, depth=6, iterations=1000, random_seed=42, verbose=False
    )
    cb_model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=100, verbose=False)
    
    # Blend predictions
    blend_pred = 0.4 * lgb_model.predict(X_val_fold) + 0.4 * xgb_model.predict(X_val_fold) + 0.2 * cb_model.predict(X_val_fold)
    oof_preds[val_idx] = blend_pred
    
    test_preds += (0.4 * lgb_model.predict(X_test_scaled) + 0.4 * xgb_model.predict(X_test_scaled) + 0.2 * cb_model.predict(X_test_scaled)) / 5
    fold_score = rmse(y_val_fold, blend_pred)
    cv_scores.append(fold_score)
    print(f"Fold {fold + 1} RMSE: {fold_score:.5f}")

cv_score = rmse(y, oof_preds)
print(f"Overall CV RMSE: {cv_score:.5f}, CV Std: {np.std(cv_scores):.5f}")



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y, oof_preds, alpha=0.5, s=1)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual BPM')
plt.ylabel('Predicted BPM')
plt.title(f'OOF Predictions (RMSE: {cv_score:.3f})')

plt.subplot(1, 2, 2)
residuals = y - oof_preds
plt.scatter(oof_preds, residuals, alpha=0.5, s=1)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted BPM')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()



final_preds = np.clip(test_preds, 60, 200)
submission = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': final_preds})

print("Submission stats:")
print(f"Min: {final_preds.min():.2f}, Max: {final_preds.max():.2f}")
print(f"Mean: {final_preds.mean():.2f}, Std: {final_preds.std():.2f}")

submission.to_csv('submission.csv', index=False)
print("Submission saved successfully!")
submission.head()





