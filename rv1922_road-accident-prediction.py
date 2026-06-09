import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from category_encoders import TargetEncoder
import optuna


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


orig_list = []
for k in [2, 10, 100]:
    df = pd.read_csv(f"/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_{k}k.csv")
    orig_list.append(df)

origi = pd.concat(orig_list, axis=0)

train_cols_without_id = [c for c in train.columns if c != 'id']


origi = origi[train_cols_without_id]
train = pd.concat([train, origi], axis=0).reset_index(drop=True)


if train.duplicated().any():
    train = train.drop_duplicates()


cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
target_col = 'accident_risk'


for c in bool_cols:
    if c in train.columns or test.columns:
        train[c] = train[c].astype(int)
        test[c] = test[c].astype(int)


def feature_engineering(df):

    df = df.copy()

    # Basic interactions
    df['speed_x_curvature'] = df['speed_limit'] * df['curvature']
    df['curv_speed'] = df['curvature'] * df['speed_limit']  
    df['curvature_squared'] = df['curvature'] ** 2
    df['speed_sq'] = df['speed_limit'] ** 2

    # Nonlinear transforms & logs (safe)
    df['accidents_log'] = np.log1p(df['num_reported_accidents'])
    df['curv_log'] = np.log1p(df['curvature'])
    df['speed_log'] = np.log1p(df['speed_limit'])
    df['inv_speed'] = 1.0 / (df['speed_limit'] + 1.0)

    # Ratios / density per lane
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['curv_per_lane'] = df['curvature'] / (df['num_lanes'] + 1)
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 1)

    # Combined risk indices
    df['danger_score'] = (df['speed_limit'] / 100.0) * (df['curvature'] ** 2)
    df['risk_density'] = df['curv_speed'] / (df['num_lanes'] + 1.0)
    df['accident_density'] = df['accidents_per_lane'] * df['speed_x_curvature']

    # Polynomial / smoother mixes
    # Use np.where to protect from negative inside sqrt though curvature and speed_limit are non-negative in domain
    df['poly_mix1'] = np.sqrt(np.maximum(df['curvature'] * df['speed_limit'], 0))
    df['poly_mix2'] = (df['num_reported_accidents'] ** 0.3) * df['speed_limit']

    # Statistical combos
    df['risk_index'] = (df['curv_speed'] * df['accidents_per_lane']) / (df['speed_limit'] + 1.0)
    df['stability_score'] = (df['num_lanes'] / (1.0 + df['curvature'])) * df['speed_limit']

    # Binary derived flags
    df['tight_lane'] = (df['num_lanes'] <= 2).astype(int)
    df['sharp_curve'] = (df['curvature'] > 0.6).astype(int)
    df['high_speed_zone'] = (df['speed_limit'] > 80).astype(int)
    df['critical_zone'] = ((df['sharp_curve'] == 1) & (df['high_speed_zone'] == 1)).astype(int)

    # Clean up infinite / extremely large values (if any)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


train = feature_engineering(train)
test = feature_engineering(test)


train.head()


train.info()


y = train['accident_risk']
X = train.drop(columns=['accident_risk'])


X = train.drop(columns=['id', 'accident_risk'])
y = train['accident_risk']
X_test = test.drop(columns=['id'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial, X_train_raw, y_train, cat_cols):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'device': 'gpu',
        'verbosity': -1,
        'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 5e-2, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
    }

    # Single validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=42
    )

    # Target encoding
    te = TargetEncoder(cols=cat_cols, smoothing=10.0)
    X_tr[cat_cols] = te.fit_transform(X_tr[cat_cols], y_tr)
    X_val[cat_cols] = te.transform(X_val[cat_cols])

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)]
    )

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Trial {trial.number} | CV RMSE: {rmse:.6f}")
    return rmse


#study = optuna.create_study(direction='minimize')
#study.optimize(lambda trial: objective(trial, X, y, cat_cols), n_trials=50)


#print("Best RMSE:", study.best_value)
#print("Best hyperparameters:", study.best_params)


best_params = {
    'n_estimators': 3513,
    'max_depth': 9,
    'learning_rate': 0.005929306312334578,
    'subsample': 0.9947438929288691,
    'colsample_bytree': 0.9766989692517208,
    'min_child_samples': 39,
    'reg_alpha': 0.9824186800478342,
    'reg_lambda': 2.4093999752833892e-05,
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'device': 'gpu',
    'verbosity': -1
}


kf = KFold(n_splits=10, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
rmse_list = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    te = TargetEncoder(cols=cat_cols, smoothing=10.0)
    X_tr[cat_cols] = te.fit_transform(X_tr[cat_cols], y_tr).astype(float)
    X_val[cat_cols] = te.transform(X_val[cat_cols]).astype(float)
    
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)]
    )

    y_pred_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_list.append(rmse)
    
    # Test Prediction
    X_test_encoded = X_test.copy()
    X_test_encoded[cat_cols] = te.transform(X_test_encoded[cat_cols]).astype(float)
    test_preds += model.predict(X_test_encoded) / kf.n_splits
    
    print(f"Fold {fold+1} RMSE: {rmse:.6f}")

print(f"\nMean CV RMSE: {np.mean(rmse_list):.6f}")


lgb.plot_importance(model, max_num_features=30, importance_type='gain', figsize=(10, 8))
plt.title("Top 30 Feature Importances (LightGBM)")
plt.show()


submission['accident_risk'] = test_preds
submission.to_csv('submission.csv', index=False)
submission.head()

