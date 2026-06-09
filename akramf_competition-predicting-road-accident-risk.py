import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
import optuna
import optuna.visualization as vis


df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv", index_col=0)
df_train


notNumberFeatures = df_train.select_dtypes(include=['object', 'bool']).columns.to_list()


print(notNumberFeatures)


encoders = {}
for feature in notNumberFeatures:
    le = LabelEncoder()
    df_train[feature] = le.fit_transform(df_train[feature])
    encoders[feature] = le


X, y = df_train.drop(columns=['accident_risk']), df_train['accident_risk']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


def objective(trial):
    param = {
        # --- Struktur dasar pohon ---
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),

        # --- Regularisasi ---
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),   # reg_lambda
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),     # reg_alpha
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0),

        # --- Booster ---
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),

        # --- Pembelajaran ---
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'tree_method': trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist']),
        'sampling_method': trial.suggest_categorical('sampling_method', ['uniform', 'gradient_based']),
        'objective': 'reg:squarederror',

        # --- DART booster spesifik ---
        'sample_type': trial.suggest_categorical('sample_type', ['uniform', 'weighted']),
        'normalize_type': trial.suggest_categorical('normalize_type', ['tree', 'forest']),
        'rate_drop': trial.suggest_float('rate_drop', 0.0, 0.5),
        'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.5),

        # --- Randomness ---
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': -1,
    }
    
    model = xgb.XGBRegressor(**param)
                              
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse


NUMBER_TRIALS = 1000

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(study_name="predict_road_accident_risk", direction='minimize') 
study.optimize(objective, n_trials=NUMBER_TRIALS, show_progress_bar=True, n_jobs=-1)


# Retrieve the best parameter values
best_params = study.best_params
print(f"\nBest parameters: {best_params}")


vis.plot_param_importances(study)


vis.plot_optimization_history(study)


best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv", index_col=0)
df_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv", index_col=0)

for feature in notNumberFeatures:
    df_test[feature] = encoders[feature].transform(df_test[feature])


df_test.shape[0], df_submission.shape[0]


y_pred = best_model.predict(df_test)

rmse = np.sqrt(mean_squared_error(df_submission, y_pred))

print(f"RMSE : {rmse}")


df_submission['accident_risk'] = y_pred
df_submission.to_csv("submission.csv", index=True)
print("Submission dataset saved!")

