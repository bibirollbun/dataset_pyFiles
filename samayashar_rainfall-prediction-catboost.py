# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import optuna


# Load the data
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col='id')
sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


# Extract target variable
y = train['rainfall']
X = train.drop(columns=['rainfall'])


# Feature Engineering
X['temp_diff'] = X['maxtemp'] - X['mintemp']
X['humidity_cloud_interaction'] = X['humidity'] * X['cloud']
X['temp_humidity'] = X['temparature'] * X['humidity']
X['wind_speed_humidity'] = X['windspeed'] * X['humidity']
X['sunshine_cloud_ratio'] = X['sunshine'] / (X['cloud'] + 1)


test['temp_diff'] = test['maxtemp'] - test['mintemp']
test['humidity_cloud_interaction'] = test['humidity'] * test['cloud']
test['temp_humidity'] = test['temparature'] * test['humidity']
test['wind_speed_humidity'] = test['windspeed'] * test['humidity']
test['sunshine_cloud_ratio'] = test['sunshine'] / (test['cloud'] + 1)


# Encode categorical variables
categorical_features = ['winddirection']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_features] = encoder.fit_transform(X[categorical_features])
test[categorical_features] = encoder.transform(test[categorical_features])


# Scale numerical features
scaler = StandardScaler()
numerical_features = ['temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed', 'temp_diff', 'humidity_cloud_interaction', 'temp_humidity', 'wind_speed_humidity', 'sunshine_cloud_ratio']
X[numerical_features] = scaler.fit_transform(X[numerical_features])
test[numerical_features] = scaler.transform(test[numerical_features])


# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Hyperparameter Optimization
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'depth': trial.suggest_int('depth', 4, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_state': 42,
        'eval_metric': 'AUC',
        'verbose': 0
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100, verbose=0)
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, y_pred_proba)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
best_params = study.best_params


# Train the optimized CatBoost model
model_cat = CatBoostClassifier(**best_params)
model_cat.fit(X_train, y_train)
y_pred_proba = model_cat.predict_proba(X_valid)[:, 1]
print(f"Optimized CatBoost ROC AUC: {roc_auc_score(y_valid, y_pred_proba):.4f}")


# Make predictions on test data
test_preds = model_cat.predict_proba(test)[:, 1]


# Create submission file
submission = pd.DataFrame({'id': test.index, 'rain': test_preds})
submission.to_csv('submission.csv', index=False)




