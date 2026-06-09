import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder


def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


for df in [train, test]:
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Duration_per_kg'] = df['Duration'] / df['Weight']
    df['Effort_Index'] = (df['Heart_Rate'] * df['Duration']) / df['Weight']
    df['Temp_Stress_Index'] = df['Body_Temp'] * df['Heart_Rate']
    df['Adjusted_Intensity'] = df['Heart_Rate'] / df['BMI']
    df['Duration_squared'] = df['Duration'] ** 2


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])


X = train.drop(columns=['id', 'Calories'])
y = train['Calories']
X_test = test.drop(columns='id')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return rmsle(y_val, preds)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)



best_params = study.best_trial.params
print("Best params:", best_params)


best_model = lgb.LGBMRegressor(**best_params)
best_model.fit(X, y)
preds = np.maximum(0, best_model.predict(X_test))


submission = pd.DataFrame({
    'id': test['id'],
    'Calories': preds
})
submission.to_csv('submission.csv', index=False)
print("submission.csv saved")




