import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


train=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train.head(3)


train.info()


train.isna().sum()


train.describe().T


test.head(3)


test.info()


test.isna().sum()


test.describe().T


X=train.drop(columns=['id','BeatsPerMinute'])
y=train['BeatsPerMinute']
test_id=test['id']
test=test.drop(columns='id',axis=1)


from sklearn.model_selection import train_test_split,StratifiedKFold,KFold
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import optuna


def catboost_objective(trial):
    params = {
        'objective': 'RMSE',
        'iterations': trial.suggest_int('iterations', 500, 2500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 5, 12),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 10.0),
        'random_strength': trial.suggest_loguniform('random_strength', 1e-8, 10.0),
        'bootstrap_type': 'Bernoulli',
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'random_state': 42,
        'verbose': 0,
        'early_stopping_rounds': 50
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model = CatBoostRegressor(**params)
        
        model.fit(X_train, y_train,
                  eval_set=(X_val, y_val), # Pass the validation set
                  early_stopping_rounds=params['early_stopping_rounds'])
        
        val_preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, val_preds, squared=False)
        rmse_scores.append(rmse)
        
    return np.mean(rmse_scores)


print("Running Optuna study...")
study = optuna.create_study(direction='minimize')
study.optimize(catboost_objective, n_trials=50, show_progress_bar=True)

print("Best hyperparameters found by Optuna:")
best_params = study.best_params
print(best_params)

# Get the best trial RMSE
print(f"Best RMSE from the study: {study.best_value:.4f}")



final_model = CatBoostRegressor(**best_params, random_state=42)
final_model.fit(X, y)


test_predictions = final_model.predict(test)


submission = pd.DataFrame({
    "id": test_id,
    "BeatsPerMinute": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")

