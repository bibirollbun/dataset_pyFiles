import shutil
shutil.rmtree("/kaggle/working/", ignore_errors=True) 

!pip install optuna-integration[sklearn]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import gc

DEVICE = 'CPU'   # can change this manually to GPU


print("Loading data...")
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
ss = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

X = train.drop(columns=['id', 'accident_risk'])
y = train['accident_risk']
test_X = test.drop(columns=['id'])

categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
print("Categorical Features:", categorical_features)

train_pool = Pool(X, y, cat_features=categorical_features)
test_pool = Pool(test_X, cat_features=categorical_features)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 650, 750),
        'depth': trial.suggest_int('depth', 7, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.05, log=True),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 6, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.4, 0.7),
        'random_strength': trial.suggest_int('random_strength', 2, 5),
        'grow_policy': trial.suggest_categorical('grow_policy', ['Depthwise']),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 40, 55),
        'task_type': DEVICE,
        'devices': '0',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': 0
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_pool_cv = Pool(X_train, y_train, cat_features=categorical_features)
        val_pool_cv = Pool(X_val, y_val, cat_features=categorical_features)

        model = CatBoostRegressor(**params)
        model.fit(train_pool_cv, eval_set=val_pool_cv, early_stopping_rounds=15, verbose=0)

        preds = model.predict(val_pool_cv)
        rmse = mean_squared_error(y_val, preds) ** 0.5
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)



# Run fast optuna
print("Starting Optuna optimization...")

study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    study_name='fast_catboost_opt'
)

study.optimize(objective, n_trials=25, timeout=40000, show_progress_bar=True)

print("\nOptuna Results:")
print(f"Best RMSE: {study.best_value:.4f}")
print("Best Parameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")


optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()


print("\nTraining final model with best Optuna parameters...")

best_params = study.best_params.copy()
best_params.update({
    'task_type': DEVICE,
    'devices': '0',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 100
})

final_model = CatBoostRegressor(**best_params)
final_model.fit(train_pool)

# Evaluate
y_pred = final_model.predict(X)
rmse = mean_squared_error(y, y_pred) ** 0.5
r2 = r2_score(y, y_pred)

# feature importances
importances = final_model.get_feature_importance()
indices = np.argsort(importances)[-20:]

plt.figure(figsize=(12, 8))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.xlabel("Importance")
plt.title("Top 20 CatBoost Feature Importances")
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# Plots
plt.figure(figsize=(8, 8))
plt.scatter(y, y_pred, alpha=0.3, s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Accident Risk")
plt.ylabel("Predicted Accident Risk")
plt.title("Predicted vs Actual")
plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=150)
plt.show()

# Submission
print("\nGenerating submission file...")
submission = ss.copy()
submission['accident_risk'] = final_model.predict(test_pool)
submission.to_csv('submission.csv', index=False)
print("submission.csv saved successfully!")

print("\n===============================")
print("SUMMARY")
print("===============================")
print(f"Best RMSE: {study.best_value:.4f}")
print(f"Final Train-Val RMSE: {rmse:.4f}")
print(f"Final Train-Val R²: {r2:.4f}")
print("===============================")

