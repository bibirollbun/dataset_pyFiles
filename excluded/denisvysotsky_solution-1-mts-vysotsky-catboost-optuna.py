import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split, StratifiedKFold
import ast

import shap
import optuna as optuna
from sklearn import metrics

import optuna

from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances

from catboost.utils import get_gpu_device_count
from sklearn.metrics import roc_auc_score

import joblib


train = pd.read_csv("/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv")
test = pd.read_csv("/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv")


train.head()


train.info()


test.info()


is_gpu_available = get_gpu_device_count()
device = 'GPU' if is_gpu_available else 'CPU'

device


train['HasCrCard'] = train['HasCrCard'].astype('int8')
train['IsActiveMember'] = train['IsActiveMember'].astype('int8')


num_cols = train.select_dtypes(include=['number']).columns.tolist()
num_cols.remove('id')
num_cols.remove('CustomerId')


train[num_cols].hist(bins=25, figsize=(15,10))
plt.tight_layout()
plt.show()


nan_counts = train.isna().sum()
print(nan_counts)


train.head()


cat_features = ['Surname',
                'Geography',
                'Gender', 
                'HasCrCard', 
                'IsActiveMember'
                ]

target = ['Exited']

features2drop = ['id', 
                 'CustomerId'
                 
                 ]

# Отбираем итоговый набор признаков для использования моделью
filtered_features = [i for i in train.columns if (i not in target and i not in features2drop)]
num_features = [i for i in filtered_features if i not in cat_features]

print("cat_features", cat_features)
print("num_features", len(num_features))
print("targets", target)


X = train[filtered_features].drop(target, axis=1, errors="ignore")
y = train["Exited"]


y.value_counts(normalize = True)


def objective(trial):
    
    #bootstrap_type = trial.suggest_categorical("bootstrap_type", [
        #"Bayesian", 
        #"Bernoulli", 
    #    "MVS"])

    params = {
        "iterations": 2000,
        "depth": 6,
        "learning_rate": trial.suggest_float("learning_rate", 0.06, 0.08, log=False),
        "l2_leaf_reg": 18,
        'max_bin': trial.suggest_int('max_bin', 200, 300),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 200),        
        "bootstrap_type": 'MVS',
        "random_strength": 0.15,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "verbose": 100,
        "random_seed": 42,
        "task_type": "GPU"
    }

    #if bootstrap_type == "Bayesian":
    #    params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)

    #if bootstrap_type == "Bernoulli":
    #    params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)

    # Стратифицированная кросс-валидация
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, valid_idx in cv.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=100)

        preds = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, preds)
        scores.append(score)

    return np.mean(scores)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Лучшие параметры
print("Best parameters:", study.best_params)
print("Best ROC AUC:", study.best_value)


best_params = study.best_params
best_params


fig = plot_optimization_history(study)
fig.show()


fig = plot_param_importances(study)
fig.show()


param_names = list(study.best_params.keys())

for param_x, param_y in combinations(param_names, 2):
    fig = plot_contour(study, params=[param_x, param_y])
    fig.update_layout(title=f'Contour plot: {param_x} vs {param_y}')
    fig.show()


joblib.dump(study, 'optuna_study2706_6_less.pkl')


final_model = CatBoostClassifier(
    **best_params,
    iterations=570,
    thread_count=-1,
    task_type='GPU',
    random_seed=42,
    #leaf_estimation_method='Newton',
    bootstrap_type='MVS',
    random_strength=0.15,
    verbose=0
)

final_model.fit(
    X, y,
    cat_features=cat_features
)

y_test_pred = final_model.predict_proba(test_df)[:, 1]


feat_importance = final_model.get_feature_importance(prettified=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importances', y='Feature Id', data=feat_importance.sort_values('Importances', ascending=False))
plt.title('CatBoost Feature Importance')
plt.tight_layout()
plt.show()


explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)


submission = pd.DataFrame({
    'id': test['id'], 
    'Exited': y_test_pred
})

submission.to_csv('test_preds_2706_5.csv', index=False, sep=',')

