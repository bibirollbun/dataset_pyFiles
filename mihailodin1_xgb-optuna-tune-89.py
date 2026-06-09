!pip install plotly -q
!pip install shap -q


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
plt.rcParams.update({
    'figure.figsize': (16, 8),    
    'figure.facecolor': 'white',    
    'figure.autolayout': True,     
})

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler

from xgboost import XGBRegressor

import shap

# from catboost import CatBoostRegressor
# from catboost.utils import get_gpu_device_count


# import lightgbm as lgb
# from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings("ignore")


formula_train = pd.read_csv("/kaggle/input/critical-temperature-of-superconductors/formula_train.csv")
formula_test = pd.read_csv("/kaggle/input/critical-temperature-of-superconductors/formula_test.csv")

train = pd.read_csv("/kaggle/input/critical-temperature-of-superconductors/train.csv")
test = pd.read_csv("/kaggle/input/critical-temperature-of-superconductors/test.csv")


train.head()


print(f"Na: {train.isna().sum()[train.isna().sum() > 0]}")
print(f"Na: {formula_train.isna().sum()[formula_train.isna().sum() > 0]}")


formula_train.head()


combined = pd.concat([formula_train, formula_test], axis=0)
zero_cols = list(combined.columns[(combined == 0).all()]) + ["material"]

formula_train.drop(columns=zero_cols, inplace=True)
formula_train.drop(columns="critical_temp", inplace=True)
formula_test.drop(columns=zero_cols, inplace=True)


train_sum = (formula_train.sum() / formula_train.shape[0]).reset_index()
test_sum = (formula_test.sum() / formula_test.shape[0]).reset_index() 

train_sum.columns = ["element", "total"]
test_sum.columns = ["element", "total"]

train_sum["dataset"] = "train"
test_sum["dataset"] = "test"

sum_df = pd.concat([train_sum, test_sum], ignore_index=True)


sns.barplot(data=sum_df, x="element", y="total", hue="dataset", palette="Spectral")
plt.xticks(rotation=90)
plt.title("Относительная доля химических элементов в train и test")
plt.show()


discrete = [
    "number_of_elements",
    "range_Valence"
]

num_param = [
    col for col in test.columns if col not in discrete
]


train['dataset'] = 'train'
test['dataset'] = 'test'

visualisation_df = pd.concat([train, test], axis=0)

train.drop(columns='dataset', inplace=True)
test.drop(columns='dataset', inplace=True)


for features in num_param[:15]:
    fig, ax = plt.subplots(nrows=1, ncols=3)
    sns.boxplot(data=visualisation_df, y='dataset', x=features, ax=ax[0], orient='h', palette='hot')
    sns.violinplot(data=visualisation_df, y='dataset', x=features, ax=ax[1], palette='Spectral')
    sns.histplot(data=visualisation_df, hue='dataset', x=features, ax=ax[2], kde=True, bins=20)
    plt.show()


correlation_matrix = train.corr(numeric_only=True) 
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot = True, fmt = '.2f', cmap = 'coolwarm', mask=mask)
plt.title('Corr matrix')
plt.show()


X = train.drop(columns="critical_temp")
target = train["critical_temp"]


# scaler = RobustScaler().set_output(transform="pandas")

# scaler.fit(X)

# X = scaler.transform(X)
# test = scaler.transform(test)


X = pd.concat([X, formula_train], axis=1)
test = pd.concat([test, formula_test], axis=1)


def objective_xgb(trial):
    max_depth=trial.suggest_int("max_depth", 5, 8)
    n_estimators=trial.suggest_int("n_estimators", 120, 200)
    min_child_weight=trial.suggest_int("min_child_weight", 1, 100)
    subsample=trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0)
    reg_alpha=trial.suggest_float("reg_alpha", 0, 1)
    reg_lambda=trial.suggest_float("reg_lambda", 0, 1)
    

    model = XGBRegressor(
        tree_method = "hist", 
        device = "cuda",
        random_state=42,
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda
    )

    scores = cross_val_score(
        model, 
        X, 
        target, 
        cv=3,  
        scoring="neg_mean_squared_error"  
    )

    rmse_scores = np.sqrt(-scores)  
    
    return rmse_scores.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective_xgb, n_trials=50)


plot_optimization_history(study).show()
plot_param_importances(study).show()
plot_contour(study, params=["max_depth", "n_estimators"]).show()
plot_slice(study, params=["max_depth", "subsample", "n_estimators"]).show()


model_xgb = XGBRegressor(
    tree_method = "hist", 
    device = "cuda",
    random_state=42,
    **study.best_trial.params
)


model_xgb.fit(
    X, 
    target,
    eval_metric="rmse",
    verbose=False
)


explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X.iloc[:10000, ])

shap.summary_plot(shap_values, X.iloc[:10000, ])


shap.initjs()
print('Correct answer:', target.iloc[2])
shap.force_plot(explainer.expected_value, shap_values[2,:], X.iloc[2,:])


pred = model_xgb.predict(test)


submission = pd.DataFrame({
    'index': test.index,
    'critical_temp': pred
})
submission.to_csv('submission.csv', index=False)


submission

