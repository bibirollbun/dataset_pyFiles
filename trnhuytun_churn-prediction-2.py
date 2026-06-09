import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import shap
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
from itertools import combinations
import joblib


train = pd.read_csv("/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv")
test = pd.read_csv("/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv")


#Preprocessing
train['HasCrCard'] = train['HasCrCard'].astype('int8')
train['IsActiveMember'] = train['IsActiveMember'].astype('int8')

# Drop unused columns
features2drop = ['id', 'CustomerId', 'Surname']
target = 'Exited'

# Final feature set
filtered_features = [i for i in train.columns if (i not in features2drop and i != target)]
X = train[filtered_features]
y = train[target]

# Apply same feature drop to test
test_df = test.drop(features2drop, axis=1)

# Encode categorical variables for XGBoost
X = pd.get_dummies(X, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)

# Align train/test features (in case test is missing some dummy columns)
X, test_df = X.align(test_df, join="left", axis=1, fill_value=0)


#Optuna Objective Function
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "auc",
        "tree_method": "gpu_hist"  # or "hist" if no GPU
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, valid_idx in cv.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=False
        )

        preds = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, preds)
        scores.append(score)

    return np.mean(scores)


#Run Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best parameters:", study.best_params)
print("Best ROC AUC:", study.best_value)

joblib.dump(study, 'optuna_study_xgb.pkl')


#Final Model Training
best_params = study.best_params
final_model = XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric="auc",
    tree_method="gpu_hist",
    random_state=42
)

final_model.fit(X, y)


#Submission
y_test_pred = final_model.predict_proba(test_df)[:, 1]

submission = pd.DataFrame({
    'id': test['id'],
    'Exited': y_test_pred
})
submission.to_csv("submission_xgb_optuna.csv", index=False)
print("Submission saved: submission_xgb_optuna.csv")


#Feature Importance & SHAP
feat_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': final_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_importance)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)

