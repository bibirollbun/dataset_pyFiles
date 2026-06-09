import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample_df = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


train.head()


train.info()


train.describe().T


train.drop(["id"], axis = 1, inplace = True)
test.drop(["id"], axis = 1, inplace = True)


plt.figure(figsize = (8,4))
ax = sns.countplot(data = train, x = "diagnosed_diabetes")
plt.title("Diabetes Count")
plt.xlabel("0:Not Diabet,  1:Diabet")

for container in ax.containers:
    ax.bar_label(container, fmt='%d')

plt.show()


plt.figure(figsize = (10,4))
ax = sns.countplot(data = train, x = "family_history_diabetes")
plt.title("Diabetes Count For family_history")
plt.xlabel("0:Not Diabet,  1:Diabet")

for container in ax.containers:
    ax.bar_label(container, fmt='%d')

plt.show()


plt.figure(figsize = (10,4))
ax = sns.countplot(train, x = "income_level", hue = "diagnosed_diabetes")
plt.title("Diabetes for Income Level")

for container in ax.containers:
    ax.bar_label(container, fmt='%d')

plt.show()


plt.figure(figsize=(10,4))
ax = sns.countplot(train, x="employment_status", hue="diagnosed_diabetes")
plt.title("Diabetes for employment_status")

for container in ax.containers:
    ax.bar_label(container, fmt='%d')

plt.show()


plt.figure(figsize=(10,4))
ax = sns.countplot(train, x = "ethnicity", hue="diagnosed_diabetes")
plt.title("Diabetes for ethinicity")

for container in ax.containers:
    ax.bar_label(container, fmt='%d')

plt.show()


target = "diagnosed_diabetes"
num_cols = train.select_dtypes(include = ["int", "float"]).columns.tolist()
cat_cols = train.select_dtypes(include = ["object"]).columns.tolist()


def hist_plot(df, col):
    plt.figure(figsize=(10,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

for i in num_cols:
    hist_plot(train, i)


num_cols_corr = train[num_cols].corr()
plt.figure(figsize = (10,5))
sns.heatmap(num_cols_corr, annot = True, fmt = ".2f", cmap = "magma")


train["BMI_Age"] = train["bmi"] * train["age"]
test["BMI_Age"]  = test["bmi"] * test["age"]

train["Glucose_BMI"] = train["cholesterol_total"] / (train["bmi"] + 1)
test["Glucose_BMI"]  = test["cholesterol_total"] / (test["bmi"] + 1)

for col in ["cholesterol_total", "triglycerides"]:
    train[f"log_{col}"] = np.log1p(train[col])
    test[f"log_{col}"]  = np.log1p(test[col])


for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")


train.head()


test.head()


X = train.drop(target, axis = 1)
y = train[target]

print("X shape:", X.shape)
print("y shape:", y.shape)


import optuna 
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

"""def objective_xgb(trial):

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'enable_categorical': True,
        'early_stopping_rounds': 100,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'device': 'cuda'
    }
    
    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        
        model.fit(
            X_train, y_train,
            eval_set = [(X_val, y_val)],
            verbose = False
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)
    
    return np.mean(fold_scores)

study = optuna.create_study(direction = "maximize")
study.optimize(objective_xgb, n_trials = 25)

print("En iyi parametreler:", study.best_params)
print(f"En iyi skor: {study.best_value:.5f}")"""


"""best_params = study.best_params.copy()
best_params.update({
    'objective': 'binary:logistic',
    'enable_categorical': True,
    'early_stopping_rounds': 100,
    'eval_metric': 'auc',
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'device': 'cuda'
})"""

best_params_xgb = {
    'n_estimators': 5000, 
    'learning_rate': 0.018008455787173833, 
    'max_depth': 5, 
    'subsample': 0.9634605502536008, 
    'colsample_bytree': 0.5707611758453766, 
    'colsample_bylevel': 0.6717095975845457,
    'enable_categorical': True,
    'eval_metric': 'auc',
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'device': 'cuda'
}

skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
oof_xgb = np.zeros(len(X))
pred_xgb = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**best_params_xgb)
    model.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)],
        verbose = False
    )
    
    oof_xgb[val_idx] = model.predict_proba(X_val)[:, 1]
    fold_auc = roc_auc_score(y_val, oof_xgb[val_idx])
    pred_xgb += model.predict_proba(test)[:, 1] / 10
    
    print(f"Fold {fold}: {fold_auc:.5f}")


overall_auc = roc_auc_score(y, oof_xgb)
print(f"Final OOF AUC: {overall_auc:.5f}")


import lightgbm as lgbm

"""def objective_lgbm(trial):
    params = { 
        
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log = True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200), 
        'max_depth': trial.suggest_int('max_depth', 3, 12), 
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100), 
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'random_state': 42,
        'enable_categorical': True,
        'device': 'gpu',
        'verbose': -1,
    }

    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X,y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgbm.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set = [(X_val, y_val)],
                 callbacks=[lgbm.early_stopping(stopping_rounds = 100, verbose = False), 
                            lgbm.log_evaluation(period = 0)]
                 )
            
        val_preds = model.predict_proba(X_val)[:,1]
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)

    return np.mean(fold_scores)

study2 = optuna.create_study(direction = "maximize")
study2.optimize(objective_lgbm, n_trials = 25)

print("En iyi parametreler:", study2.best_params)
print(f"En iyi skor: {study2.best_value:.5f}")"""


"""best_params2 = study2.best_params.copy()
best_params2.update({
    'objective': 'binary',
    'metric': 'auc',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'device': 'gpu'
})"""

best_params_lgbm = {
    'objective': 'binary',
    'n_estimators': 5000,
    'metric': 'auc',
    'learning_rate': 0.033165707040378425, 
    'num_leaves': 161,     
    'max_depth': 4,
    'min_child_samples': 75,  
    'feature_fraction': 0.41033015077037566,
    'bagging_fraction': 0.6624023800121338,
    'bagging_freq': 3,
    'enable_categorical': True,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1,
    'device': 'gpu'
}


skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

oof_lgbm = np.zeros(len(X))
pred_lgbm = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X,y), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgbm.LGBMClassifier(**best_params_lgbm)
        model.fit(X_train, y_train,
                 eval_set = [(X_val, y_val)],
                 callbacks=[lgbm.early_stopping(stopping_rounds = 100, verbose = False), 
                            lgbm.log_evaluation(period = 0)])

        oof_lgbm[val_idx] = model.predict_proba(X_val)[:,1]
        fold_auc = roc_auc_score(y_val, oof_lgbm[val_idx])
        pred_lgbm += model.predict_proba(test)[:, 1] / 10
    
        print(f"Fold {fold}: {fold_auc:.5f}")


overall_auc = roc_auc_score(y, oof_lgbm)
print(f"Final OOF AUC: {overall_auc:.5f}")


import catboost as ctb

best_params_ctbc = {
    "cat_features": cat_cols,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 5000,         
    "learning_rate": 0.04,
    "depth": 9,
    "l2_leaf_reg": 10,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.2,
    "min_data_in_leaf": 50,
    "od_type": "Iter",
    "od_wait": 100,
    "verbose": 0,
    "task_type": "GPU"  
}

skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

oof_ctbc = np.zeros(len(X))
pred_ctbc = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X,y), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = ctb.CatBoostClassifier(**best_params_ctbc)
        model.fit(X_train, y_train,
                 eval_set = [(X_val, y_val)])

        oof_ctbc[val_idx] = model.predict_proba(X_val)[:,1]
        fold_auc = roc_auc_score(y_val, oof_ctbc[val_idx])
        pred_ctbc += model.predict_proba(test)[:, 1] / 10
    
        print(f"Fold {fold}: {fold_auc:.5f}")


overall_auc = roc_auc_score(y, oof_ctbc)
print(f"Final OOF AUC: {overall_auc:.5f}")


final_oof = (0.6 * oof_lgbm) + (0.2 * oof_xgb) + (0.2 * oof_ctbc)
ensemble_score = roc_auc_score(y, final_oof)
print(f"Weighted Ensemble OOF AUC: {ensemble_score:.5f}")

test_predictions = (0.6 * pred_lgbm) + (0.2 * pred_xgb) + (0.2 * pred_ctbc)
print(f"Test prediction Score: {np.mean(test_predictions)}")


submission = pd.DataFrame({
    'id': sample_df["id"],
    'prediction': test_predictions
})

submission.to_csv('submission_ensemble_model19.csv', index=False)


plt.figure(figsize = (10,4))
sns.kdeplot(submission["prediction"], label = 'Test', fill = True, color = 'red', alpha = 1)
sns.kdeplot(final_oof, label = 'Train', fill = True, color = 'blue', alpha = 0.5)
plt.title('Train/Test Distribution')
plt.legend()
plt.tight_layout()
plt.show()


model_scores = {
    "LightGBM": roc_auc_score(y, oof_lgbm),
    "CatBoost": roc_auc_score(y, oof_ctbc),
    "XGBoost": roc_auc_score(y, oof_xgb),
    "Ensemble_Model": roc_auc_score(y, final_oof)
}

pd.DataFrame.from_dict(
    model_scores,
    orient = "index",
    columns = ["ROC-AUC"]
).sort_values("ROC-AUC", ascending = False)




