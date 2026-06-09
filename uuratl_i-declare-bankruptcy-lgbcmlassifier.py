import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


path = '/kaggle/input/playground-series-s5e11'
train_data = pd.read_csv(os.path.join(path, 'train.csv'))
train_df = train_data.copy()
train_df.head()


train_df.info()


train_df.describe().T


train_df = train_df.drop(columns=['id'])
cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()


for col in cat_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_df, x=col)
    plt.title(f'Distribution of {col.capitalize()}')
    plt.show()


for col in num_cols:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1])
    
    # --- Histogram ---
    sns.histplot(data=train_df, x=col, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Distribution of {col}', fontsize=14)
    axes[0].set_xlabel('')  # remove x label to save space
    
    # --- Boxplot ---
    sns.boxplot(data=train_df, x=col, ax=axes[1], color='skyblue')
    axes[1].set_xlabel(col, fontsize=12)
    
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(6, 6))
plt.pie(
    train_df["loan_paid_back"].value_counts(),
    labels=train_df["loan_paid_back"].value_counts().index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Target Distribution")
plt.show()


train_df["loan_paid_back"].value_counts(normalize=True)


import optuna
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


pos = train_df["loan_paid_back"].sum()
neg = len(train_df) - pos
scale_pos_weight = neg / pos
print(f'Scale Pos Weight: {scale_pos_weight}')


X = train_df.drop(columns=['loan_paid_back'])
y = train_df['loan_paid_back']


for col in cat_cols:
    X[col] = X[col].astype("category")


def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 100,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 512, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'scale_pos_weight': scale_pos_weight,
        'device_type': 'cpu',
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1 
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='auc',
            categorical_feature=cat_cols,
            callbacks=[
                early_stopping(200),
                log_evaluation(0)
            ]
        )

        preds = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)
        auc_scores.append(auc)

    return np.mean(auc_scores)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)


print("âœ… Best AUC:", study.best_value)
print("ğŸ�† Best Params:", study.best_params)


best_model = lgb.LGBMClassifier(**study.best_params)
best_model.fit(X, y)


import shap





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)


shap.summary_plot(shap_values[1], X_test)


shap.summary_plot(shap_values[1], X_test, plot_type="bar")


shap_importance = pd.DataFrame({
    'feature': X_test.columns,
    'mean_abs_shap': np.abs(shap_values[1]).mean(axis=0)
}).sort_values(by='mean_abs_shap', ascending=False)
shap_importance


top_features = shap_importance['feature'].values[: 5]
top_features


for feature in top_features:
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    interaction_strengths = shap.approximate_interactions(feature, shap_values[1], X_test)
    top_interactions = [X_test.columns[i] for i in interaction_strengths[:5]]
    for i, interaction in enumerate(top_interactions):
        shap.dependence_plot(
            feature,
            shap_values[1],
            X_test,                  
            ax=axes[i],            
            show=False,            
            interaction_index=interaction )
        axes[i].set_title(f'{interaction.capitalize()} vs {feature.capitalize()}')
    plt.tight_layout()
    plt.show()


submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
test_data = pd.read_csv(os.path.join(path, 'test.csv'))
submission.head()


test_data = test_data.drop(columns=['id'])
for col in cat_cols:
    test_data[col] = test_data[col].astype("category")


predictions = best_model.predict_proba(test_data)
submission['loan_paid_back'] = predictions[:, 1]


submission.to_csv('submission.csv', index=False)

