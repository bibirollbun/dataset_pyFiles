import numpy as np
import pandas as pd
import math
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col="id")


train.head()


train.isna().sum()


test.isna().sum()


categorical_cols = test.select_dtypes(include=['object']).columns
numerical_cols = test.select_dtypes(include=['int64', 'float64']).columns

print(f"The categorical value columns are: {categorical_cols.values}")
print(f"The numerical value columns are: {numerical_cols.values}")


sns.set_style('whitegrid')
sns.countplot(data=train, x='y', palette='Set2')
plt.title('Count of y')
plt.xlabel('y')
plt.ylabel('Count')
plt.show()


n_plots = len(categorical_cols)
cols_per_row = math.ceil(n_plots / 3)

plt.figure(figsize=(5 * cols_per_row, 10))

for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, cols_per_row, i)
    sns.countplot(x=col, hue='y', data=train, palette='Set2')
    plt.title(f"{col} vs y count")
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


n_plots = len(numerical_cols)
cols_per_row = math.ceil(n_plots / 3)

plt.figure(figsize=(5 * cols_per_row, 10))

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, cols_per_row, i)
    sns.histplot(x=col, hue='y', data=train, fill=True, palette='Set2', bins=30)
    plt.title(f"{col} vs y count")

plt.tight_layout()
plt.show()


encoder = OrdinalEncoder()
train[categorical_cols] = encoder.fit_transform(train[categorical_cols])
test[categorical_cols] = encoder.transform(test[categorical_cols])


X = train.drop('y', axis=1)
y = train['y']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22, stratify=y)


def objective_xgb(trial):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'logloss',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'random_state': 22,
        'use_label_encoder': False
    }

    model = XGBClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    return score.mean()

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=30)
best_xgb = XGBClassifier(**study_xgb.best_params, objective='binary:logistic', 
                         tree_method='hist', device='cuda', use_label_encoder=False, random_state=22)


print(f"Best XGB parameters:{study_xgb.best_params}")


def objective_lgbm(trial):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    params = {
        'objective': 'binary',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int("num_leaves", 31, 200),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
        'device': 'gpu',
        'random_state': 22,
        'verbosity': -1
    }

    model = LGBMClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    return score.mean()

study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(objective_lgbm, n_trials=30)

best_lgbm = LGBMClassifier(**study_lgbm.best_params, objective='binary', 
                           device='gpu', random_state=22, verbosity=-1)


print(f"Best LightGBM parameters:{study_lgbm.best_params}")


meta_model = XGBClassifier(
    objective='binary:logistic',
    tree_method='hist',
    device='cuda',
)

model = StackingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('lgbm', best_lgbm),
    ],
    final_estimator=meta_model,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    passthrough=True,
    stack_method='predict_proba'
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"Classifier Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}")
print(f"Classification Report: \n{classification_report(y_test, y_pred)}")


sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
test_preds = model.predict_proba(test)
test_preds_proba = test_preds[:, 1]
submission = pd.DataFrame({
    'id': sub['id'],
    'y': test_preds_proba
})

submission.to_csv('submission.csv', index=False)




