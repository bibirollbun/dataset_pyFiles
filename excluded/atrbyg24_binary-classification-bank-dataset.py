import pandas as pd
import numpy as np
import warnings

import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline, Pipeline
import optuna

import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

%matplotlib inline


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv',index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv',index_col = 'id')


train.info()


train.describe()


train.head()


num_cols = train.select_dtypes(include='number').columns.tolist()
cat_cols = train.select_dtypes(include = 'object').columns.tolist()


for col in cat_cols:
    print(f'\n {train[col].value_counts()}')


for col in num_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=train,x=col,hue='y')
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(18, 10))
for i, col in enumerate(num_cols):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(data=train, y=col)
    plt.title(f"Boxplot: {col}")
    plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(
    train[num_cols].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Between Numerical Features", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder()
)

log_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    FunctionTransformer(np.log1p, feature_names_out='one-to-one'),
    StandardScaler()
)

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

preprocessing = ColumnTransformer([
    ('log', log_pipeline, ['balance','duration','campaign']),
    ('cat', cat_pipeline, make_column_selector(dtype_include=object))
    ],
    remainder=default_num_pipeline)

model = Pipeline(steps=[
    ('preprocessor', preprocessing),
    ('scaler', StandardScaler()),
    ('classifier', lgb.LGBMClassifier(objective = 'binary',random_state=42,verbose=-1))
])


X = train.drop('y', axis=1).copy()
y = train['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=17)


model.fit(X_train, y_train)


test_pred = model.predict_proba(X_test)[:,1]
test_labels = (test_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, test_labels)
roc_auc = roc_auc_score(y_test, test_pred)
print(f"Accuracy: {accuracy}")
print(f"ROC AUC Score: {roc_auc}")
print(classification_report(y_test, test_labels))


def objective(trial):
    params = {
        'classifier__n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'classifier__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'classifier__max_depth': trial.suggest_int('max_depth', 4, 10),
        'classifier__subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'classifier__colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    
    model = Pipeline(steps=[
    ('preprocessor', preprocessing),
    ('scaler', StandardScaler()),
    ('classifier', lgb.LGBMClassifier(objective = 'binary',random_state=42,verbose=-1))
])
    
    model.set_params(**params)
    
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize') 
study.optimize(objective, n_trials=50) 

print("Best trial:", study.best_trial)
best_lgm = Pipeline([
    ('preprocessor', preprocessing),
    ('scaler', StandardScaler()),
    ('classifier', lgb.LGBMClassifier(**study.best_params, random_state=42))
])
best_lgm.fit(X_train, y_train)


test_pred = best_lgm.predict_proba(X_test)[:,1]
test_labels = (test_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, test_labels)
roc_auc = roc_auc_score(y_test, test_pred)
print(f"Accuracy: {accuracy}")
print(f"ROC AUC Score: {roc_auc}")
print(classification_report(y_test, test_labels))


submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
submission['y'] = best_lgm.predict_proba(test)[:,1]
submission.to_csv("submission.csv", index=False)




