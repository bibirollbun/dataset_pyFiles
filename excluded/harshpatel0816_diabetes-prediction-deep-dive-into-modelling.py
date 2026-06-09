import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, classification_report, roc_curve

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train.head()


test.head()


target = 'diagnosed_diabetes'
id_col = 'id'


X = train.drop(columns=[target, id_col])
y = train[target]


X_train, X_valid, y_train, y_valid = train_test_split(
X, y,
test_size=0.2,
stratify=y,
random_state=42
)


nominal_features = [
    'gender',
    'ethnicity',
    'employment_status'
]

ordinal_features = [
    'education_level',
    'income_level',
    'smoking_status'
]


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns


ordinal_categories = [
    ['No formal', 'Highschool', 'Graduate', 'Postgraduate'],
    ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'],
    ['Never', 'Former', 'Current']
]


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),

        ('ord', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(categories=ordinal_categories))
        ]), ordinal_features),

        ('nom', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ]), nominal_features)
    ],
    remainder='drop'
)


log_reg_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', LogisticRegression(
        max_iter=1000,
        solver='lbfgs'
    ))
])

log_reg_pipeline.fit(X_train, y_train)


y_valid_prob = log_reg_pipeline.predict_proba(X_valid)[:,1]
roc_auc_logreg = roc_auc_score(y_valid, y_valid_prob)


print(roc_auc_logreg)


fpr, tpr, _ = roc_curve(y_valid, y_valid_prob)


plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc_logreg:.4f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc_logreg = cross_val_score(log_reg_pipeline, X, y, cv=cv, scoring='roc_auc')
print('LogReg CV AUC:', cv_auc_logreg.mean())


sns.kdeplot(y_valid_prob, label='LogReg')
plt.title('Predicted Probability Distribution')
plt.legend()
plt.show()


rf_pipeline = Pipeline([
('prep', preprocessor),
('model', RandomForestClassifier(
n_estimators=300,
max_depth=None,
min_samples_leaf=20,
n_jobs=-1,
random_state=42
))
])


rf_pipeline.fit(X_train, y_train)


y_valid_prob_rf = rf_pipeline.predict_proba(X_valid)[:,1]
roc_auc_rf = roc_auc_score(y_valid, y_valid_prob_rf)


roc_auc_rf


sns.kdeplot(y_valid_prob_rf, label='RandomForest')
plt.title('Predicted Probability Distribution')
plt.legend()
plt.show()


from sklearn.ensemble import ExtraTreesClassifier


et_pipeline = Pipeline([
('prep', preprocessor),
('model', ExtraTreesClassifier(
n_estimators=400,
max_depth=None,
min_samples_leaf=20,
n_jobs=-1,
random_state=42
))
])


et_pipeline.fit(X_train, y_train)


y_valid_prob_et = et_pipeline.predict_proba(X_valid)[:,1]
roc_auc_et = roc_auc_score(y_valid, y_valid_prob_et)
roc_auc_et


sns.kdeplot(y_valid_prob_et, label='RandomForest')
plt.title('Predicted Probability Distribution')
plt.legend()
plt.show()


from sklearn.ensemble import GradientBoostingClassifier


gb_pipeline = Pipeline ([
    ('prep', preprocessor),
    ('model', GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42))
])


gb_pipeline.fit(X_train, y_train)


y_valid_prob_gb = gb_pipeline.predict_proba(X_valid)[:,1]
roc_auc_gb = roc_auc_score(y_valid, y_valid_prob_gb)
roc_auc_gb


sns.kdeplot(y_valid_prob_gb, label='RandomForest')
plt.title('Predicted Probability Distribution')
plt.legend()
plt.show()


from xgboost import XGBClassifier

xgb_pipeline = Pipeline ([
    ('prep', preprocessor),
    ('model', XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        tree_method='hist',
        random_state=42))
])


xgb_pipeline.fit(X_train, y_train)


y_valid_prob_xgb = xgb_pipeline.predict_proba(X_valid)[:,1]
roc_auc_xgb = roc_auc_score(y_valid, y_valid_prob_xgb)
roc_auc_xgb


sns.kdeplot(y_valid_prob_xgb, label='RandomForest')
plt.title('Predicted Probability Distribution')
plt.legend()
plt.show()


import lightgbm as lgb

lgbm_pipeline = Pipeline ([
    ('prep', preprocessor),
    ('model', lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary',
        random_state=42))
])


lgbm_pipeline.fit(X_train, y_train)


y_valid_prob_lgbm = lgbm_pipeline.predict_proba(X_valid)[:,1]
roc_auc_lgbm = roc_auc_score(y_valid, y_valid_prob_lgbm)
roc_auc_lgbm


sns.kdeplot(y_valid_prob_lgbm, label='RandomForest')
plt.title('Predicted Probability Distribution')
plt.legend()
plt.show()


print(f"LogReg AUC: {roc_auc_logreg:.4f}")
print(f"RF AUC : {roc_auc_rf:.4f}")
print(f"ET AUC : {roc_auc_et:.4f}")
print(f"GB AUC : {roc_auc_gb:.4f}")
print(f"XGB AUC : {roc_auc_xgb:.4f}")
print(f"LGB AUC : {roc_auc_lgbm:.4f}")


X_test = test.drop(columns=['id'])


test_pred_proba = lgbm_pipeline.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': test_pred_proba
})

submission.head()


submission.to_csv('submission.csv', index=False)


test_pred_proba = xgb_pipeline.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': test_pred_proba
})

submission.head()


submission.to_csv('submission_xgboost.csv', index=False)


test_pred_proba = gb_pipeline.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': test_pred_proba
})

submission.head()


submission.to_csv('submission_gb.csv', index=False)


test_pred_proba = rf_pipeline.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': test_pred_proba
})

submission.head()


submission.to_csv('submission_rf.csv', index=False)

