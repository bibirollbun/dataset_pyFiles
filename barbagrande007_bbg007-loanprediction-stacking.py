import warnings
warnings.filterwarnings('ignore')

import os
import time
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, IsolationForest, StackingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import optuna


# Set Matplotlib defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rc("figure", autolayout=True, figsize=(16, 12))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
),
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=True,
)


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')
submit = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


train.head()


def create_summary(df):
    describe = df.describe().transpose()
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary["MissingValues"] = df.isna().sum()
    summary["UniqueValues"] = df.nunique()
    summary["Value_1"] = df.iloc[0]
    summary["Value_2"] = df.iloc[1]
    summary["Value_3"] = df.iloc[2]
    summary = pd.concat([summary, describe], axis=1)
    
    return summary

create_summary(train)


cols = 3
rows = int(np.ceil(len(train.columns) / cols))

fig,ax = plt.subplots(nrows=rows,ncols=cols,figsize=(20,20))
ax = ax.flatten()

plt.suptitle("Visualize all features",size=24, y=1.01)

for i,col in enumerate(train.columns):
    if train[col].dtype == float or train[col].dtype == int:
        sns.boxplot(data=train,y=col,ax=ax[i],orient="vertical", hue='loan_paid_back')
        ax[i].set_title(f"{col}")
    else:
        sns.countplot(data=train,x=col,ax=ax[i], hue='loan_paid_back')
        ax[i].set_title(f"{col}")
        ax[i].tick_params(axis='x', rotation=90)

# Remove empty subplots
for i in range(len(train.columns), len(ax)):
    fig.delaxes(ax[i])

plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
plt.title("Distribution of Target Variable - loan_paid_back")
sns.countplot(data=train, x='loan_paid_back')
plt.show()


class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train['loan_paid_back']), y=train['loan_paid_back'])
class_weights = dict(enumerate(class_weights))
class_weights


def feature_engineering(df):

    # Extracting grade and subgrade from subgrade column
    df['grade'] = df['grade_subgrade'].str[0]
    df['subgrade_num'] = df['grade_subgrade'].str[1].astype(int)
    grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    ordinal_encoder = OrdinalEncoder(categories=[grade_order])
    df['grade_encoded'] = ordinal_encoder.fit_transform(df[['grade']]).astype(int)
    df['grade_subgrade_score'] = (df['grade_encoded'] - 1) * 5 + df['subgrade_num']
    
    # Simple log transformations for skewed data
    df['loan_amount_log'] = np.log1p(df['loan_amount'])
    df['annual_income_log'] = np.log1p(df['annual_income'])
    
    # Predictive interactions
    df['grade_income_interaction'] = df['grade_encoded'] * df['annual_income']
    df['grade_loan_interaction'] = df['grade_encoded'] * df['loan_amount']
    
    # Drop columns that are no longer needed
    df.drop(columns=['grade_subgrade', 'grade'], inplace=True)

    return df

train = feature_engineering(train)
test = feature_engineering(test)


train.head()


X = train.drop(columns=['loan_paid_back'])
y = train['loan_paid_back']
X_test = test.copy()


num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)


# Outlier removal using IsolationForest
iso = IsolationForest(contamination='auto')

iso_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('outliers', iso)
])

outlier_mask = iso_pipeline.fit_predict(X) == 1

X = X[outlier_mask]
y = y[outlier_mask]

print(f"Removed {sum(~outlier_mask)} outliers from training data")
print(f"New training data shape: {X.shape}")


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=51, stratify=y)


# # FIXED parameters
# fixed_cbc_params = {
#     "loss_function": "Logloss",
#     "eval_metric": "AUC",
#     "grow_policy": "SymmetricTree",
#     "boosting_type": "Plain",
#     "verbose": False,
#     "random_seed": 51,
# }

# fixed_xgb_params = {
#     'random_state': 51,
# }


# def objective(trial):
    
#     cbc_params = {
#         # CRITICAL parameters with extended ranges
#         "iterations": trial.suggest_int("iterations", 800, 3000), 
#         "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
#         "depth": trial.suggest_int("depth", 4, 10),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2, 25),
        
#         # IMPORTANT parameters with refined ranges
#         "rsm": trial.suggest_float("rsm", 0.7, 1.0),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 150),
#         "class_weights": trial.suggest_categorical("class_weights", [class_weights, None]),
#         "random_strength": trial.suggest_float("random_strength", 0.5, 6.0),
#     }

#     xgb_params = {
#         "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
#         "class_weight": trial.suggest_categorical("class_weight", ["balanced", None, class_weights]),
#     }
    
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=51)

#     model1 = cb.CatBoostClassifier(**cbc_params, **fixed_cbc_params)
#     model2 = xgb.XGBClassifier(**xgb_params, **fixed_xgb_params)

#     stacking = StackingClassifier([
#         ('model1', model1),
#         ('model2', model2)
#     ], final_estimator=LogisticRegression(),cv=cv, stack_method='predict_proba')
    
    

#     stacking_pipeline = Pipeline(steps=[
#         ("preprocessor", preprocessor),
#         ("classifier", stacking)
#     ])
    
#     scores = cross_val_score(stacking_pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    
#     return np.mean(scores)

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)


# study.best_params


model1_params = {
    'iterations': 1080,
    'learning_rate': 0.20597877617323748,
    'depth': 4,
    'l2_leaf_reg': 16.60639473944353,
    'rsm': 0.9018852571321285,
    'min_data_in_leaf': 135,
    'class_weights': None,
    'random_strength': 4.546569697115103
}

model2_params = {
    'n_estimators': 674,
    'max_depth': 7,
    'subsample': 0.8580102762954551,
    'colsample_bytree': 0.9752253743878795,
    'reg_alpha': 2.4253463679024767,
    'reg_lambda': 8.532888690555001,
    'class_weight': {0: 2.4853305439330544, 1: 0.6259236154724823}
}

cv = StratifiedKFold(n_splits=20, shuffle=True, random_state=51)

model1 = cb.CatBoostClassifier(**model1_params, verbose=0, random_seed=51)
model2 = xgb.XGBClassifier(**model2_params, random_state=51)



best_model = StackingClassifier([
        ('model1', model1),
        ('model2', model2)
    ], final_estimator=LogisticRegression(),cv=cv, stack_method='predict_proba')

best_model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", best_model)
    ])

best_model_pipeline.fit(X_train, y_train)


y_pred = best_model_pipeline.predict_proba(X_val)[:, 1]
y_pred_binary = (y_pred > 0.5).astype(int)
auc = roc_auc_score(y_val, y_pred)

fig, ax = plt.subplots(1,2,figsize=(20,6))

ConfusionMatrixDisplay.from_predictions(y_val, y_pred_binary, ax=ax[0])
ax[0].grid(False)
ax[0].set_title("Confusion Matrix")
RocCurveDisplay.from_predictions(y_val, y_pred, ax=ax[1])
ax[1].set_title(f"ROC AUC Curve (AUC = {auc:.4f})")
plt.show()


best_model_pipeline.fit(X, y)

results = best_model_pipeline.predict_proba(X_test)[:, 1]



submit['loan_paid_back'] = results

submit.to_csv('submission.csv', index=False)
submit.head()




