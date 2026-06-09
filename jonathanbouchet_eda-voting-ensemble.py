import numpy as np 
import pandas as pd
pd.set_option('display.max_columns', 220)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import time

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv("/kaggle/input/bank-churn-lessgo/train.csv", sep=",")
df_test = pd.read_csv("/kaggle/input/bank-churn-lessgo/test.csv", sep=",")


df_train.head()


df_train.dtypes


pd.DataFrame(df_train.isnull().sum()).transpose()


df_train["Exited"].value_counts()


def plot_feature(c: str) -> None:
    """make a boxplot by target variable and histograms for both classes for feature 'c'
    args: c: str: numerical feature
    returns: plot
    """
    fig, ax = plt.subplots(1,2, figsize=(12,2))
    sns.boxplot(data=df_train, x="Exited", y=c, ax=ax[0], linewidth=1)
    sns.histplot(data=df_train[df_train["Exited"] == 0][c], label="not churned", ax=ax[1], bins=50)
    sns.histplot(data=df_train[df_train["Exited"] == 1][c], label="churned", ax=ax[1], bins=50)
    plt.legend(loc='best')
    plt.show()

for c in ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']:
    plot_feature(c)


fig, ax = plt.subplots(4,2, figsize=(12,8))
for cnt, feature in enumerate(['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']):
    i = int(cnt/2)
    j = int(cnt%2)
    sns.boxplot(data=df_train, x="Gender", y=feature, ax=ax[i][j], linewidth=1)
    plt.legend(loc='best')
    plt.tight_layout()
plt.show()


fig, ax = plt.subplots(4,2, figsize=(12,8))
for cnt, feature in enumerate(['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']):
    i = int(cnt/2)
    j = int(cnt%2)
    sns.boxplot(data=df_train, x="Geography", y=feature, ax=ax[i][j], linewidth=1)
    plt.legend(loc='best')
    plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1,2, figsize=(12,3))
sns.countplot(data=df_train, x="Geography", ax=ax[0])
sns.countplot(data=df_train, x="Gender", ax=ax[1])
plt.show()


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, cross_validate
from sklearn import set_config
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression


# convert some features to categorical. I cast to integer first to remove the 0 after the comma
df_train["NumOfProducts"] = df_train["NumOfProducts"].apply(lambda x: str(int(x)))
df_train["HasCrCard"] = df_train["HasCrCard"].apply(lambda x: str(int(x)))
df_train["IsActiveMember"] = df_train["IsActiveMember"].apply(lambda x: str(int(x)))

df_test["NumOfProducts"] = df_test["NumOfProducts"].apply(lambda x: str(int(x)))
df_test["HasCrCard"] = df_test["HasCrCard"].apply(lambda x: str(int(x)))
df_test["IsActiveMember"] = df_test["IsActiveMember"].apply(lambda x: str(int(x)))


num_cols: list[str] = ['CreditScore','Age','Tenure','Balance','EstimatedSalary']
cat_cols: list[str] = ['Geography','Gender','NumOfProducts','HasCrCard','IsActiveMember']
target: str = "Exited"

X = df_train[num_cols + cat_cols]
y = df_train[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=y,
    shuffle=True,
    random_state=1234)

print(f"train: {X_train.shape}, test: {X_test.shape}")


# Create a pipeline
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), 
           ("scaler", StandardScaler())]
)

categorical_onehot_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_onehot_transformer, cat_cols),
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor), 
    ('classifier', RandomForestClassifier(random_state=1234))
])

pipeline


param_grid = {
    'classifier__max_depth': [6],
    'classifier__n_estimators':[100],
    'classifier__class_weight': ['balanced', {0:1, 1:1}, {0:2, 1:1}, {0:5, 1:1}, {0:10, 1:1}, {0:1, 1:2}, {0:1, 1:5}, {0:1, 1:10}],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', verbose=1)
start = time.time()
grid_search.fit(X_train, y_train)
end = time.time()

# Print the best hyperparameters and score
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
print(f"timing: {end-start}")

# Evaluate the best model on the train set
best_model = grid_search.best_estimator_
train_score = best_model.score(X_train, y_train)
print("Train score:", train_score)

# Evaluate the best model on the test set
test_score = best_model.score(X_test, y_test)
print("Test score:", test_score)



grid_scores = grid_search.cv_results_
result_df = pd.DataFrame.from_dict(grid_scores, orient='columns')

cols = ['param_classifier__class_weight','mean_test_score','std_test_score','rank_test_score']
result_df[cols].style.background_gradient(cmap="viridis")


# Create a base pipeline
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), 
           ("scaler", StandardScaler())]
)

categorical_onehot_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_onehot_transformer, cat_cols),
    ],
    remainder='passthrough'
)

rf = RandomForestClassifier(random_state=1234)
xgboost = xgb.XGBClassifier(objective="binary:logistic")
ens = VotingClassifier(estimators=[("rf", rf), ("xgboost", xgboost)], voting='soft')

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor), 
    ('voting_ensemble', ens)
])

pipeline



# change to your needs
param_grid = {
'voting_ensemble__rf__max_depth': [6],
'voting_ensemble__rf__class_weight': ['balanced', {0:1, 1:1}],
'voting_ensemble__rf__n_estimators':[100],
'voting_ensemble__xgboost__max_depth': [6],
'voting_ensemble__xgboost__n_estimators':[100],
'voting_ensemble__xgboost__learning_rate': [0.01, 0.1],
'voting_ensemble__xgboost__scale_pos_weight': [1, 2, 4],
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', verbose=1)
start = time.time()
grid_search.fit(X_train, y_train)
end = time.time()

# Print the best hyperparameters and score
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
print(f"timing: {end-start}")

# Evaluate the best model on the train set
best_model = grid_search.best_estimator_
train_score = best_model.score(X_train, y_train)
print("Train score:", train_score)

# Evaluate the best model on the test set
test_score = best_model.score(X_test, y_test)
print("Test score:", test_score)



# run predictions
df_test = df_test[num_cols + cat_cols]
preds = best_model.predict(df_test)
preds_probs = best_model.predict_proba(df_test)[:,1]


from sklearn.metrics import roc_curve, RocCurveDisplay, auc

fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(3,3))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.5f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.legend(loc="best", fontsize=10)
plt.show()




