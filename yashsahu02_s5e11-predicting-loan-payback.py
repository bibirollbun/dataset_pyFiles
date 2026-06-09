# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


## shape of train and test 
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)


## first five rows (.head())
print("Frst Five Rows in Train: =>")
display(train.head())
print("-"*50)
print("Frst Five Rows in Test: =>")
display(test.head())


## sample submission file
sample_submission.head()


## check is any null values 
train.isnull().sum()


test.isnull().sum()


## there is no any need and use of id feature so drop it
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)


train.head(3)


import matplotlib.pyplot as plt 
import seaborn as sns


# ## function that plots the graphs and basic details 
# def show_details_and_plots(df, feature):
#     target_feature = "loan_paid_back"
    
#     # if the feature is object type (categorical)
#     if(df[feature].dtype=='O'):
#         print("Feature Name:", feature)
#         print(f"Total Unique Categories: {df[feature].nunique()}")
#         print("Value Counts:")
#         display(df[feature].value_counts())
#         print("*"*40)
        
#         plt.figure(figsize=(18,13))
#         plt.subplot(2,2,1)
#         sns.countplot(x=feature, data=df)
#         plt.title(f"Count Plot for {feature}")

#         plt.subplot(2,2,2)
#         # df[feature].value_counts().plot().pie(autopct='%1.1f%%')
#         plt.pie(data=df,x=df[feature].value_counts().values, labels=df[feature].value_counts().index, autopct='%.1f%%')
#         plt.title(f"Pie Chart for {feature}")

#         plt.subplot(2,2,3)
#         sns.boxplot(x=feature, y=target_feature, data=df)
#         plt.title(f"Boxplot for {feature}")

#         plt.subplot(2,2,4)
#         sns.barplot(x=feature, y=target_feature, data=df)
#         plt.title(f"Bar Plot for {feature}")
#         plt.show()
        
#     # for the numerical features
#     elif(df[feature].dtype!='O'):
#         plt.figure(figsize=(20,20))
#         plt.subplot(2,2,1)
#         sns.histplot(data=df, x=feature, kde=True)
#         plt.title(f"Histplot for {feature}")

#         plt.subplot(2,2,2)
#         sns.kdeplot(data=df, x=feature)
#         plt.title(f"kde Plot for {feature}")


#         plt.subplot(2,2,3)
#         sns.boxplot(x=df[feature])
#         plt.title(f"Box Plot for {feature}")

#         plt.subplot(2,2,4)
#         sns.scatterplot(data=df, x=feature, y=target_feature)
#         plt.title(f"Scatter Plot between {feature} and {target_feature}")
#         plt.show()
        
#     else:
#         print(f"{feature} is neither Numeric nor Categorical...")

#     print("*"*200)
#     print()



# for feature in train.columns:
#     show_details_and_plots(train, feature)


train.head()


print(train['grade_subgrade'].nunique())
print(test['grade_subgrade'].nunique())


grade_subgrade_freq_map = train['grade_subgrade'].value_counts().to_dict()


grade_subgrade_freq_map


from sklearn.preprocessing import LabelEncoder
gender_le = LabelEncoder()
marital_status_le = LabelEncoder()
education_level_le = LabelEncoder()
employment_status_le = LabelEncoder()
loan_purpose_le = LabelEncoder()

def preprocess(train, test):
    train['gender'] = gender_le.fit_transform(train['gender'])
    test['gender'] = gender_le.transform(test['gender'])
    
    train['marital_status'] = marital_status_le.fit_transform(train['marital_status'])
    test['marital_status'] = marital_status_le.transform(test['marital_status'])
    
    train['education_level'] = education_level_le.fit_transform(train['education_level'])
    test['education_level'] = education_level_le.transform(test['education_level'])
    
    train['employment_status'] = employment_status_le.fit_transform(train['employment_status'])
    test['employment_status'] = employment_status_le.transform(test['employment_status'])
    
    train['loan_purpose'] = loan_purpose_le.fit_transform(train['loan_purpose'])
    test['loan_purpose'] = loan_purpose_le.transform(test['loan_purpose'])

    train['grade_subgrade'] = train['grade_subgrade'].map(grade_subgrade_freq_map)
    test['grade_subgrade'] = test['grade_subgrade'].map(grade_subgrade_freq_map)
    return 0


preprocess(train, test)


train.head()


# def add_features(df):
#     # Ratio or Interaction features
#     df["credit_to_loan"] = df["credit_score"] / (df["loan_amount"] + 1)
#     df['income_to_credit_ratio'] = df['annual_income'] / df["credit_score"]
#     df["loan_interest_amt"] = df["loan_amount"] * df["interest_rate"]
#     df['total_payable_ammount'] = df['loan_amount'] + df["loan_interest_amt"]
#     df["percent_paid"] = df["total_payable_ammount"] / (df["loan_amount"] + 1)

#     # Log transformations
#     log_cols = ["loan_amount", "annual_income", "total_payable_ammount"]
#     for col in log_cols:
#         df[col+"_log"] = np.log1p(df[col])

#     # Interaction features
#     df["credit_income_interaction"] = df["credit_score"] * df["annual_income"]
#     return df


# train = add_features(train)
# test  = add_features(test)


train


test


X = train.drop('loan_paid_back',axis=1)
y = train['loan_paid_back']


X


y


y = y.astype('int')


y


sample_submission.head()


from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

def score_dataset(X, y, model=CatBoostClassifier(verbose=0)):
    X = X.copy()

    # Metric for classification: ROC AUC Score
    score = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="roc_auc"
    )

    return score.mean()



score_dataset(X, y)


def add_features(df):
    # Ratio or Interaction features
    df["credit_to_loan"] = df["credit_score"] / (df["loan_amount"] + 1)
    df['income_to_credit_ratio'] = df['annual_income'] / df["credit_score"]
    df["loan_interest_amt"] = df["loan_amount"] * df["interest_rate"]
    df['total_payable_ammount'] = df['loan_amount'] + df["loan_interest_amt"]
    df["percent_paid"] = df["total_payable_ammount"] / (df["loan_amount"] + 1)

    # Log transformations
    log_cols = ["loan_amount", "annual_income", "total_payable_ammount"]
    for col in log_cols:
        df[col+"_log"] = np.log1p(df[col])

    # Interaction features
    df["credit_income_interaction"] = df["credit_score"] * df["annual_income"]
    return df


train = add_features(train)
test  = add_features(test)


X2 = train.drop('loan_paid_back',axis=1)

score_dataset(X2, y)


def add_features(df):
    # Ratio or Interaction features
    df["credit_to_loan"] = df["credit_score"] / (df["loan_amount"] + 1)
    df['income_to_credit_ratio'] = df['annual_income'] / df["credit_score"]
    df["loan_interest_amt"] = df["loan_amount"] * df["interest_rate"]
    df['total_payable_ammount'] = df['loan_amount'] + df["loan_interest_amt"]
    df["percent_paid"] = df["total_payable_ammount"] / (df["loan_amount"] + 1)
    return df





train3 = add_features(train)
test3  = add_features(test)


train3


X3 = train3.drop('loan_paid_back',axis=1)

score_dataset(X3, y)


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

## there is no any need and use of id feature so drop it
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

preprocess(train, test)

train4 = add_features(train)
test4  = add_features(test)

X4 = train4.drop('loan_paid_back',axis=1)

score_dataset(X4, y)








from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# Dictionary of models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    # "RidgeClassifier": RidgeClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "ExtraTreesClassifier": ExtraTreesClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0),
    "LGBMClassifier": LGBMClassifier(verbose=-1),
    "CatBoostClassifier": CatBoostClassifier(verbose=0),
    "RandomForestClassifier": RandomForestClassifier(),
}

# define num fold 
N_SPLITS = 5
model_name_list = [] ## list to store name of model in every iteration
model_score_list = [] ## list to store avg score in every iteration

skf = StratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

# iterate in models dictionary 
for model_name, model in models.items():
    print(model_name,"==============>\n")
    score_list = []
    test_pred = np.zeros(test.shape[0])
    
    for i, (train_idx, test_idx) in enumerate(skf.split(X,y), 1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]

        ## train the model 
        model.fit(X_train_fold, y_train_fold)
        ## prediction 
        y_test_fold_pred = model.predict_proba(X_test_fold)[:,1]
        score = roc_auc_score(y_test_fold, y_test_fold_pred)
        print(f"Fold {i} ROC AUC Score: {score:.5f}")
        score_list.append(score)

        # Prediction on test data (for ensemble averaging)
        test_pred += model.predict_proba(test)[:,1]

    avg_score = np.mean(score_list)
    print(f"Average ROC AUC Score: {avg_score:.5f}")
    print('-'*60, '\n')

    # append model name and avg score in respective lists
    model_name_list.append(model_name)
    model_score_list.append(avg_score)

    # divide the pred by num_folds to get exact prediction in proper form 
    predicted_proba = test_pred/N_SPLITS
    ## save to sample_submission file 
    sample_submission['loan_paid_back'] = predicted_proba
    ## save the submisson file for each model
    sample_submission.to_csv(f"{model_name}_prediction.csv", index=False)
    print(f"File saved as {model_name}_prediction.csv")
    display(sample_submission.head())
    print("*"*60)


# Performance tracking of each model and how much it is scoring 
performance_df = pd.DataFrame({
    "Model Name": model_name_list,
    "ROC AUC Score": model_score_list
})


performance_df


performance_df.sort_values(by='ROC AUC Score', ascending=False)


import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Objective function for Optuna
def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 10000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "random_state": 42,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "gpu_hist",   # GPU acceleration
        "n_jobs": -1
    }

    model = XGBClassifier(**params, use_label_encoder=False)

    # 5-fold cross validation (ROC-AUC)
    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring='roc_auc'
    )

    return np.mean(scores)


# Create & run Optuna study
study = optuna.create_study(
    direction="maximize",
    study_name="XGB_Optimization",
    storage="sqlite:///xgb_optimization.db",
    load_if_exists=True
)

study.optimize(objective, n_trials=50, show_progress_bar=True)


# Print Best Results
print("\n\nBest Trial:")
print(f"AUC Score: {study.best_value:.4f}")
print("Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")



xgb_best_params = study.best_params


xgb_best_params


xgb_best_params.update(
    {
        "random_state": 42,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "gpu_hist",   # GPU acceleration
        "n_jobs": -1
    }
)


xgb_best_params


from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

model = XGBClassifier(**xgb_best_params)

## define stratified k fold 
N_SPLITS = 5

score_list = []
test_pred_proba = np.zeros(test.shape[0])
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)
for fold, (train_idx, test_idx) in enumerate(skf.split(X,y), 1):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]

    X_test_fold = X.iloc[test_idx]
    y_test_fold = y.iloc[test_idx]

    ## train the model 
    model.fit(X_train_fold, y_train_fold)
    ## make prediction and evaluate 
    y_test_fold_pred = model.predict_proba(X_test_fold)[:, 1]

    score = roc_auc_score(y_test_fold, y_test_fold_pred)
    score_list.append(score)
    print(f"Fold {fold}: ROC AUC Score: {score}")

    test_pred_proba+=model.predict_proba(test)[:, 1]

print(f"\nAvg ROC AUC Score: {np.mean(score_list)}")
## dividing the test prediction by num of folds 
xgb_predicted_proba = test_pred_proba/N_SPLITS

## save to sample submission file 
sample_submission['loan_paid_back'] = xgb_predicted_proba
print("First 5 rows of submission file...")
display(sample_submission.head())
print()

##save submission file
file_name = "xgb_optuna_prediction"
sample_submission.to_csv(f"{file_name}.csv", index=False)
print(f"File saved as {file_name}.csv")


lgbm_best_params = {
'n_estimators': 794,
 'max_depth': 12,
 'learning_rate': 0.06641860545446504,
 'num_leaves': 64,
 'min_child_samples': 83,
 'subsample': 0.6742590272473401,
 'colsample_bytree': 0.5745092310873625,
 'reg_alpha': 6.687623152245592,
 'reg_lambda': 3.830433485650677,
 'random_state': 42,
 'device_type': 'gpu',
 'verbosity': -1
}


from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

model = LGBMClassifier(**lgbm_best_params)

## define stratified k fold 
N_SPLITS = 5

score_list = []
test_pred_proba = np.zeros(test.shape[0])
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)
for fold, (train_idx, test_idx) in enumerate(skf.split(X,y), 1):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]

    X_test_fold = X.iloc[test_idx]
    y_test_fold = y.iloc[test_idx]

    ## train the model 
    model.fit(X_train_fold, y_train_fold)
    ## make prediction and evaluate 
    y_test_fold_pred = model.predict_proba(X_test_fold)[:, 1]

    score = roc_auc_score(y_test_fold, y_test_fold_pred)
    score_list.append(score)
    print(f"Fold {fold}: ROC AUC Score: {score}")

    test_pred_proba+=model.predict_proba(test)[:, 1]

print(f"Avg ROC AUC Score: {np.mean(score_list)}")
## dividing the test prediction by num of folds 
lgbm_predicted_proba = test_pred_proba/N_SPLITS

## save to sample submission file 
sample_submission['loan_paid_back'] = lgbm_predicted_proba
print("First 5 rows of submission file...")
display(sample_submission.head())
print()

##save submission file
file_name = "lgbm_optuna_prediction"
sample_submission.to_csv(f"{file_name}.csv", index=False)
print(f"File saved as {file_name}.csv")





catboost_best_params = {
    'iterations': 1430,
 'depth': 4,
 'learning_rate': 0.09469411396597646,
 'l2_leaf_reg': 12.131061172425502,
 'border_count': 238,
 'random_strength': 4.782420988806461,
 'bagging_temperature': 0.2452734673108604,
 'grow_policy': 'SymmetricTree',
 'random_state': 42,
 'eval_metric': 'AUC',
 'task_type': 'GPU',
 'verbose': 0
}


from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

model = CatBoostClassifier(**catboost_best_params)

## define stratified k fold 
N_SPLITS = 5

score_list = []
test_pred_proba = np.zeros(test.shape[0])
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)
for fold, (train_idx, test_idx) in enumerate(skf.split(X,y), 1):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]

    X_test_fold = X.iloc[test_idx]
    y_test_fold = y.iloc[test_idx]

    ## train the model 
    model.fit(X_train_fold, y_train_fold)
    ## make prediction and evaluate 
    y_test_fold_pred = model.predict_proba(X_test_fold)[:, 1]

    score = roc_auc_score(y_test_fold, y_test_fold_pred)
    score_list.append(score)
    print(f"Fold {fold}: ROC AUC Score: {score}")

    test_pred_proba+=model.predict_proba(test)[:, 1]

print(f"Avg ROC AUC Score: {np.mean(score_list)}")
## dividing the test prediction by num of folds 
catboost_predicted_proba = test_pred_proba/N_SPLITS

## save to sample submission file 
sample_submission['loan_paid_back'] = catboost_predicted_proba
print("First 5 rows of submission file...")
display(sample_submission.head())
print()

##save submission file
file_name = "catboost_optuna_prediction"
sample_submission.to_csv(f"{file_name}.csv", index=False)
print(f"File saved as {file_name}.csv")



average_blending_predicted_proba = (xgb_predicted_proba + lgbm_predicted_proba + catboost_predicted_proba)/3
sample_submission["loan_paid_back"] = average_blending_predicted_proba
sample_submission.to_csv("avg_blended_submission.csv", index=False)


sample_submission.head()


weighted_blending_predicted_proba = (
    0.40 * lgbm_predicted_proba +
    0.20 * xgb_predicted_proba +
    0.40 * catboost_predicted_proba
)

sample_submission["loan_paid_back"] = weighted_blending_predicted_proba
sample_submission.head()
sample_submission.to_csv("weighted_blended_submission.csv", index=False)


sample_submission.head()




