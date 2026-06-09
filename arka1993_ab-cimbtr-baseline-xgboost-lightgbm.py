# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
"""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# load the data
raw_df_train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
raw_df_test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


# differentiate the categorical and numerical cols
def identify_numerical_catergorical_columns(df):
    """
    Identify the Numerical and the Categorical Columns in the dataframe
    """
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    return numerical_cols, categorical_cols


# create data pre procesing pipeline
def get_data_transformer_object(numerical_cols, categorical_cols):
    num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
                    
                ]
            )

    cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='constant', fill_value='other')),
                    ("label_encoder", OrdinalEncoder())
                ]
            )

    preprocessor = ColumnTransformer(
                [
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols)
                ]
            )

    return preprocessor


# final data creation step
target = 'efs'
df_features = raw_df_train.drop(columns=[target, 'ID', 'efs_time'], axis = 1)
print("df_features shape : {}".format(df_features.shape))


df_label  = np.array(raw_df_train[target]).reshape(-1, 1)
print("Train Label Shape  : {}".format(df_label.shape))

numerical_cols, categorical_cols = identify_numerical_catergorical_columns(df_features)
print("----- Numerical columns -------")
print(numerical_cols)
print("----- Categorical columns -------")
print(categorical_cols)
preprocesser_obj = get_data_transformer_object(numerical_cols, categorical_cols)

processed_df_arr = preprocesser_obj.fit_transform(df_features)

X_train, X_valid, y_train, y_valid = train_test_split(processed_df_arr, df_label, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_valid.shape)


from xgboost import XGBClassifier
# create model instance
xgboost_baseline_model = XGBClassifier(
    n_estimators=100,       # Number of trees
    learning_rate=0.1,      # Step size shrinkage
    max_depth=4,            # Maximum tree depth
    objective='binary:logistic',  # For binary classification
    eval_metric='logloss',# Evaluation metric
    enable_categorical=True,
    random_state=42
)
# fit model
xgboost_baseline_model.fit(X_train, y_train)
# make predictions
preds = xgboost_baseline_model.predict(X_valid)
y_pred_proba = xgboost_baseline_model.predict_proba(X_valid)[:, 1] 
accuracy_score(y_valid, preds)


test_df = raw_df_test.drop(columns = ['ID'], axis = 1)
numerical_cols_test, categorical_cols_test = identify_numerical_catergorical_columns(df_features)
print(numerical_cols)
print(categorical_cols_test)
preprocesser_obj_test = get_data_transformer_object(numerical_cols_test, categorical_cols_test)
processed_test_arr = preprocesser_obj_test.fit_transform(test_df)
processed_test_arr.shape


"""
submission_prob = np.max(xgboost_baseline_model.predict_proba(processed_test_arr), axis = 1)
print(submission_prob)
"""


"""
submission_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
submission_df.head()
submission_df["prediction"] = submission_prob
submission_df.to_csv("/kaggle/working/submission.csv")
"""


!pip install mlflow


import lightgbm as lgb
import mlflow
mlflow.set_experiment("CIM BTR Light GBM baseline")


# create the lighgbm dataset
train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_valid, label=y_valid)


# defining the LightGBM parameters
with mlflow.start_run():
    params = {
        "objective": "binary",
        "metric": "binary_logloss",  # Use "auc" for area under the curve metric
        "boosting_type": "gbdt",    # Gradient Boosted Decision Trees
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1
    }
    for key, value in params.items():
        mlflow.log_param(key, value)
        
    
    # Train the model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets = [validation_data],
        callbacks = [
            lgb.early_stopping(stopping_rounds = 50)
        ]
    )
    
    y_pred_prob = model.predict(X_valid, num_iteration=model.best_iteration)
    y_pred = (y_pred_prob > 0.5).astype(int)
    print(y_pred)
    
    evaluation_score = accuracy_score(y_valid, y_pred)
    print("Evaluation score is {}".format(evaluation_score))

    mlflow.log_metric("accuracy", evaluation_score)
    mlflow.sklearn.log_model(model, "Light GBM baseline model")


!mlflow ui


y_pred_prob = model.predict(processed_test_arr, num_iteration=model.best_iteration)
submission_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
submission_df.head()
submission_df["prediction"] = y_pred_prob
submission_df.to_csv("/kaggle/working/submission.csv")

