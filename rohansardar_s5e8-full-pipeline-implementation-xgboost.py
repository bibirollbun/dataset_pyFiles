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


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


# checking for missing values
print(train.isna().sum())
print()
print(test.isna().sum())


# statistical description of the dataset
# for training
train.describe()


# for testing
test.describe()


# printing the columns of 'object' datatype
object_cols = train.select_dtypes(include="object").columns.tolist()
print(f"The object columns are: \n{object_cols}")


# printing the unique values of each object columns
for col_name in object_cols:
    print(f"{col_name}:\n{sorted(train[col_name].unique())}\n")


# importing necessary modules
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# partitioning into feature matrix X and target vector y 
X = train.drop(['id', 'y'], axis=1)
y = train['y']

# dropping id from test dataset
test.drop(['id'], axis=1, inplace=True)


# applying columntransformer to apply standard scaler to numerical values and one-hot encoding to
# categorical values
data_processor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), object_cols)
    ]
)
data_processor


# splitting the dataset for training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# obtained this hyperparameters from Optuna
best_params={
    'max_leaves': 71,
    'min_child_weight': 7.627525480669875,
    'learning_rate': 0.07600597724564652,
    'subsample': 0.9990411964287336,
    'colsample_bylevel': 0.962210548867428,
    'colsample_bytree': 0.677377783758824,
    'reg_alpha': 4.02840911563676,
    'reg_lambda': 0.6770091573319115,
    'n_estimators': 1401
}

# Fit data processor separately
processor_fitted = data_processor.fit(X_train)
# Transform train and val for XGBoost
X_train_proc = processor_fitted.transform(X_train)
X_val_proc = processor_fitted.transform(X_val)

model = XGBClassifier(
    **best_params,
    objective='binary:logistic',
    tree_method='hist',
    device="cuda",
    eval_metric="auc",
    random_state=42
)
model


# creating the pipeline to apply the transformations
pipeline = Pipeline(
    steps=[
        ('data_processor', data_processor),
        ('xgb_classifier', model)
    ])
pipeline


%%time
# fitting the pipeline which already contains the proecssing steps and model
pipeline.named_steps['xgb_classifier'].fit(
    X_train_proc, y_train,
    eval_set=[(X_val_proc, y_val)],
    early_stopping_rounds=250,
    verbose=False
)


# calculate the accuracy, roc-auc
y_pred = pipeline.predict(X_val)
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_val, y_pred)}")
print(f"Classification Report: \n{classification_report(y_val, y_pred)}")


sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
test_preds = pipeline.predict_proba(test)
test_preds_proba = test_preds[:, 1]
submission = pd.DataFrame({
    'id': sub['id'],
    'y': test_preds_proba
})
submission.to_csv('submission.csv', index=False)




