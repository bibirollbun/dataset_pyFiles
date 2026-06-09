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

#this python 3 environment comes with many helpful analytics libraries installed
#it is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
#for example, here's several helpful packages to load

import numpy as np #linear algebra
import pandas as pd #data processing, csv file i/o (e.g. pd.read_csv)

#input data files are available in the read-only "../input/" directory
#for example, running this (by clicking run or pressing shift+enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#you can write up to 20gb to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "save & run all" 
#you can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#import necessary libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#load datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s4e6/test.csv")

#feature engineering
#creating a feature: academic risk based on course failures
train_df["academic_risk"] = (train_df["Curricular units 1st sem (enrolled)"] - train_df["Curricular units 1st sem (approved)"]) + \
                            (train_df["Curricular units 2nd sem (enrolled)"] - train_df["Curricular units 2nd sem (approved)"])
test_df["academic_risk"] = (test_df["Curricular units 1st sem (enrolled)"] - test_df["Curricular units 1st sem (approved)"]) + \
                           (test_df["Curricular units 2nd sem (enrolled)"] - test_df["Curricular units 2nd sem (approved)"])

#creating a feature: study consistency as the ratio of evaluations attempted
train_df["study_consistency"] = (train_df["Curricular units 1st sem (evaluations)"] + train_df["Curricular units 2nd sem (evaluations)"]) / \
                                (train_df["Curricular units 1st sem (enrolled)"] + train_df["Curricular units 2nd sem (enrolled)"] + 0.01)
test_df["study_consistency"] = (test_df["Curricular units 1st sem (evaluations)"] + test_df["Curricular units 2nd sem (evaluations)"]) / \
                               (test_df["Curricular units 1st sem (enrolled)"] + test_df["Curricular units 2nd sem (enrolled)"] + 0.01)

#encoding target variable
label_encoder = LabelEncoder()
train_df["Target"] = label_encoder.fit_transform(train_df["Target"]) #converts 'Graduate', 'Dropout', 'Enrolled' into numbers

#separating features and target variable
X_train = train_df.drop(columns=["Target", "id"]) #dropping target and id
y_train = train_df["Target"] #target variable
X_test = test_df.drop(columns=["id"]) #dropping id column in test set

#getting feature types
numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

#defining preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="mean")), #handling missing values
    ('scaler', StandardScaler()) #scaling numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")), #handling missing values for categorical data
    ('onehot', OneHotEncoder(handle_unknown="ignore")) #encoding categorical features
])

#combining transformations 
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

#defining the model (Random Forest or Decision Tree)
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

#creating a pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])

#fitting the model
pipeline.fit(X_train, y_train)

#making predictions on test data
predictions = pipeline.predict(X_test)

#converting predictions back to original labels
predictions = label_encoder.inverse_transform(predictions)

#creating submission file
submission = pd.DataFrame({"id": test_df["id"], "academic_risk_prediction": predictions})
submission.to_csv("/kaggle/working/submission.csv", index=False)

#displaying submission file
submission.head()





