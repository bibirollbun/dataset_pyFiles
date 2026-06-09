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


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# loading data
train_df = pd.read_csv("../input/playground-series-s4e6/train.csv")
test_df = pd.read_csv("../input/playground-series-s4e6/test.csv")



# creating performance ratio: approved units / enrolled units
train_df["performance_ratio"] = (train_df["Curricular units 1st sem (approved)"] + 
                                 train_df["Curricular units 2nd sem (approved)"]) / (
                                 train_df["Curricular units 1st sem (enrolled)"] + 
                                 train_df["Curricular units 2nd sem (enrolled)"])

# replacing infinite values and handling division by zero
train_df["performance_ratio"].replace([np.inf, -np.inf], 0, inplace=True)
train_df["performance_ratio"].fillna(0, inplace=True)

# creating economic pressure index: combining macroeconomic factors
train_df["economic_pressure"] = train_df["Unemployment rate"] + train_df["Inflation rate"] - train_df["GDP"]

# applying the same fix to test set
test_df["performance_ratio"] = (test_df["Curricular units 1st sem (approved)"] + 
                                test_df["Curricular units 2nd sem (approved)"]) / (
                                test_df["Curricular units 1st sem (enrolled)"] + 
                                test_df["Curricular units 2nd sem (enrolled)"])

test_df["performance_ratio"].replace([np.inf, -np.inf], 0, inplace=True)
test_df["performance_ratio"].fillna(0, inplace=True)

test_df["economic_pressure"] = test_df["Unemployment rate"] + test_df["Inflation rate"] - test_df["GDP"]



# defining target variable
target = "Target"

# separating features and target
X = train_df.drop(columns=["id", target])
y = train_df[target]

# selecting features (removing id and target from test set too)
X_test = test_df.drop(columns=["id"])



# identifying categorical and numerical features
categorical_features = ["Marital status", "Application mode", "Course", "Daytime/evening attendance",
                        "Previous qualification", "Nacionality", "Mother's qualification", "Father's qualification",
                        "Mother's occupation", "Father's occupation", "Gender", "Scholarship holder", "International"]

numerical_features = ["Application order", "Previous qualification (grade)", "Admission grade",
                      "Age at enrollment", "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
                      "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
                      "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
                      "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
                      "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
                      "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
                      "Unemployment rate", "Inflation rate", "GDP", "performance_ratio", "economic_pressure"]

# creating preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])


# initializing classifier 
model = RandomForestClassifier(n_estimators=100, random_state=42)  # DecisionTreeClassifier(random_state=42)

# creating full pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])



# splitting training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# fitting the model
pipeline.fit(X_train, y_train)

# evaluating the model
y_pred = pipeline.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))


# making predictions
test_predictions = pipeline.predict(X_test)

# preparing submission file
submission = pd.DataFrame({"id": test_df["id"], "Target": test_predictions})

# saving submission file
submission.to_csv("submission.csv", index=False)

# displaying first few rows of submission
submission.head()

