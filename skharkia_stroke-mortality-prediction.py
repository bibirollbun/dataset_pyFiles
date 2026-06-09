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


# Import Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb


# Load the train & test data set
train_data = pd.read_csv("/kaggle/input/stroke-trial-prediction/Train1.csv", encoding = "latin1")
test_data = pd.read_csv("/kaggle/input/stroke-trial-prediction/Test1.csv", encoding = "latin1")


# Handle target variable
train_data["PatientDied"] = train_data["DIED"].map({1: "Y", 0: "N"})
y = train_data["PatientDied"].map({"Y": 1, "N": 0})


# Exclude the columns that are directly related to the patient death or leak future information
leak_keywords = ["DEAD", "FDEAD", "DALIVE", "DALIVED", "PatientDied", "DIED"]
leak_cols = [col for col in train_data.columns if any (k in col for k in leak_keywords)]


# Drop the "excluded columns" from the train & test dataset
train_features = train_data.drop(columns = leak_cols + ["ID"])
test_features = test_data.drop (columns = leak_cols + ["ID"], errors = "ignore")


# Matching common columns between train & test dataset
common_cols = list(set(train_features.columns).intersection(set(test_features.columns)))
X_train = train_features[common_cols]
X_test = test_features[common_cols]


# Storing test ids for submission
test_ids = test_data["ID"]


# Identifying Categorical columns
cat_cols = X_train.select_dtypes(include = "object").columns.tolist()


# Fill missing and encode categoricals
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    full_data = pd.concat([X_train[col].fillna('Unknown'), X_test[col].fillna('Unknown')])
    le.fit(full_data)
    X_train[col] = le.transform(X_train[col].fillna('Unknown'))
    X_test[col] = le.transform(X_test[col].fillna('Unknown'))
    le_dict[col] = le


# Fill numeric columns
X_train = X_train.fillna(X_train.median(numeric_only = True))
X_test = X_test.fillna(X_test.median(numeric_only = True))


# Split the training data 
X_tr,X_val, y_tr,y_val = train_test_split(X_train, y, stratify = y, test_size = 0.03, random_state = 42)


# train xgboost
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.09,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_tr, y_tr)
val_preds = model.predict(X_val)

# Pring Validation Accuracy
print(f"Validation accuracy:{accuracy_score(y_val, val_preds):.4f}")


# Run the model on full train dataset
model.fit(X_train, y)

# Predict on test dataset
predictions = model.predict(X_test)


# Prepare submission file in csv
submission = pd.DataFrame({"ID": test_ids, "PatientDied": predictions})
submission["PatientDied"] = submission["PatientDied"].map({1: "Y", 0: "N"})

submission.to_csv("submission.csv", index = False)

