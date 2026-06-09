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

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")



sample_submission


train


train.info()


train.describe()


test.info()


test.describe()


print(train.isnull().sum())
print(test.isnull().sum())


train.drop('id',axis=1,inplace=True)


train


test.drop('id',axis=1,inplace=True)


for col in ["job", "contact", "poutcome", "education"]:
    unknown_count = (train[col] == "unknown").sum()
    total_count = len(train[col])
    percent = (unknown_count / total_count) * 100
    print(f"{col}: {unknown_count} unknowns ({percent:.2f}%)")



for col in ["job", "contact", "poutcome", "education"]:
    unknown_count = (test[col] == "unknown").sum()
    total_count = len(test[col])
    percent = (unknown_count / total_count) * 100
    print(f"{col}: {unknown_count} unknowns ({percent:.2f}%)")



train.drop('poutcome',axis=1,inplace=True)


test.drop('poutcome',axis=1,inplace=True)


edu_mode = train["education"].mode()[0]
train["education"] = train["education"].replace("unknown", edu_mode)


edu_mode = test["education"].mode()[0]
test["education"] = test["education"].replace("unknown", edu_mode)


test


train


test


print(train.duplicated().sum())
print(test.duplicated().sum())


for col in ["job", "contact", "education"]:
    unknown_count = (train[col] == "unknown").sum()
    total_count = len(train[col])
    percent = (unknown_count / total_count) * 100
    print(f"{col}: {unknown_count} unknowns ({percent:.2f}%)")



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

x = train.drop("y",axis=1)
y = train["y"]

x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)



for col in ["job", "contact", "education"]:
    unknown_count = (test[col] == "unknown").sum()
    total_count = len(test[col])
    percent = (unknown_count / total_count) * 100
    print(f"{col}: {unknown_count} unknowns ({percent:.2f}%)")



import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import numpy as np

numeric_features = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
binary_features = ["default", "housing", "loan"]
ordinal_features = ["education", "month", "marital"]
categorical_features = ["job", "contact"]


yes_no_transformer = FunctionTransformer(
    lambda x: np.where(x == "yes", 1, 0)
)

ord_encoder = OrdinalEncoder(categories=[
    ["primary", "secondary", "tertiary", "unknown"],             
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec","unknown"],
    ["single","married","divorced","unknown"]
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num",  RobustScaler(), numeric_features),
        ("bin", yes_no_transformer, binary_features),
        ("ord", ord_encoder, ordinal_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="gini",        
        max_depth=10,            
        min_samples_split=5,      
        min_samples_leaf=2,
        class_weight="balanced", 
        random_state=42
    ))
])

pipeline.fit(x_train, y_train)

y_val_pred = pipeline.predict(x_val)   
y_test_pred = pipeline.predict(test)


submission = sample_submission.copy()
submission["y"] = y_test_pred
submission.to_csv("submission.csv",index=False)
print("✅ Submission file saved as submission.csv")

