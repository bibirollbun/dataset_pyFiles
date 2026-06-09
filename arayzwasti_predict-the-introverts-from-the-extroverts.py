# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest , f_regression
from sklearn.tree import DecisionTreeRegressor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_dataset = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_dataset = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


y = train_dataset["Personality"]
X = train_dataset.drop(columns=["id","Personality"])


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.2, random_state= 42)


numerical_feature = [col for col in X.columns if X[col].dtype in ['int', 'float64']]
categorical_feature = [col for col in X.columns if X[col].dtype == "object"]


numerical_imputer = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value= 0)),
        ("MMS", MinMaxScaler())
    ])
numerical_imputer


categorical_imputer = Pipeline(
    steps= [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("OHE", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]
)
print(categorical_imputer)


preprocessor = ColumnTransformer(
    transformers = [
        ("numerical_imputer", numerical_imputer, numerical_feature),
        ("categorical_feature", categorical_imputer, categorical_feature)
    ]
)
print(preprocessor)


model_selection = SelectKBest(score_func=f_regression, k=8)
print(model_selection)


MODEL = DecisionTreeRegressor(random_state = 42)


my_pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("model_selection", model_selection),
    ("MODEL", MODEL)
])


print(my_pipeline)


my_pipeline.fit(X_train, y_train)
X_final_test = test_dataset.drop(columns=["id"])
preds = my_pipeline.predict(X_final_test)


submission = pd.DataFrame({
    "id": test_dataset["id"],
    "Personality": np.where(preds >= 0.5, "Extrovert", "Introvert")
})
submission.to_csv("submission.csv", index=False)

print("submission.csv file is ready!")

