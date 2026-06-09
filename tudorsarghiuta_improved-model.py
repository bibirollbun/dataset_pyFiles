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


import numpy as np
import pandas as pd
import os


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


train_data.head()


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


train_data["Has_Laptop"] = train_data["Laptop Compartment"].map({"Yes": 1, "No": 0})
test_data["Has_Laptop"] = test_data["Laptop Compartment"].map({"Yes": 1, "No": 0})


y = train_data["Price"]

features = ["Brand", "Material", "Size", "Compartments", 
    "Has_Laptop", "Waterproof", "Style", "Color",
    "Weight Capacity (kg)"
]
X = train_data[features]
X_test = test_data[features]


categorical = ["Brand", "Material", "Size", "Waterproof", "Style", "Color"]
numerical = ["Compartments", "Weight Capacity (kg)", "Has_Laptop"]


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                          ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

preprocessor = ColumnTransformer(transformers=[("cat", categorical_transformer, categorical),
    ("num", numerical_transformer, numerical)])


model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)
pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_valid)
rmse = mean_squared_error(y_valid, preds, squared=False)
print(f"Validation RMSE: {rmse:.2f}")


pipeline.fit(X, y)
predictions = pipeline.predict(X_test)
submission = pd.DataFrame({"id": test_data["id"], "Price": predictions})
submission.to_csv("improved_submission.csv", index=False)

