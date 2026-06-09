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

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

from lightgbm import LGBMRegressor

# Load datasets
train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

# Add feature: Weight_per_Compartment
train_data["Weight_per_Compartment"] = train_data["Weight Capacity (kg)"] / (train_data["Compartments"] + 1)
test_data["Weight_per_Compartment"] = test_data["Weight Capacity (kg)"] / (test_data["Compartments"] + 1)

# Target
y = train_data["Price"]

# Feature list
features = [
    "Brand",
    "Material",
    "Size",
    "Compartments",
    "Laptop Compartment",
    "Waterproof",
    "Style",
    "Color",
    "Weight Capacity (kg)",
    "Weight_per_Compartment"
]

X = train_data[features]
X_test_final = test_data[features]

# Feature types
categorical = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical = ["Compartments", "Weight Capacity (kg)", "Weight_per_Compartment"]

# Preprocessing pipelines
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_transformer, categorical),
    ("num", numerical_transformer, numerical)
])

# LightGBM model
model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)

# Fit and evaluate
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_valid)
mae = mean_absolute_error(y_valid, preds)
print(f"Validation MAE: {mae:.2f}")

# Fit on full data
pipeline.fit(X, y)

# Predict test set
predictions = pipeline.predict(X_test_final)

# Create submission file
submission = pd.DataFrame({
    "id": test_data["id"],
    "Price": predictions
})
submission.to_csv("lightgbm_submission.csv", index=False)
print("LightGBM submission successfully saved!")





