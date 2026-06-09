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


# California Homelessness Prediction Challenge

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# ======================
# Load data
# ======================
DATA_PATH = "/kaggle/input/california-homelessness-prediction-challenge/"

train = pd.read_csv(DATA_PATH + "train.csv")
test = pd.read_csv(DATA_PATH + "test.csv")
sample_sub = pd.read_csv(DATA_PATH + "sample_submission.csv")

TARGET = "HOMELESS_RATE"
ID_COL = "ID"

X = train.drop(columns=[TARGET, ID_COL])
y = train[TARGET]

# ======================
# Preprocessing
# ======================
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ]), cat_cols)
    ],
    remainder="drop"
)

# ======================
# Model
# ======================
model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

pipe = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

# ======================
# Train model (no CV to avoid memory/save issues)
# ======================
pipe.fit(X, y)

# ======================
# Predict
# ======================
test_preds = pipe.predict(test.drop(columns=[ID_COL]))

# ======================
# SAFE save submission
# ======================
submission = pd.DataFrame({
    "ID": test[ID_COL].values,
    "HOMELESS_RATE": test_preds.astype(np.float32)
})

# Save explicitly to Kaggle working directory
output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)

print("Saved successfully at:", output_path)
print(submission.head())


