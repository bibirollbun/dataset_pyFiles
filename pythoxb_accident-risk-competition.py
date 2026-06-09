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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Load training data
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
X = train.drop(["accident_risk"], axis=1)
y = train["accident_risk"]
X.head()


cat_features = ["road_type","lighting","weather","time_of_day",
                "holiday","school_season","road_signs_present","public_road"]
num_features = ["num_lanes","curvature","speed_limit","num_reported_accidents"]



# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)


# Model (baseline RF)
import lightgbm as lgb
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", lgb_model)
])



# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)




# Evaluate
y_pred = pipeline.predict(X_val)
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))
print("R²:", r2_score(y_val, y_pred))



# Load test data
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

# Predict
preds = pipeline.predict(test)

# Format submission
submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": preds
})

# Round if needed (optional)
submission["accident_risk"] = submission["accident_risk"].round(3)

submission.to_csv("submission.csv", index=False)
print(submission.head())





