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


!pip install xgboost


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.model_selection import RandomizedSearchCV


import joblib
best_model1 = joblib.load("/kaggle/input/best-forest-cover-model6/tensorflow2/default/1/best_forest_cover_model.pkl")
print("Model loaded successfully!")


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


#Load data
train = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")
test  = pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")

# Feature engineering (same for train & test)
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Hillshade_Total"] = df["Hillshade_9am"] + df["Hillshade_Noon"] + df["Hillshade_3pm"]
    df["Hillshade_Mean"] = df["Hillshade_Total"] / 3.0

    radians = np.deg2rad(df["Aspect"])
    df["Aspect_Sin"] = np.sin(radians)
    df["Aspect_Cos"] = np.cos(radians)

    df["Hydrology_Abs_V"] = df["Vertical_Distance_To_Hydrology"].abs()
    df["Dist_Road_Fire_sum"] = (
        df["Horizontal_Distance_To_Roadways"] +
        df["Horizontal_Distance_To_Fire_Points"]
    )
    return df

train_fe = feature_engineer(train)
test_fe  = feature_engineer(test)

#Encode labels 1..7 -> 0..6 
y_raw = train_fe["Cover_Type"].astype(int)        # original labels 1..7
le = LabelEncoder()
y = le.fit_transform(y_raw)                       # encoded labels 0..6

# Features
X = train_fe.drop(columns=["Cover_Type", "Id"], errors="ignore")
feature_cols = X.columns                          # save column order

# Train/validation split for sanity check 
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    num_class=7,             # 7 classes
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42
)

best_model1.fit(X_train, y_train)
y_valid_pred_enc = best_model1.predict(X_valid)
print("Validation accuracy:", accuracy_score(y_valid, y_valid_pred_enc))

# Optional: cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print("CV scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

# Retrain on full train set for final model 
final_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    num_class=7,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42
)

best_model1.fit(X, y)

# Build X_test
X_test = test_fe[feature_cols]
print("X_test shape:", X_test.shape)

#Predict for submission 
test_pred_enc = best_model1.predict(X_test)          # 0..6
test_pred = le.inverse_transform(test_pred_enc)      # back to 1..7

print("Unique predictions:", sorted(np.unique(test_pred)))

# Build submission
submission = pd.DataFrame({
    "Id": test["Id"],
    "Cover_Type": test_pred.astype(int)
})

print(submission.head())
print("Submission unique Cover_Type:", sorted(submission["Cover_Type"].unique()))

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")

