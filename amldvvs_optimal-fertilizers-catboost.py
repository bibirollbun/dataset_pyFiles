#installing and importing libraries
!pip install catboost --quiet

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


#loading data
train_path = "/kaggle/input/playground-series-s5e6/train.csv"
df = pd.read_csv(train_path)

#dropping id column
df = df.drop(columns=["id"])

#encoding target
target_col = "Fertilizer Name"
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

#saving label encoder for later use
joblib.dump(le, "fertilizer_label_encoder.pkl")

#preparing features and target
X = df.drop(columns=[target_col])
y = df[target_col]

#categorical columns
categorical_cols = ["Soil Type", "Crop Type"]

#splitting data
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#creating catboost pool
train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
valid_pool = Pool(X_valid, y_valid, cat_features=categorical_cols)

#training catboost model
model = CatBoostClassifier(
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    task_type="GPU",
    verbose=100
)

model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

#saving model
model.save_model("fertilizer_catboost.cbm")

print("Model saved as fertilizer_catboost.cbm and label encoder saved.")

