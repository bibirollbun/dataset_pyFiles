import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")

train_full = pd.concat([train, train_extra], ignore_index=True)

X = train_full.drop(columns=["Price"])
y = train_full["Price"]
X_test = test.copy()


X["Material_Size"] = X["Material"] + "_" + X["Size"].astype(str)
X_test["Material_Size"] = X_test["Material"] + "_" + X_test["Size"].astype(str)


categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()


numeric_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])


model = HistGradientBoostingRegressor(random_state=42, max_iter=300, learning_rate=0.05)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_valid)
mae = mean_absolute_error(y_valid, y_pred)
print(f"Validation MAE: {mae:.2f}")


final_preds = pipeline.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],
    "price": final_preds
})

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)


print(f"âœ… Submission saved at: {submission_path}")
print("ðŸ“‚ Files in working directory:", os.listdir("/kaggle/working/"))


