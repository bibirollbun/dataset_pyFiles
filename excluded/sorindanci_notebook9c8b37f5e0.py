import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

y = df_train["Price"]
X = df_train.drop(["Price", "id"], axis=1)
X_test_final = df_test.drop("id", axis=1)

X = X.fillna(X.mean(numeric_only=True))
X_test_final = X_test_final.fillna(X.mean(numeric_only=True))

num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", BayesianRidge())
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("Mean Squared Error:", mean_squared_error(y_val, y_pred))
print("R² Score:", r2_score(y_val, y_pred))

test_preds = model.predict(X_test_final)

submission = pd.DataFrame({
    "id": df_test["id"],
    "Price": test_preds
})
submission.to_csv("submission_improved.csv", index=False)


