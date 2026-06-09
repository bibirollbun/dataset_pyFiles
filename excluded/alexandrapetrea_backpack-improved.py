import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error

from lightgbm import LGBMRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
submission_template = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


train["Compart_weight_ratio"] = train["Weight Capacity (kg)"] / (train["Compartments"] + 1)
test["Compart_weight_ratio"] = test["Weight Capacity (kg)"] / (test["Compartments"] + 1)


features = [
    "Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color",
    "Compartments", "Weight Capacity (kg)", "Compart_weight_ratio"
]

X = train[features]
X_test = test[features]
y = train["Price"]

cat_feats = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
num_feats = ["Compartments", "Weight Capacity (kg)", "Compart_weight_ratio"]


cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer(transformers=[
    ("cat", cat_pipeline, cat_feats),
    ("num", num_pipeline, num_feats)
])


regressor = LGBMRegressor(
    n_estimators=250,
    learning_rate=0.07,
    max_depth=7,
    random_state=2025
)


pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", regressor)
])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)

pipeline.fit(X_train, y_train)
val_preds = pipeline.predict(X_valid)
mae = mean_absolute_error(y_valid, val_preds)
print(f"Validation MAE: {mae:.2f}")


pipeline.fit(X, y)


final_preds = pipeline.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "Price": final_preds
})

submission.to_csv("lightgbm_pipeline_submission.csv", index=False)
print("Submission saved as lightgbm_pipeline_submission.csv")

