import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

X = train.drop(["Price", "id"], axis=1)
y = train["Price"]

num_cols = ["Compartments", "Weight Capacity (kg)"]
cat_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]

num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoded = encoder.fit_transform(X[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

X = X.drop(columns=cat_cols).reset_index(drop=True)
X = pd.concat([X, encoded_df.reset_index(drop=True)], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE auf dem Validierungsset: {rmse:.2f}")

X_test = test.drop(["id"], axis=1)
X_test[num_cols] = num_imputer.transform(X_test[num_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
encoded_test = encoder.transform(X_test[cat_cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cat_cols))
X_test = X_test.drop(columns=cat_cols).reset_index(drop=True)
X_test = pd.concat([X_test, encoded_test_df.reset_index(drop=True)], axis=1)

preds = model.predict(X_test)
submission = test[["id"]].copy()
submission["Price"] = preds
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv gespeichert!")



