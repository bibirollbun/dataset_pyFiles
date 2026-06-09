import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns


df1 = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
df2 = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df = pd.concat([df1, df2])


df.head()


for col in df.columns.tolist():
    df[col] = df[col].fillna(df[col].mode())


def feature_engineer(df):
    df["is_affordable"] = (df["Brand"] == "Jansport").astype(int)
    df["is_expensive_material"] = df["Material"].isin(["Leather", "Polyester"]).astype(int)
    df["has_laptop_compartment"] = df["Laptop Compartment"].map({"Yes": 1, "No": 0})
    df["is_waterproof"] = df["Waterproof"].map({"Yes": 1, "No": 0})
    df["utility_score"] = df["Compartments"] + df["has_laptop_compartment"]
    df["size_encoded"] = df["Size"].map({"Large": 3, "Medium": 2, "Small": 1})
    return df


df = feature_engineer(df)


features = df.columns.tolist()
features.remove("id")
features.remove("Price")
X = df[features]
y = df.Price


X = pd.get_dummies(X, dtype=float)


y = y.fillna(y.mode())


to_be_normalized = ["Weight Capacity (kg)", "Compartments"]
max_vals = [X[col].max() for col in to_be_normalized]
min_vals = [X[col].min() for col in to_be_normalized]
for i, col in enumerate(to_be_normalized):
    X[col] = (X[col] - min_vals[i]) / (max_vals[i] - min_vals[i])


X_train, X_inter, y_train, y_inter = train_test_split(X, y, test_size=0.3)
X_valid, X_cv, y_valid, y_cv = train_test_split(X_inter, y_inter, test_size=0.5)


model = xgb.XGBRegressor(n_estimators=10_000, early_stopping_rounds=5, max_depth=4, colsample_bytree=0.7, learning_rate=0.1, subsample=0.7)


model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])


X_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
id_col = X_test.id
X_test = feature_engineer(X_test)
X_test = X_test[features]


X_test = pd.get_dummies(X_test, dtype=float)


for col in X_test.columns.tolist():
    X_test[col] = X_test[col].fillna(X_test[col].mode())


max_vals = [X_test[col].max() for col in to_be_normalized]
min_vals = [X_test[col].min() for col in to_be_normalized]
for i, col in enumerate(to_be_normalized):
    X_test[col] = (X_test[col] - min_vals[i]) / (max_vals[i] - min_vals[i])


preds = model.predict(X_test)


output = pd.DataFrame({
    "id": id_col,
    "Price": preds
})
output.to_csv("submission.csv", index=False)


cv_score = np.sqrt(mean_squared_error(model.predict(X_cv), y_cv))


cv_score

