import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder



train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

train.head()



train["Weight Capacity (kg)"] = train["Weight Capacity (kg)"].fillna(train["Weight Capacity (kg)"].median())
test["Weight Capacity (kg)"] = test["Weight Capacity (kg)"].fillna(train["Weight Capacity (kg)"].median())

categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", 
                    "Waterproof", "Style", "Color"]

for col in categorical_cols:
    train[col] = train[col].fillna("Unknown")
    test[col] = test[col].fillna("Unknown")



label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le



X = train.drop(["id", "Price"], axis=1)
y = train["Price"]
X_test = test.drop(["id"], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print("Validation RMSE:", rmse)



test_preds = model.predict(X_test)
submission["Price"] = test_preds
submission.to_csv("baseline_submission.csv", index=False)


