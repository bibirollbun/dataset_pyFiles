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


plt.figure(figsize=(8, 5))
sns.histplot(train["Price"], kde=True, bins=40)
plt.title("Distribution of Backpack Prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(data=train, x="Size", y="Price")
plt.title("Price Distribution by Backpack Size")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()


plt.figure(figsize=(10, 5))
brand_means = train.groupby("Brand")["Price"].mean().sort_values(ascending=False)
sns.barplot(x=brand_means.index, y=brand_means.values)
plt.xticks(rotation=45)
plt.title("Average Backpack Price by Brand")
plt.xlabel("Brand")
plt.ylabel("Average Price")
plt.show()



train["Weight Capacity (kg)"] = train["Weight Capacity (kg)"].fillna(train["Weight Capacity (kg)"].median())
test["Weight Capacity (kg)"] = test["Weight Capacity (kg)"].fillna(train["Weight Capacity (kg)"].median())

cat_columns = ["Brand", "Material", "Size", "Laptop Compartment", 
               "Waterproof", "Style", "Color"]

encoders = {}

for col in cat_columns:
    train[col] = train[col].fillna("None")
    test[col] = test[col].fillna("None")
    
    le = LabelEncoder()
    le.fit(list(train[col].astype(str)) + list(test[col].astype(str)))
    
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    
    encoders[col] = le

X = train.drop(columns=["id", "Price"])
y = train["Price"]
X_test = test.drop(columns=["id"])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101)


model = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=101)
model.fit(X_train, y_train)


val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
score = round(rmse, 2)
print(f"Baseline RMSE: {score}")



test_preds = model.predict(X_test)
submission["Price"] = test_preds
submission.to_csv("baseline_submission_custom.csv", index=False)
print("Your submission was successfully saved!")

