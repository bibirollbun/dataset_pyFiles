import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
train_data.head()

test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
test_data.head()

y = train_data["Price"]

features = ["Brand", "Material", "Size", "Compartments", "Laptop Compartment", "Waterproof", "Style", "Weight Capacity (kg)"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

imputer = SimpleImputer(strategy='most_frequent')  
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test)

model = RandomForestRegressor(n_estimators=100, max_depth=9, random_state=1)
model.fit(X_imputed, y)
predictions = model.predict(X_test_imputed)


output = pd.DataFrame({"id" : test_data["id"], "Price" : predictions})
output.to_csv("submission.csv", index=False)   



