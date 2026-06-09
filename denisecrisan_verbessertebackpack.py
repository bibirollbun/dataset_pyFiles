

import numpy as np 
import pandas as pd 



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




trainData = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
testData = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

trainData.describe()
trainData.columns


features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof',
            'Style', 'Color', 'Compartments', 'Weight Capacity (kg)']

trainData = trainData.dropna(axis=0)

y = trainData.Price
X = trainData[features]
X_test = testData[features]

X_encoded = pd.get_dummies(X)
X_test_encoded=pd.get_dummies(X_test)





from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
X_test_encoded = X_test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

train_X, val_X, train_y, val_y = train_test_split(X_encoded, y, random_state=1)

xgb_model = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42)
xgb_model.fit(train_X, train_y)
val_preds_xgb = xgb_model.predict(val_X)
mae_xgb = mean_absolute_error(val_y, val_preds_xgb)
print(f"XGBoost MAE: {mae_xgb:.2f}")



#verbessert das Modell

from xgboost import XGBRegressor
X_test = pd.get_dummies(testData[features])

model = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=1)
model.fit(X_encoded,y)
predictions = model.predict(X_test)

output = pd.DataFrame({'id': testData.id, 'Price': predictions})
print(output)
output.to_csv('submission.csv', index=False)


