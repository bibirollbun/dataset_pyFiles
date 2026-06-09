import numpy as np 
import pandas as pd 

train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_data.head()


test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
test_data.head()


y = train_data["Price"]

features = ["Brand","Material","Size","Compartments","Laptop Compartment","Waterproof"]  #Feature Selection
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
model.fit(X,y)
predictions = model.predict(X_test)

output = pd.DataFrame({'id': test_data.id, 'Price': predictions})
output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")

