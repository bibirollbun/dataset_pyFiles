import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import train_test_split
train=pd.read_csv("/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/train.csv")
test=pd.read_csv("/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/test.csv")

cat_cols=['F5','F7','F8','F9','F10','F13']
test["relationship_probability"]=-1  
combined=pd.concat([train, test], axis=0, ignore_index=True)
combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

train_encoded=combined[combined["relationship_probability"] != -1]
test_encoded=combined[combined["relationship_probability"] == -1]

X=train_encoded.drop(["relationship_probability", "ID"], axis=1)
y=train_encoded["relationship_probability"]

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_val)
mse = mean_squared_error(y_val, predictions)
rmse=np.sqrt(mse)
print("validations rmse",rmse)

test_features = test_encoded.drop(["relationship_probability", "ID"], axis=1)
test_predictions = model.predict(test_features)

submission = pd.DataFrame({
    "ID": test_encoded["ID"],
    "relationship_probability": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("submission.csv generated!")






