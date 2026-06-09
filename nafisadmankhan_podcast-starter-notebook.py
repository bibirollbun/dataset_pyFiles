import pandas as pd 

train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)


X = train_df.drop(columns=["id", "Listening_Time_minutes"])
y = train_df["Listening_Time_minutes"]


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


X_test = test_df.drop(columns=["id"])
# y_test = False


# Choose Classifier
from sklearn.ensemble import HistGradientBoostingRegressor
model = HistGradientBoostingRegressor(random_state=42)


# Fit the model
model.fit(X_train, y_train)


predictions = model.predict(X_val)


import numpy as np
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_val, predictions))
print(f"Validation Root Mean Squared Error: {rmse:.4f}")


predictions = model.predict(X_test)


# Submission
submission = pd.DataFrame({
    "id": test_df["id"],
    "Listening_Time_minutes": predictions
})

submission.to_csv("submission.csv", index=False)




