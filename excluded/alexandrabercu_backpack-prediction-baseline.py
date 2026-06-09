
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder




train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

train = train.dropna(subset=["Price"])

X = train.drop(columns=["id", "Price"])
y = train["Price"]




categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

label_encoders = {}
X_encoded = X.copy()
for col in categorical_features:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le


X_encoded[numerical_features] = X_encoded[numerical_features].fillna(X_encoded[numerical_features].mean())




X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)



from lightgbm import LGBMRegressor, early_stopping, log_evaluation

params = {
    "objective": "regression",
    "metric": "rmse",
    "random_state": 42
}

model = LGBMRegressor(**params, n_estimators=1000)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
)





y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")




X_test = test.drop(columns=["id"])
for col in categorical_features:
    X_test[col] = X_test[col].astype(str)
    X_test[col] = label_encoders[col].transform(X_test[col])
X_test[numerical_features] = X_test[numerical_features].fillna(X_test[numerical_features].mean())




test_preds = model.predict(X_test)

submission = sample_submission.copy()
submission["Price"] = test_preds
submission.to_csv("baseline_submission.csv", index=False)
submission.head()


