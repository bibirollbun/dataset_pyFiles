import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

train = train.dropna(subset=["Price"])

X = train.drop(columns=["id", "Price"])
y = train["Price"]
y


X_test = test.drop(columns=["id"])

print(f"Training shape: {X.shape}")
print(f"Test shape: {X_test.shape}")


# Encode categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]], ignore_index=True)
    le.fit(combined.fillna('Missing'))
    
    X[col] = le.transform(X[col].fillna('Missing'))
    X_test[col] = le.transform(X_test[col].fillna('Missing'))

# Fill remaining Nan values
X = X.fillna(0)
X_test = X_test.fillna(0)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
val_predictions = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"Validation RMSE: {rmse:.2f}")


predictions = model.predict(X_test)

submission = sample_submission.copy()
submission['Price'] = predictions

submission.to_csv('submission.csv', index=False)
submission.head(10)




