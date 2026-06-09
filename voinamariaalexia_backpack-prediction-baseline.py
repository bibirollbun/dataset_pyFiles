import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')

X = train_data.drop(columns=["id", "Price"])
y = train_data["Price"]

numeric_features = ['Compartments', 'Weight Capacity (kg)']
categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

numeric_imputer = SimpleImputer(strategy='mean')
X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])

categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_categorical = encoder.fit_transform(X[categorical_features])
encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))

X = X.drop(columns=categorical_features)
X = pd.concat([X.reset_index(drop=True), encoded_df], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"RMSE auf Validierungsdaten (Linear Regression): {rmse:.4f}")

X_test = test_data.drop(columns=["id"])
X_test[numeric_features] = numeric_imputer.transform(X_test[numeric_features])
X_test[categorical_features] = categorical_imputer.transform(X_test[categorical_features])

encoded_test = encoder.transform(X_test[categorical_features])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_features))

X_test = X_test.drop(columns=categorical_features)
X_test = pd.concat([X_test.reset_index(drop=True), encoded_test_df], axis=1)

predictions = model.predict(X_test)
submission = sample_submission.copy()
submission["Price"] = predictions
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("Die Datei 'submission.csv' wurde erfolgreich gespeichert.")


