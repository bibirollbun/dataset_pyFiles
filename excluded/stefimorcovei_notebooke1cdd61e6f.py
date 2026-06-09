import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


X = train_data.drop(['Price', 'id'], axis=1)
y = train_data['Price']

numeric_features = ['Compartments', 'Weight Capacity (kg)']
categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

numeric_imputer = SimpleImputer(strategy='mean')
X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])

categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])


encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_categorical_data = encoder.fit_transform(X[categorical_features])
encoded_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_features), index=X.index)

X = X.drop(columns=categorical_features)
X = pd.concat([X, encoded_df], axis=1)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

mse = mean_squared_error(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print(f"ðŸ“‰ Mean Squared Error: {mse:.2f}")
print(f"ðŸ“‰ Mean Absolute Error: {mae:.2f}")
print(f"ðŸ“ˆ R-squared: {r2:.4f}")


X_test_data = test_data.drop(['id'], axis=1)

X_test_data[numeric_features] = numeric_imputer.transform(X_test_data[numeric_features])
X_test_data[categorical_features] = categorical_imputer.transform(X_test_data[categorical_features])

encoded_test_data = encoder.transform(X_test_data[categorical_features])
encoded_test_df = pd.DataFrame(encoded_test_data, columns=encoder.get_feature_names_out(categorical_features), index=X_test_data.index)

X_test_data = X_test_data.drop(columns=categorical_features)
X_test_data = pd.concat([X_test_data, encoded_test_df], axis=1)

test_preds = model.predict(X_test_data)


submission = test_data[['id']].copy()
submission['Price'] = test_preds
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("âœ… Submission file saved to /kaggle/working/submission.csv")
print(submission.head())

