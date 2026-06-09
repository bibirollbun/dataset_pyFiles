import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt



train = pd.read_csv('/kaggle/input/playground-series-s4e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e4/test.csv')


# Drop the id column as it's not useful for modeling
train = train.drop(columns=['id'])
X_test = test.drop(columns=['id'])


# Separate features and target variable
X = train.drop(columns=['Rings'])
y = train['Rings']


# Preprocess categorical and numerical columns manually
categorical_features = ['Sex']
numerical_features = X.select_dtypes(include=['float64']).columns.tolist()


# One-hot encode categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_categorical_encoded = encoder.fit_transform(X[categorical_features])
X_test_categorical_encoded = encoder.fit_transform(X_test[categorical_features])


# Standardize numerical features
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X[numerical_features])
X_test_numerical_scaled = scaler.fit_transform(X_test[numerical_features])


# Combine preprocessed features
X_preprocessed = np.hstack((X_numerical_scaled, X_categorical_encoded))
X_test_preprocessed = np.hstack((X_test_numerical_scaled, X_test_categorical_encoded))


X_test_preprocessed


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)


y_train_log = np.log1p(y_train)


# Lasso Regression Model
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train_log)


# Make predictions and evaluate performance
y_pred_log = lasso.predict(X_test)
y_pred_lasso = np.expm1(y_pred_log)
y_test_pred_log = lasso.predict(X_test_preprocessed)
y_test_pred_lasso = np.expm1(y_test_pred_log)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)


print("Lasso Regression Results:")
print(f"Mean Squared Error: {lasso_mse}")
print(f"R^2 Score: {lasso_r2}")
print(f"Selected Alpha: {lasso.alpha_}")
print()


# Get the feature names
numerical_feature_names = numerical_features
categorical_feature_names = encoder.get_feature_names_out(categorical_features)
all_feature_names = numerical_feature_names + list(categorical_feature_names)

# Plot Lasso Coefficients with Feature Names
lasso_coefficients = lasso.coef_

plt.figure(figsize=(12, 6))
plt.bar(all_feature_names, lasso_coefficients, color='skyblue')
plt.title("Lasso Regression Coefficients")
plt.xlabel("Feature Names")
plt.ylabel("Coefficient Value")
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xticks(rotation=45, ha='right')  # Rotate feature names for better visibility
plt.tight_layout()
plt.show()


# Predicted vs. Actual Values Visualization for Lasso
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lasso, alpha=0.5, color='blue')
plt.title("Lasso Regression: Predicted vs. Actual")
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()


submission = pd.DataFrame({'id':test['id'], 'Rings':y_test_pred_lasso})


submission.describe()


submission.to_csv('submission.csv', index=False)




