from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load train test data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')



# Separate features and target
X = train_data.drop(['Price', 'id'], axis=1)
y = train_data['Price']

# Handle missing values
numeric_features = ['Compartments', 'Weight Capacity (kg)']
numeric_imputer = SimpleImputer(strategy='mean')
X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])

categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])



# One-hot encode categorical variables
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_categorical_data = encoder.fit_transform(X[categorical_features])
encoded_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_features))

# Combine encoded and numeric features
X = X.drop(categorical_features, axis=1)
X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)



# Feature engineering
X['Price_per_kg'] = y / X['Weight Capacity (kg)']
X['Density'] = X['Weight Capacity (kg)'] / (X['Compartments'] + 1)

premium_brands_columns = [col for col in encoded_df.columns if 'Brand_' in col]
X['Premium_Brand'] = X[premium_brands_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

X['Feature_Score'] = (X['Laptop Compartment_Yes'] == 1).astype(int) + \
                     (X['Waterproof_Yes'] == 1).astype(int) + \
                     (X['Style_Backpack'] == 1).astype(int)

brand_price_mean = train_data.groupby('Brand')['Price'].mean()

def calculate_brand_price_ratio(row):
    active_brand_column = [col for col in row.index if col.startswith('Brand_') and row[col] == 1]
    if active_brand_column:
        active_brand = active_brand_column[0].replace('Brand_', '')
        if active_brand in brand_price_mean.index:
            return brand_price_mean[active_brand] / row['Price_per_kg']
    return np.nan

X['Brand_Price_Ratio'] = X.apply(calculate_brand_price_ratio, axis=1)
X['Brand_Price_Ratio'] = X['Brand_Price_Ratio'].fillna(X['Brand_Price_Ratio'].mean())



# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)



# Predict and evaluate
y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("Gradient Boosting Results:")
print(f"  Mean Squared Error: {mse}")
print(f"  Mean Absolute Error: {mae}")
print(f"  R-squared: {r2}")



# Actual vs predicted
plt.figure(figsize=(8,6))
plt.scatter(y_val, y_pred, color='blue', edgecolor='black', alpha=0.5)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.grid(True)
plt.show()

# Residual distribution
residuals = y_val - y_pred
plt.figure(figsize=(10,6))
plt.hist(residuals, bins=30, edgecolor='black', color='lightcoral')
plt.title('Residuals Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



# Metrics chart
metrics = ['MSE', 'R-squared']
values = [mse, r2]
plt.figure(figsize=(8,6))
plt.bar(metrics, values, color=['blue', 'green'])
plt.title('Model Performance')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Feature importance plot
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(12,6))
plt.barh(features, importances, color='purple')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(True)
plt.show()



# Prepare test data
X_test_data = test_data.drop(['id'], axis=1)
X_test_data[numeric_features] = numeric_imputer.transform(X_test_data[numeric_features])
X_test_data[categorical_features] = categorical_imputer.transform(X_test_data[categorical_features])

encoded_test_data = encoder.transform(X_test_data[categorical_features])
encoded_test_df = pd.DataFrame(encoded_test_data, columns=encoder.get_feature_names_out(categorical_features))

X_test_data = X_test_data.drop(categorical_features, axis=1)
X_test_data = pd.concat([X_test_data.reset_index(drop=True), encoded_test_df.reset_index(drop=True)], axis=1)

# Feature engineering on test set
X_test_data['Price_per_kg'] = X_test_data['Weight Capacity (kg)'] / X_test_data['Weight Capacity (kg)']
X_test_data['Density'] = X_test_data['Weight Capacity (kg)'] / (X_test_data['Compartments'] + 1)
X_test_data['Premium_Brand'] = X_test_data[premium_brands_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
X_test_data['Feature_Score'] = (X_test_data['Laptop Compartment_Yes'] == 1).astype(int) + \
                                (X_test_data['Waterproof_Yes'] == 1).astype(int) + \
                                (X_test_data['Style_Backpack'] == 1).astype(int)
X_test_data['Brand_Price_Ratio'] = X_test_data.apply(calculate_brand_price_ratio, axis=1)
X_test_data['Brand_Price_Ratio'] = X_test_data['Brand_Price_Ratio'].fillna(X_test_data['Brand_Price_Ratio'].mean())

# Predict and save
predictions = model.predict(X_test_data)
submission = test_data[['id']].copy()
submission['Price'] = predictions
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("Your submission was successfully saved!")


