from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

# Loading the training and test datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


# Prepare features (X) and target (y)
X = train_data.drop(['Price', 'id'], axis=1)  
y = train_data['Price']


# Handle missing data
numeric_features = ['Compartments', 'Weight Capacity (kg)']
numeric_imputer = SimpleImputer(strategy='mean')
X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])

categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])

# One hot encoding categorical data
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_categorical_data = encoder.fit_transform(X[categorical_features])

encoded_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_features))
X = X.drop(categorical_features, axis=1)
X = pd.concat([X, encoded_df], axis=1)



#Feature Engineering

# 1. Price_per_kg 
X['Price_per_kg'] = y / X['Weight Capacity (kg)']

# 2. Density 
X['Density'] = X['Weight Capacity (kg)'] / (X['Compartments'] + 1)

# 3. Premium_Brand
premium_brands_columns = [col for col in encoded_df.columns if 'Brand_' in col]
premium_brands = ['Brand_Samsonite', 'Brand_Tumi', 'Brand_Rimowa']
print(f"Premium brands columns found: {premium_brands_columns}")

X['Premium_Brand'] = X[premium_brands_columns].sum(axis=1)
X['Premium_Brand'] = X['Premium_Brand'].apply(lambda x: 1 if x > 0 else 0)

# 4. Feature_Score 
X['Feature_Score'] = (X['Laptop Compartment_Yes'] == 1).astype(int) + \
                     (X['Waterproof_Yes'] == 1).astype(int) + \
                     (X['Style_Backpack'] == 1).astype(int)

# 5. Brand_Price_Ratio 
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

print("NaN values before splitting:\n", X.isna().sum())



# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r_2 = r2_score(y_true, y_pred)
    return mse, mae, r_2

#Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


mse_rf, mae_rf, r2_rf = evaluate_model(y_test, y_pred_rf)
print(f"Random Forest:")
print(f"  Mean Squared Error: {mse_rf}")
print(f"  Mean Absolute Error: {mae_rf}")
print(f"  R-squared: {r2_rf}")


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf, color='blue', edgecolor='black', alpha=0.5)
plt.title('Actual vs Predicted Prices - Random Forest')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()


residuals = y_test - y_pred_rf


plt.figure(figsize=(10,6))
plt.hist(residuals, bins=30, edgecolor='black', color='lightcoral')
plt.title('Distribution of Residuals (Errors) - Random Forest')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


metrics = ['MSE', 'R-squared']
values = [mse_rf, r2_rf]

plt.figure(figsize=(8,6))
plt.bar(metrics, values, color=['blue', 'green'])
plt.title('Model Performance - Random Forest')
plt.ylabel('Value')
plt.grid(True)
plt.show()


importances = rf_model.feature_importances_
features = X.columns

# Plot the feature importance
plt.figure(figsize=(12,6))
plt.barh(features, importances, color='purple')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(True)
plt.show()


#Preparing Test data
X_test_data = test_data.drop(['id'], axis=1)

X_test_data[numeric_features] = numeric_imputer.transform(X_test_data[numeric_features])
X_test_data[categorical_features] = categorical_imputer.transform(X_test_data[categorical_features])

encoded_test_data = encoder.transform(X_test_data[categorical_features])
encoded_test_df = pd.DataFrame(encoded_test_data, columns=encoder.get_feature_names_out(categorical_features))

X_test_data = X_test_data.drop(categorical_features, axis=1)
X_test_data = pd.concat([X_test_data, encoded_test_df], axis=1)

# Feature engineering for the test set 
X_test_data['Price_per_kg'] = X_test_data['Weight Capacity (kg)'] / X_test_data['Weight Capacity (kg)']  # You can adapt this if needed
X_test_data['Density'] = X_test_data['Weight Capacity (kg)'] / (X_test_data['Compartments'] + 1)
X_test_data['Premium_Brand'] = X_test_data[premium_brands_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
X_test_data['Feature_Score'] = (X_test_data['Laptop Compartment_Yes'] == 1).astype(int) + \
                                (X_test_data['Waterproof_Yes'] == 1).astype(int) + \
                                (X_test_data['Style_Backpack'] == 1).astype(int)
X_test_data['Brand_Price_Ratio'] = X_test_data.apply(calculate_brand_price_ratio, axis=1)
X_test_data['Brand_Price_Ratio'] = X_test_data['Brand_Price_Ratio'].fillna(X_test_data['Brand_Price_Ratio'].mean())


predictions = rf_model.predict(X_test_data)

submission = test_data[['id']].copy()  
submission['Price'] = predictions 

submission.to_csv('/kaggle/working/submission.csv', index=False)

print("Your submission was successfully saved!")




