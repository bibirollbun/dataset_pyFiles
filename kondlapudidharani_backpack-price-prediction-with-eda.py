import numpy as np
import pandas as pd

# Set random seed for NumPy
np.random.seed(42)



# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')



train_df.shape


test_df.shape


train_df.head()


# Basic information about the dataset
train_df.info()


train_df.isnull().sum()


print(train_df['Price'].max())
print(train_df['Price'].min())


skewness = train_df['Price'].skew()
print(f"Skewness of Price: {skewness}")


train_df['Size'].value_counts()


# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


msno.matrix(train_df)


# Visualize the distribution of the target variable 'Price'
plt.figure(figsize=(10,6))
sns.histplot(train_df['Price'], kde=True, color='blue')
plt.title('Price Distribution')
plt.show()


# Bar plot of price for each brand
plt.figure(figsize=(10, 6))
sns.barplot(x='Brand', y='Price', data=train_df, palette='viridis')
plt.title('Price Comparison by Brand')
plt.xlabel('Brand')
plt.ylabel('Price')
plt.show()


# 3. Box Plot: Comparison of Material Types by Price
plt.figure(figsize=(10, 6))
sns.boxplot(x='Material', y='Price', data=train_df, palette='Set2')
plt.title('Price Comparison by Material Type')
plt.xlabel('Material')
plt.ylabel('Price')
plt.show()


# Machine learning models and tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Function for feature engineering
def create_features(df):

  
    # Define weight capacity bins
    bins = [0, 5, 10, 20, 30]  # Example weight capacity bins
    labels = ['Light', 'Medium', 'Heavy', 'Extra Heavy']  # Corresponding labels

    df['weight_capacity_category'] = pd.cut(df['Weight Capacity (kg)'], bins=bins, labels=labels)

    return df

train_df = create_features(train_df)
test_df  =create_features(test_df)



train_df.head()


train_df.info()



categorical_cols = ['Brand','Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Example: Fill missing values based on a condition or logic
def custom_imputer(series):
    # Example: If a category is missing, you could replace it with the category "Custom" or based on some logic
    return series.fillna('Custom')

for col in categorical_cols:
    train_df[col] = custom_imputer(train_df[col])
    test_df[col] = custom_imputer(test_df[col])



# Handle missing numerical values (fill with mean or drop rows)

train_df['Weight Capacity (kg)'].fillna(train_df['Weight Capacity (kg)'].mean(), inplace=True)
test_df['Weight Capacity (kg)'].fillna(test_df['Weight Capacity (kg)'].mean(), inplace=True)



train_df.isnull().sum()


premium_brands = ['Nike', 'Adidas', 'Under Armour']
train_df['Is_Premium_Brand'] = train_df['Brand'].apply(lambda x: 1 if x in premium_brands else 0)
test_df['Is_Premium_Brand'] = test_df['Brand'].apply(lambda x: 1 if x in premium_brands else 0)


# One-Hot Encoding for categorical variables
train_df = pd.get_dummies(train_df, drop_first=True)
test_df  = pd.get_dummies(test_df, drop_first=True)


X = train_df.drop(columns=['Price', 'id'])
y = train_df['Price']


# Scaling the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split the dataset into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


# Initialize the XGBoost Regressor with default parameters
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Hyperparameter tuning using GridSearchCV (You can also use RandomizedSearchCV)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'min_child_weight': [1, 3, 5]
}

# Using RandomizedSearchCV for faster hyperparameter search (GridSearchCV can be used as an alternative)
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=3, verbose=2, random_state=42)

# Fit the RandomizedSearchCV to find the best parameters
random_search.fit(X_train, y_train)

# Get the best model
best_model = random_search.best_estimator_


# Evaluate the best model
y_pred = best_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R2 Score: {r2}")

# Optionally, perform Cross-Validation for a better estimate of performance
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)  # Since cross_val_score gives negative MSE, we need to invert it

print(f"Cross-Validation RMSE: {cv_rmse.mean()} Â± {cv_rmse.std()}")

# Use Early Stopping for training to avoid overfitting
early_stopping_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, random_state=42)
early_stopping_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=False)

# Predict with the model trained using early stopping
y_pred_early_stopping = early_stopping_model.predict(X_test)

# Evaluate the early stopping model
mse_early_stopping = mean_squared_error(y_test, y_pred_early_stopping)
rmse_early_stopping = mse_early_stopping ** 0.5  # Root Mean Squared Error
r2_early_stopping = r2_score(y_test, y_pred_early_stopping)

# Print early stopping evaluation metrics
print(f"Early Stopping - MSE: {mse_early_stopping}")
print(f"Early Stopping - RMSE: {rmse_early_stopping}")
print(f"Early Stopping - R2: {r2_early_stopping}")


"""from sklearn.model_selection import cross_val_score

# Perform cross-validation with RMSE as the evaluation metric
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')

# Calculate and print the average RMSE across all folds
average_rmse = -cv_scores.mean()  # Convert negative RMSE to positive
print(f"Average RMSE from 5-fold cross-validation: {average_rmse:.4f}")
"""


# Prepare the test dataset
X_test = test_df.drop(columns=['id'])
X_test_scaled = scaler.transform(X_test)

# Predictions using the best model (Random Forest)
final_predictions = best_model.predict(X_test_scaled)

# Prepare the submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Price': final_predictions
})

# Save the submission file
submission.to_csv('submission.csv', index=False)



submission.head()

