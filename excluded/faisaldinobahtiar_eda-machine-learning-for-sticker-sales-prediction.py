import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load dataset
train_path = '/kaggle/input/playground-series-s5e1/train.csv'
test_path = '/kaggle/input/playground-series-s5e1/test.csv'
submission_path = '/kaggle/input/playground-series-s5e1/sample_submission.csv'


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


# Data Cleaning
train_df.dropna(subset=['num_sold'], inplace=True)
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])


# Feature Engineering
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['dayofweek'] = train_df['date'].dt.dayofweek
train_df['weekofyear'] = train_df['date'].dt.isocalendar().week


# Extract features for test set
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['dayofweek'] = test_df['date'].dt.dayofweek
test_df['weekofyear'] = test_df['date'].dt.isocalendar().week


# Encode categorical variables
encoder = LabelEncoder()
for col in ['country', 'store', 'product']:
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])


# Scale numerical features
scaler = StandardScaler()
numerical_features = ['year', 'month', 'day', 'dayofweek', 'weekofyear']
train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
test_df[numerical_features] = scaler.transform(test_df[numerical_features])


# Prepare training data
X = train_df[['country', 'store', 'product', 'year', 'month', 'day', 'dayofweek', 'weekofyear']]
y = train_df['num_sold']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Hyperparameter tuning for Random Forest
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_model = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, 
                                   n_iter=5, cv=3, scoring='neg_mean_absolute_error', 
                                   n_jobs=4, random_state=42)

random_search.fit(X_train, y_train)
best_rf_model = random_search.best_estimator_


# Train Additional Models
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)


ab_model = AdaBoostRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
ab_model.fit(X_train, y_train)


# Evaluate Models
y_pred_rf = best_rf_model.predict(X_val)
y_pred_gb = gb_model.predict(X_val)
y_pred_ab = ab_model.predict(X_val)


mae_rf = mean_absolute_error(y_val, y_pred_rf)
mae_gb = mean_absolute_error(y_val, y_pred_gb)
mae_ab = mean_absolute_error(y_val, y_pred_ab)


print(f'MAE (Random Forest): {mae_rf}')
print(f'MAE (Gradient Boosting): {mae_gb}')
print(f'MAE (AdaBoost): {mae_ab}')


# Select the best model
best_model = min([(best_rf_model, mae_rf), (gb_model, mae_gb), (ab_model, mae_ab)], key=lambda x: x[1])[0]


# Predict on test data
X_test = test_df[['country', 'store', 'product', 'year', 'month', 'day', 'dayofweek', 'weekofyear']]
test_df['num_sold'] = best_model.predict(X_test)


# Save predictions
submission = test_df[['id', 'num_sold']]
submission.to_csv('submission.csv', index=False)


# Visualization: Monthly Sales Trends (Fix FutureWarning)
plt.figure(figsize=(10, 6))
sns.lineplot(data=train_df, x='month', y='num_sold', hue='year', errorbar=None)
plt.title("Monthly Sales Trends")
plt.show()


# Visualization: Feature Importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': best_model.feature_importances_})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance")
plt.show()


# Additional Analysis: Sales Distribution
plt.figure(figsize=(10, 6))
sns.histplot(train_df['num_sold'], bins=30, kde=True)
plt.title("Sales Distribution")
plt.show()


# Additional Analysis: Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()

