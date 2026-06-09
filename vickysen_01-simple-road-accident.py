
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# load dataset
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv') 


# Getting shape of all 3 dataset 
train.shape,test.shape,sample_submission.shape


# print top 5 
train.head()


test.head()


# Gettin top 5 ros of submission file
sample_submission.head(3)


# getting number of missing values in the entire dataset
train.isnull().sum().sum()


# similarly we can find in test dataset
test.isnull().sum().sum()


train.dtypes



# Select columns with 'object' or 'category' dtype
cat_cols = train.select_dtypes(include=['object', 'category','bool']).columns
cat_cols


num_cols = train.select_dtypes(include='number').columns
num_cols


# Visualising vateogiral columns 
# Set up the figure size and grid layout

n_cols = 3  # 3 plots per row
n_rows = (len(cat_cols) + 2) // 3  # Ceiling division to ensure enough rows
plt.figure(figsize=(15, 5 * n_rows))  # Adjust height based on number of rows

# Create countplots
for i, col in enumerate(cat_cols, 1):
    plt.subplot(3, n_cols, i)
    sns.countplot(x=col, data=train, hue=col)  # palette='coolwarm')
    plt.title(f'Distribution of {col}', fontsize=12)
    plt.xlabel(col, fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.legend(loc='upper right')
    plt.xticks(rotation=45)

plt.tight_layout()

# Save and show plot
# plt.savefig('countplots.png')
plt.show()


def print_unique_values(df, columns):
    for col in columns:
        unique_vals = df[col].unique()
        print(f"Unique values in {col}: {unique_vals}")
        print('-' * 50)

# Column list
cols = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 
        'time_of_day', 'holiday', 'school_season']

# Call function
print_unique_values(train, cols)


# Calculate correlation matrix
corr_matrix = train[num_cols].corr()

# Set figure size
plt.figure(figsize=(10, 8))

# Create heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=14)
plt.xticks(rotation=60)  # Rotate x-axis labels by 45 degrees
# plt.yticks(rotation=45)  # Rotate x-axis labels by 45 degrees

# Style axis labels
plt.xlabel('Features', fontsize=13, color='blue', fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('Features', fontsize=13, color='blue', fontweight='bold', fontfamily='Times New Roman')

plt.title('Correlation Matrix of Numerical Features', fontsize=14, color='blue', fontweight='bold', fontfamily='Times New Roman')

# Adjust layout and show plot
plt.tight_layout()
plt.show()


# encode all columns at once
train_encoded = pd.get_dummies(train, columns=cat_cols, drop_first=True)


train_encoded.sample(5)


train_encoded.shape


# similarly encode test data set yeah control
cat_cols = test.select_dtypes(include=['object', 'category','bool']).columns
test_encoded = pd.get_dummies(test, columns=cat_cols, drop_first=True)
test_encoded.shape


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


train.columns


id_ =test['id']


# Separate features and target 
X = train_encoded.drop(['id', 'accident_risk'], axis=1)  
y = train_encoded['accident_risk']
X_test = test_encoded.drop(['id'], axis=1)


# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train XGBoost regression model
model = XGBRegressor(objective='reg:squarederror', random_state=42)
# model.fit(X_train, y_train)


# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],  # Number of trees
    'max_depth': [3, 5, 7],     # Maximum depth of trees
    'learning_rate': [0.01, 0.1], # Step size for updates
    'subsample': [0.8, 1.0],     # Fraction of samples used per tree
    'colsample_bytree': [0.8, 1.0] # Fraction of features used per tree
}


from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)

# Fit the model (replace X and y with your data)
grid_search.fit(X, y)


# Print best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score (RMSE):", -grid_search.best_score_)

# Use the best model from grid search
best_model = grid_search.best_estimator_

# Make predictions on test data
test_preds = best_model.predict(X_test)


# Validate model on user defined test
y_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation Mean Squared Error: {mse:.4f}")


# Prepare submission
submission = sample_submission.copy()
submission['accident_risk'] = test_preds  # Adjust 'target' to match sample_submission
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")


# Feature importance plot
plt.figure(figsize=(10, 6))
feature_imp = pd.DataFrame({'Feature': X.columns, 'Importance': best_model.feature_importances_})
feature_imp = feature_imp.sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis')
plt.title('Feature Importance', fontsize=14, color='blue', fontweight='bold', fontfamily='Times New Roman')
plt.xlabel('Importance', fontsize=12, color='blue', fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('Features', fontsize=12, color='blue', fontweight='bold', fontfamily='Times New Roman')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# after removing feature, can proceed like this




