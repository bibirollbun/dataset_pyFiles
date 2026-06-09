# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from IPython.core.display import HTML

# Center all plots
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")

# Set a consistent default figure size
plt.rcParams['figure.figsize'] = (6, 4)  # width=6, height=4 inches


# Load train and test data
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


# Check data structure
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("\nFirst 5 rows of train data:")
display(train.head())


import warnings
warnings.filterwarnings("ignore")

# Target Distribution
plt.figure()
sns.histplot(train['accident_risk'], kde=True, bins=50)
plt.title('Distribution of Accident Risk')
plt.xlabel('Accident Risk')
plt.show()



import warnings
warnings.filterwarnings("ignore")

# Correlation Heatmap
plt.figure()
numeric_cols = train.select_dtypes(include=[np.number]).columns
correlation_matrix = train[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# Define features and target
X = train.drop(['id', 'accident_risk'], axis=1)
y = train['accident_risk']


# Identify categorical features
categorical_features = ['road_type', 'lighting', 'weather', 'road_signs_present', 
                       'public_road', 'time_of_day', 'holiday', 'school_season']

# Convert boolean columns to strings for CatBoost
for col in ['road_signs_present', 'public_road', 'holiday', 'school_season']:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)


# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Categorical features: {categorical_features}")


# Initialize CatBoost Regressor
catboost_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=categorical_features,
    random_seed=42,
    verbose=200,
    early_stopping_rounds=50,
    loss_function='RMSE'
)


# Train the model
catboost_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    verbose=200
)


# Make predictions on validation set
y_pred = catboost_model.predict(X_val)

# Calculate RMSE
val_rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"Validation RMSE: {val_rmse:.4f}")


# Cross-validation
cv_scores = cross_val_score(catboost_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print(f"Cross-validation RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")


# Feature Importance
feature_importance = catboost_model.get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))


# Detailed feature importance plot
plt.figure()
sns.barplot(data=importance_df.head(10), x='importance', y='feature', palette='viridis')
plt.title('Top 10 Most Important Features')
plt.xlabel('Importance Score')
plt.show()


# Plot actual vs predicted values
plt.scatter(y_val, y_pred, alpha=0.1, marker='.', s=5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()


# Prepare test data (excluding the id column for prediction)
test_features = test.drop('id', axis=1)


# Make predictions on test set
test_predictions = catboost_model.predict(test_features)


# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': test_predictions
})

# Save submission file
submission.to_csv('submission.csv', index=False)

print(f"Submission shape: {submission.shape}")
print("\nFirst 5 rows of submission:")
display(submission.head())


# Save your trained model for Stack Overflow Code Challenge #10 
# https://stackoverflow.com/beta/challenges/79780240/challenge-10-road-safety-challenge-joint-with-kaggle
catboost_model.save_model('accident_risk_model.cbm')

# Verify
import os
print(f"Model file size: {os.path.getsize('accident_risk_model.cbm') / (1024*1024):.2f} MB")

