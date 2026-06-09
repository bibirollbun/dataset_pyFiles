# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# 1. Load Data
# First, we load the training and testing data. We need to use the full file paths provided by Kaggle.

path_train = '/kaggle/input/playground-series-s5e10/train.csv'
path_test = '/kaggle/input/playground-series-s5e10/test.csv'

# Loading CSVs
try:
    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)
    print(f"Training data loaded: {train_df.shape}")
    print(f"Test data loaded: {test_df.shape}")
except FileNotFoundError:
    print("ERROR: Could not find data files. Check the file paths in the 'path_train' and 'path_test' variables.")

print("\n Training Data Head")
print(train_df.head())

print("\n Training Data Info")
# this is great for checking for nulls and data types
train_df.info()


# 2. Exploratory Data Analysis (EDA)
# Let's explore the data to find patterns and understand the relationships between features and the target variable.

print("\n Starting Exploratory Data Analysis (EDA)")

# Target Variable: Accident Risk - Let's see the distribution of the value we are trying to predict.
print("Plotting Target Variable Distribution")
plt.figure(figsize=(10, 6))
sns.histplot(train_df['accident_risk'], kde=True, bins=50)
plt.title('Distribution of Accident Risk (Target Variable)')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')
plt.show()

# Numerical Feature Correlation - A heatmap shows us which numerical features are most correlated with 'accident_risk'.
print("\n Plotting Numerical Correlation Heatmap...")
# Select only numerical columns
numerical_cols = train_df.select_dtypes(include=np.number).columns
corr_matrix = train_df[numerical_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Categorical Features vs. Target - Let's see how different categories relate to accident risk.
print("\n Plotting Categorical Feature Relationships")

# Plot 1: Weather vs. Accident Risk
plt.figure(figsize=(10, 6))
sns.barplot(data=train_df, x='weather', y='accident_risk')
plt.title('Average Accident Risk by Weather Condition')
plt.xlabel('Weather')
plt.ylabel('Average Accident Risk')
plt.show()

# Plot 2: Road Type vs. Accident Risk
plt.figure(figsize=(10, 6))
sns.barplot(data=train_df, x='road_type', y='accident_risk')
plt.title('Average Accident Risk by Road Type')
plt.xlabel('Road Type')
plt.ylabel('Average Accident Risk')
plt.show()

print("EDA Complete")

# ## 3. Preprocessing and Feature Engineering
# Now we convert our data into a format the model can understand.
# - Convert boolean columns (True/False) to integers (1/0)
# - Convert categorical columns (like 'weather') to numbers using One-Hot Encoding.

print("\n Starting Preprocessing")

# Store test IDs for final submission
test_ids = test_df['id']

# Separate target variable (y) from training features (X)
y_train = train_df['accident_risk']
X_train = train_df.drop(columns=['id', 'accident_risk'])
X_test = test_df.drop(columns=['id'])

# Identify different column types
categorical_cols = X_train.select_dtypes(include=['object']).columns
boolean_cols = X_train.select_dtypes(include=['bool']).columns

# Convert Booleans (bool)
print(f"Converting boolean columns: {list(boolean_cols)}")
for col in boolean_cols:
    X_train[col] = X_train[col].astype(int)
    X_test[col] = X_test[col].astype(int)

# Convert Categorical (object)
print(f"One-hot encoding categorical columns: {list(categorical_cols)}")
X_train_processed = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_processed = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align Columns
# This is a critical step. It ensures both train and test datasets have the exact same columns, in the same order.
print("Aligning training and testing data columns:")
X_train_final, X_test_final = X_train_processed.align(X_test_processed, join='inner', axis=1, fill_value=0)

print(f"Original number of features: {len(X_train.columns)}")
print(f"Number of features after preprocessing: {len(X_train_final.columns)}")


# Apply StandardScaler - LinearRegression and KNN need scaled data to work properly.
print("Applying StandardScaler:")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final) # Use transform only, don't re-fit

print(f"Number of features after preprocessing: {len(X_train_final.columns)}")

print("Preprocessing is Complete")

# 4. Create validation split - We split our scaled training data to test our models.

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled,  # Use the scaled data
    y_train,
    test_size=0.2,   # 20% for valid. 
    random_state=42
)
print(f"Training split shape: {X_train_split.shape}")
print(f"Validation split shape: {X_val_split.shape}")


# 5. Model Training
# Model 1: Linear Regression
print("Training Linear Regression model:")
model_lr = LinearRegression()
model_lr.fit(X_train_split, y_train_split)

# Evaluate Linear Regression
y_pred_lr = model_lr.predict(X_val_split)
rmse_lr = np.sqrt(mean_squared_error(y_val_split, y_pred_lr))
print(f"Linear Regression Validation RMSE: {rmse_lr}")

# Plot Feature Importance - For Linear Regression, "importance" is shown by the model's coefficients.
print("\nPlotting Linear Regression Coefficients")

# Get the feature names from the *processed* dataframe
feature_names = X_train_final.columns

# Create a DataFrame for easy plotting
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model_lr.coef_
})

# Get Top 20 Features (Top 10 positive, top 10 negative) - We sort by the absolute value to find the most impactful
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

# Plot the top 20 most impactful features
plt.figure(figsize=(12, 8))
sns.barplot(
    data=coef_df.head(20),
    x='Coefficient',
    y='Feature',
    palette='vlag'
)
plt.title('Top 20 Most Impactful Features (Linear Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()

# Model 2: K-Neighbors Regressor
print("\nTraining K-Neighbors Regressor (k=10):")
# We'll be picking 10 neighbors as a good starting guess/point
model_knn = KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
model_knn.fit(X_train_split, y_train_split)

# Evaluate KNN Regressor
y_pred_knn = model_knn.predict(X_val_split)
rmse_knn = np.sqrt(mean_squared_error(y_val_split, y_pred_knn))
print(f"K-Neighbors Regressor Validation RMSE: {rmse_knn}")

# Compare Models
print("\n Model Comparison")
print(f"Linear Regression RMSE: {rmse_lr}")
print(f"K-Neighbors         RMSE: {rmse_knn}")

print("Model Training is Complete")


# 6. Create Submission File
print("\n Creating Submission File")
# Choose the model with the lower RMSE
if rmse_lr < rmse_knn:
    print("Linear Regression was better. Retraining on 100% of data.")
    final_model = LinearRegression()
else:
    print("K-Neighbors Regressor was better. Retraining on 100% of data.")
    final_model = KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
    
# Retrain the chosen model on all the data - We use the *full* scaled training set (X_train_scaled, y_train)
final_model.fit(X_train_scaled, y_train)
print("Final model training complete.")

# Make predictions on the final test data - We use the scaled test set (X_test_scaled)
test_predictions = final_model.predict(X_test_scaled)

# Create the submission DataFrame in the required format
submission_df = pd.DataFrame({
    'id': test_ids,
    'accident_risk': test_predictions
})

# Save the submission file to a CSV.
submission_df.to_csv('submission.csv', index=False)

print("\n Submission File Head")
print(submission_df.head())

