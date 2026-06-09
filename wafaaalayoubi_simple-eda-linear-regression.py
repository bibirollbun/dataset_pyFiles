# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn for modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error


# Set some display options for pandas
pd.set_option('display.max_columns', None)


# Load the datasets
# The file paths are standard for Kaggle notebooks
train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')



print("--- Train Data Info ---")
train_df.info()


print("\n--- Test Data Info ---")
test_df.info()


print("\n--- First 5 rows of Train Data ---")
display(train_df.head())


print("\n--- Descriptive Statistics of Train Data ---")
display(train_df.describe())


# Calculate the correlation matrix for all numeric columns in the training data
corr_matrix = train_df.corr()

# Set up the matplotlib figure for a larger display
plt.figure(figsize=(12, 10))

# Generate the heatmap
sns.heatmap(corr_matrix, 
            annot=True,       # Display the correlation values on the heatmap
            fmt='.2f',        # Format the values to two decimal places
            cmap='coolwarm',  # Use a color scheme that highlights positive/negative correlations
            linewidths=.5)    # Add lines between cells for clarity

plt.title('Correlation Matrix of Song Features', fontsize=16)
plt.show()


# To make it even clearer, let's display the correlations with the target variable in a sorted list
print("\nCorrelation of each feature with BeatsPerMinute:")
print(corr_matrix['BeatsPerMinute'].sort_values(ascending=False))


# Define the target variable name
target = 'BeatsPerMinute'

# Define the list of features to be used for training
# We exclude the 'id' column and the target variable itself
features = [col for col in train_df.columns if col not in ['id', target]]

# Create our feature matrix (X) and target vector (y)
X = train_df[features]
y = train_df[target]

# Create the feature matrix for the test set
# It's crucial that this has the same columns as X
X_test = test_df[features]


# Verify the shapes of our new dataframes
print(f"Shape of training features (X): {X.shape}")
print(f"Shape of training target (y): {y.shape}")
print(f"Shape of test features (X_test): {X_test.shape}")


# Display the first few rows of the feature matrix to confirm
print("\nFirst 5 rows of X:")
display(X.head())


# Split the training data (X, y) into a training set and a validation set
# We'll use 80% for training and 20% for validation.
# random_state ensures that the split is the same every time we run the code.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---
# 1. Create an instance of the Linear Regression model
model = LinearRegression()


# 2. Fit the model ONLY to the new, smaller training set
model.fit(X_train, y_train)
print("Linear Regression model has been trained successfully.")


# --- Model Evaluation ---
# 3. Make predictions on the validation set (data it hasn't seen)
val_predictions = model.predict(X_val)


# 4. Calculate the evaluation metrics
r2 = r2_score(y_val, val_predictions)
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

print("\n--- Model Evaluation on Validation Set ---")
print(f"R-squared (R²): {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# --- Retrain the Model on the Full Dataset ---
# We create a new model instance and train it on ALL of our training data (X and y)
final_model = LinearRegression()
final_model.fit(X, y)

print("Final model has been trained on the full training dataset.")


# --- Generate Predictions on the Test Set ---
# Use the final model to make predictions on the test data
test_predictions = final_model.predict(X_test)

print("Predictions generated for the test set.")

# --- Create the Submission File ---
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'BeatsPerMinute': test_predictions
})
# Save the DataFrame to a CSV file for submission
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("First 5 rows of the submission file:")
display(submission_df.head())




