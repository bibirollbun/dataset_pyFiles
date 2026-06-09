# Core libraries for data manipulation and analysis
import pandas as pd  # Import pandas for data manipulation and analysis, commonly used for reading and writing CSV files and creating DataFrames.
import numpy as np   # Import numpy for numerical operations, especially for array manipulation and mathematical functions.

# Visualization libraries
import matplotlib.pyplot as plt  # Import matplotlib.pyplot for creating static, animated, and interactive visualizations.
import seaborn as sns            # Import seaborn for making statistical graphics; it's built on top of matplotlib and provides a high-level interface.

# Machine learning libraries
from sklearn.model_selection import KFold          # Import KFold from scikit-learn for creating cross-validation folds.
from sklearn.preprocessing import LabelEncoder     # Import LabelEncoder from scikit-learn to convert categorical labels into numerical format.
from sklearn.metrics import mean_squared_error   # Import mean_squared_error to calculate the MSE metric, which we'll use for RMSE.
import lightgbm as lgb                           # Import the LightGBM library, a fast, distributed, high-performance gradient boosting framework.

# Utility libraries
import time                # Import the time module to measure the execution time of code blocks.
import gc                  # Import the garbage collection module to manually manage memory.
from contextlib import contextmanager # Import contextmanager to create a simple utility for timing code blocks.

# Set some global options for better display
pd.set_option('display.max_columns', None) # Set a pandas option to display all columns of a DataFrame without truncation.
sns.set_style('whitegrid')                 # Set the aesthetic style of the plots to 'whitegrid' using seaborn.

# Timer utility function
@contextmanager # This decorator transforms the generator function into a proper context manager.
def timer(title): # Define a function named 'timer' that takes a title string as an argument.
    t0 = time.time() # Record the starting time before the code block is executed.
    yield # Yield control back to the code block within the 'with' statement.
    print(f"{title} - done in {time.time() - t0:.2f}s") # After the block, calculate and print the elapsed time.

# Function to calculate RMSE
def rmse(y_true, y_pred): # Define a function to calculate the Root Mean Squared Error.
    return np.sqrt(mean_squared_error(y_true, y_pred)) # Return the square root of the mean squared error between true and predicted values.


# Load the datasets
with timer("Loading data"): # Use the timer utility to measure how long data loading takes.
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv') # Read the competition's training data into a pandas DataFrame.
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')   # Read the competition's test data into a pandas DataFrame.
    original_df = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv') # Read the external, original dataset into a DataFrame.
    sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv') # Read the sample submission file to understand the required format.

# Display the shape of the datasets
print("Shape of Competition Train Data:", train_df.shape) # Print the number of rows and columns in the training dataset.
print("Shape of Competition Test Data:", test_df.shape)   # Print the number of rows and columns in the test dataset.
print("Shape of Original Data:", original_df.shape)       # Print the number of rows and columns in the original dataset.


print("Competition Train Data Head:") # Print a descriptive header for the output.
display(train_df.head()) # Display the first 5 rows of the training DataFrame for a quick inspection.

print("\nCompetition Test Data Head:") # Print a descriptive header for the output, with a newline for spacing.
display(test_df.head()) # Display the first 5 rows of the test DataFrame.

print("\nOriginal Data Head:") # Print a descriptive header for the output.
display(original_df.head()) # Display the first 5 rows of the original dataset DataFrame.


# Standardize column names
def standardize_columns(df): # Define a function to process DataFrame column names.
    df.columns = df.columns.str.lower().str.replace(' ', '_') # Convert all column names to lowercase and replace spaces with underscores.
    return df # Return the DataFrame with the modified column names.

train_df = standardize_columns(train_df) # Apply the standardization function to the training DataFrame.
test_df = standardize_columns(test_df)   # Apply the standardization function to the test DataFrame.
original_df = standardize_columns(original_df) # Apply the standardization function to the original DataFrame.

# This step is a placeholder for aligning column names if they differed significantly.
# In this specific case, the column names are already well-aligned after standardization.
if 'accident_probability' in original_df.columns: # Check if a column named 'accident_probability' exists in the original dataset.
    original_df.rename(columns={'accident_probability': 'accident_risk'}, inplace=True) # Rename it to 'accident_risk' to match the competition data.

# Combine the training data
combined_train_df = pd.concat([train_df, original_df], ignore_index=True) # Concatenate the competition training data and the original data into a single DataFrame.
print("Shape of Combined Training Data:", combined_train_df.shape) # Print the shape of the newly combined training DataFrame.


plt.figure(figsize=(12, 6)) # Create a new figure for plotting with a specified size (12 inches wide, 6 inches high).
sns.histplot(train_df['accident_risk'], color='blue', label='Competition Train', kde=True, stat="density", linewidth=0) # Plot a histogram and KDE of the target variable for the competition training data.
sns.histplot(original_df['accident_risk'], color='orange', label='Original Data', kde=True, stat="density", linewidth=0) # Overlay a histogram and KDE of the target variable for the original dataset.
plt.title('Distribution of Accident Risk') # Set the title for the plot.
plt.legend() # Display the legend, which shows the labels for each plotted dataset.
plt.show() # Render and display the plot.


display(combined_train_df.describe()) # Generate and display descriptive statistics for the numerical columns in the combined training DataFrame.


categorical_features = combined_train_df.select_dtypes(include=['object', 'category']).columns # Identify columns with 'object' or 'category' data types.

for col in categorical_features: # Loop through each identified categorical column.
    print(f"Value counts for {col}:") # Print a header indicating which column's value counts are being shown.
    print(combined_train_df[col].value_counts()) # Print the frequency of each unique value in the column.
    print("-" * 30) # Print a separator line for better readability.


# For correlation, we need to handle categorical features. We'll label encode them for this visualization.
temp_df = combined_train_df.copy() # Create a temporary copy of the DataFrame to avoid altering the original during this visualization step.
for col in categorical_features: # Iterate over the list of categorical feature names.
    temp_df[col] = LabelEncoder().fit_transform(temp_df[col]) # Apply label encoding to each categorical column to convert it to a numerical format for the correlation matrix.

plt.figure(figsize=(16, 12)) # Create a new figure with a large size for better readability of the heatmap.
correlation_matrix = temp_df.corr() # Calculate the pairwise correlation of columns in the temporary DataFrame.
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f") # Generate a heatmap of the correlation matrix, with annotations, a specific color map, and formatted to two decimal places.
plt.title('Correlation Matrix of Features') # Set the title of the heatmap plot.
plt.show() # Display the generated heatmap.


with timer("Encoding categorical features"): # Use the timer to measure the execution time of the encoding process.
    for col in categorical_features: # Iterate over the list of categorical feature names.
        le = LabelEncoder() # Create an instance of the LabelEncoder.
        combined_train_df[col] = le.fit_transform(combined_train_df[col]) # Fit the encoder on the training data and transform the column to its encoded version.
        test_df[col] = le.transform(test_df[col]) # Use the *same* fitted encoder to transform the corresponding column in the test data.


def create_features(df): # Define a function to create new features for a given DataFrame.
    # Note: Using 'num_lanes' as seen in the correlation matrix, not 'number_of_lanes'.
    df['speed_to_lanes'] = df['speed_limit'] / df['num_lanes'] # Create a new feature representing the ratio of speed limit to the number of lanes.
    
    # Note: Using 'lighting' and 'weather' as seen in the correlation matrix.
    df['light_x_weather'] = df['lighting'] * df['weather'] # Create an interaction feature between lighting and weather conditions.
    
    # Time of day categorization
    bins = [0, 6, 12, 18, 24] # Define the hour bins for categorizing the time of day (Night, Morning, Afternoon, Evening).
    labels = [0, 1, 2, 3] # Define the numerical labels for these categories.
    # Note: Using 'time_of_day' as seen in the correlation matrix, not 'hour_of_day'.
    df['time_of_day_category'] = pd.cut(df['time_of_day'], bins=bins, labels=labels, right=False) # Create a new column by binning the 'time_of_day' feature.
    df['time_of_day_category'] = df['time_of_day_category'].astype(int) # Convert the new categorical feature to an integer type.
    
    return df # Return the DataFrame with the newly added features.

with timer("Creating features"): # Use the timer to measure the feature engineering process.
    combined_train_df = create_features(combined_train_df) # Apply the feature creation function to the combined training DataFrame.
    test_df = create_features(test_df) # Apply the feature creation function to the test DataFrame.


# Prepare data for LightGBM
X = combined_train_df.drop(['id', 'accident_risk'], axis=1) # Create the feature matrix 'X' by dropping the identifier and target columns.
y = combined_train_df['accident_risk'] # Create the target vector 'y' containing only the 'accident_risk' column.
X_test = test_df.drop('id', axis=1) # Create the test feature matrix by dropping the identifier column.

# Align columns - crucial step to ensure train and test sets have the same features in the same order
train_cols = X.columns # Get the column names from the training features.
test_cols = X_test.columns # Get the column names from the test features.
if not all(train_cols == test_cols): # Check if all column names in train and test are identical.
    raise ValueError("Train and test columns are not aligned!") # Raise an error if they are not, as this would break the model's predict function.

# LightGBM parameters
params = { # Define a dictionary of parameters for the LightGBM model.
    'objective': 'regression_l1', # Specify the objective function for the regression (L1 loss, also known as Mean Absolute Error).
    'metric': 'rmse', # Specify the evaluation metric to be used (Root Mean Squared Error).
    'n_estimators': 2000, # Set the maximum number of boosting rounds (trees) to be built.
    'learning_rate': 0.01, # Set the learning rate, which controls the step size at each iteration.
    'feature_fraction': 0.8, # Specify the fraction of features to be considered for each tree.
    'bagging_fraction': 0.8, # Specify the fraction of data to be used for each tree (data subsampling).
    'bagging_freq': 1, # Specify the frequency for bagging (perform bagging at every iteration).
    'lambda_l1': 0.1, # Specify the L1 regularization term.
    'lambda_l2': 0.1, # Specify the L2 regularization term.
    'num_leaves': 31, # Set the maximum number of leaves in one tree.
    'verbose': -1, # Set verbosity to -1 to suppress detailed output during training.
    'n_jobs': -1, # Use all available CPU cores for training.
    'seed': 42, # Set a random seed for reproducibility.
    'boosting_type': 'gbdt', # Specify the boosting type as Gradient Boosting Decision Tree.
}

# Cross-validation setup
N_SPLITS = 5 # Define the number of folds for cross-validation.
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42) # Initialize the KFold cross-validator with shuffling for randomness.

oof_preds = np.zeros(len(X)) # Initialize a numpy array filled with zeros to store out-of-fold predictions for the training data.
test_preds = np.zeros(len(X_test)) # Initialize a numpy array to store the aggregated predictions for the test data.
feature_importances = pd.DataFrame(index=X.columns) # Create a DataFrame to store the feature importances from each fold.

with timer("Training model with K-Fold Cross-Validation"): # Use the timer to measure the entire training and validation process.
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)): # Loop through each fold, getting training and validation indices.
        print(f"===== Fold {fold+1} =====") # Print the current fold number.
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx] # Create the training data for the current fold using the indices.
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx] # Create the validation data for the current fold.
        
        model = lgb.LGBMRegressor(**params) # Initialize a new LightGBM Regressor model with the specified parameters for each fold.
        
        model.fit(X_train, y_train, # Train the model on the current fold's training data.
                  eval_set=[(X_val, y_val)], # Provide the validation set to monitor performance.
                  eval_metric='rmse', # Specify the evaluation metric for the validation set.
                  callbacks=[lgb.early_stopping(100, verbose=False)]) # Use early stopping to prevent overfitting if the validation score doesn't improve for 100 rounds.
        
        val_preds = model.predict(X_val) # Make predictions on the validation set.
        oof_preds[val_idx] = val_preds # Store these predictions in the corresponding part of the out-of-fold predictions array.
        
        fold_test_preds = model.predict(X_test) # Make predictions on the entire test set using the model trained on this fold.
        test_preds += fold_test_preds / N_SPLITS # Add the predictions to the total test predictions, averaging over the number of folds.
        
        feature_importances[f'fold_{fold+1}'] = model.feature_importances_ # Store the feature importances from the current model.
        
        fold_rmse = rmse(y_val, val_preds) # Calculate the RMSE for the current fold's validation predictions.
        print(f"Fold {fold+1} RMSE: {fold_rmse}") # Print the RMSE for this fold.
        
        gc.collect() # Manually trigger garbage collection to free up memory after each fold.

# Overall OOF RMSE
overall_rmse = rmse(y, oof_preds) # Calculate the overall RMSE using all the out-of-fold predictions against the true training labels.
print(f"\nOverall OOF RMSE: {overall_rmse}") # Print the final, overall cross-validated RMSE score.


feature_importances['mean'] = feature_importances.mean(axis=1) # Calculate the mean feature importance across all folds.
feature_importances.sort_values('mean', ascending=False, inplace=True) # Sort the features by their mean importance in descending order.

plt.figure(figsize=(10, 8)) # Create a new figure for the plot with a specified size.
sns.barplot(x='mean', y=feature_importances.index, data=feature_importances) # Create a bar plot to visualize the feature importances.
plt.title('LightGBM Feature Importances (Mean over folds)') # Set the title for the feature importance plot.
plt.show() # Display the plot.


# Create the submission file
submission_df = pd.DataFrame({'id': test_df['id'], 'accident_risk': test_preds}) # Create a new DataFrame for submission with 'id' and the predicted 'accident_risk'.

# Ensure predictions are within the [0, 1] range
submission_df['accident_risk'] = np.clip(submission_df['accident_risk'], 0, 1) # Clip the predictions to ensure they fall within the valid range of 0 to 1.

# Save the submission file
submission_df.to_csv('submission.csv', index=False) # Save the DataFrame to a CSV file named 'submission.csv' without the DataFrame index.

print("Submission file created successfully!") # Print a confirmation message.
display(submission_df.head()) # Display the first few rows of the final submission file.

