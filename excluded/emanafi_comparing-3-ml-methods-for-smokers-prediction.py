# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import subprocess
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s3e24/train.csv', encoding='latin-1')
test_df = pd.read_csv('/kaggle/input/playground-series-s3e24/test.csv', encoding='latin-1')


# Remove 'id' column from both datasets early (before any preprocessing)
# Store test IDs separately for later use in submission
test_ids = test_df['id'].copy()

# Drop 'id' from both datasets
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])


# Remove duplicates from training data early (before any preprocessing)
print(f'Train duplicate rows (before): {train_df.duplicated().sum()}')
train_df.drop_duplicates(inplace=True)
print(f'Train duplicate rows (after): {train_df.duplicated().sum()}')

# Remove duplicates from test data
print(f'Test duplicate rows (before): {test_df.duplicated().sum()}')
test_df.drop_duplicates(inplace=True)
print(f'Test duplicate rows (after): {test_df.duplicated().sum()}')


# Preview the training dataset
print("Training dataset:")
train_df.head()


# Preview the testing dataset
print("Testing dataset:")
test_df.head()


# Review the basic info and summary statistics of the training dataset
print(train_df.info())
print(train_df.describe())


# Review the basic info and summary statistics of the testing dataset
print(test_df.info())
print(test_df.describe())


# Check for missing values in training data
print("Missing values in train_df:")
print(train_df.isnull().sum() / len(train_df) * 100)


# Check for missing values in test data
print("Missing values in test_df:")
print(test_df.isnull().sum() / len(test_df) * 100)


# Print the data types for each column
print(train_df.dtypes)


# Print the data types for each column
print(test_df.dtypes)


# Define a function to handle outliers for float columns
def treat_outliers_float(column, method="median"):
    """
    Handle outliers for float columns using the IQR method.
    Outliers can be replaced with the median or mean.

    Parameters:
    column (pd.Series): The column to process.
    method (str): Replacement method for outliers, either 'median' or 'mean'.

    Returns:
    pd.Series: Column with outliers handled.
    """
    # Visualize data to spot outliers
    sns.boxplot(x=column)
    plt.title(f'Boxplot Before Outlier Treatment (Float) - {column.name}')
    plt.show()

    # Calculate IQR
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Choose replacement method
    if method == "median":
        replacement_value = column.median()
    elif method == "mean":
        replacement_value = column.mean()
    else:
        raise ValueError("Method must be 'median' or 'mean'")

    # Replace outliers
    column = column.apply(lambda x: replacement_value if x < lower_bound or x > upper_bound else x)

    # Visualize data after outlier treatment
    sns.boxplot(x=column)
    plt.title(f'Boxplot After Outlier Treatment (Float) - {column.name}')
    plt.show()
    
    return column

# Define a function to handle outliers for int columns
def treat_outliers_int(column, method="mean"):
    """
    Handle outliers for integer columns using the IQR method.
    Outliers can be replaced with the median or mean.

    Parameters:
    column (pd.Series): The column to process.
    method (str): Replacement method for outliers, either 'median' or 'mean'.

    Returns:
    pd.Series: Column with outliers handled.
    """
    # Visualize data to spot outliers
    sns.boxplot(x=column)
    plt.title(f'Boxplot Before Outlier Treatment (Int) - {column.name}')
    plt.show()

    # Calculate IQR
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Choose replacement method
    if method == "median":
        replacement_value = column.median()
    elif method == "mean":
        replacement_value = column.mean()
    else:
        raise ValueError("Method must be 'median' or 'mean'")

    # Replace outliers
    column = column.apply(lambda x: replacement_value if x < lower_bound or x > upper_bound else x)

    # Visualize data after outlier treatment
    sns.boxplot(x=column)
    plt.title(f'Boxplot After Outlier Treatment (Int) - {column.name}')
    plt.show()

    return column


# Iterate through all columns in train_df
for col in train_df.columns:
    if train_df[col].dtype == 'float64':
        train_df[col] = treat_outliers_float(train_df[col])
    elif train_df[col].dtype == 'int64':
        train_df[col] = treat_outliers_int(train_df[col])


# For test_df, apply the same outlier bounds calculated from train_df
# This prevents data leakage
for col in test_df.columns:
    if col in train_df.columns and test_df[col].dtype in ['float64', 'int64']:
        # Use training data statistics
        train_col = train_df[col]
        Q1 = train_col.quantile(0.25)
        Q3 = train_col.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Use median as replacement value from training data
        replacement_value = train_col.median()
        
        # Apply to test data
        test_df[col] = test_df[col].apply(lambda x: replacement_value if x < lower_bound or x > upper_bound else x)


# Select numerical columns for standardization
standardized_cols = [
    'age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'systolic', 
    'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride', 
    'HDL', 'LDL', 'hemoglobin', 'serum creatinine', 'AST', 'ALT', 'Gtp'
]

# Standardize
standard_scaler = StandardScaler()
train_df[standardized_cols] = standard_scaler.fit_transform(train_df[standardized_cols])
test_df[standardized_cols] = standard_scaler.transform(test_df[standardized_cols])

# Save the scaler for future use
joblib.dump(standard_scaler, 'standard_scaler.pkl')
print("StandardScaler saved as 'standard_scaler.pkl'")

# Select numerical columns for normalization
normalized_cols = [
    'eyesight(left)', 'eyesight(right)', 'hearing(left)', 'hearing(right)', 
    'Urine protein', 'dental caries'
]

# Add 'smoking' column to normalization only if it exists in the test_df
if 'smoking' in test_df.columns:
    normalized_cols.append('smoking')

# Normalize
minmax_scaler = MinMaxScaler()
train_df[normalized_cols] = minmax_scaler.fit_transform(train_df[normalized_cols])
test_df[normalized_cols] = minmax_scaler.transform(test_df[normalized_cols])

# Save the scaler for future use
joblib.dump(minmax_scaler, 'minmax_scaler.pkl')
print("MinMaxScaler saved as 'minmax_scaler.pkl'")


# Compute the correlation matrix for numerical features
train_correlation_matrix = train_df.corr()
test_correlation_matrix = test_df.corr()


# Display the correlation matrix
print("Correlation Matrix for train_df:")
print(train_correlation_matrix)


# Display the correlation matrix
print("Correlation Matrix for test_df:")
print(test_correlation_matrix)


# Set up a massive figure size for better clarity
plt.figure(figsize=(120, 100))

# Create a heatmap with annotations, larger font sizes, and a color scheme
sns.heatmap(
    train_correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    annot_kws={"size": 25},  # Adjust annotation font size here
)

# Rotate x-axis and y-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=25)  # Larger font size
plt.yticks(rotation=0, fontsize=25)  # Larger font size

# Set the title with a larger font size
plt.title("Correlation Matrix for Training", fontsize=40)

# Adjust layout to fit everything neatly
plt.tight_layout()

# Display the plot
plt.show()


# Set up a massive figure size for better clarity
plt.figure(figsize=(120, 100))

# Create a heatmap with annotations, larger font sizes, and a color scheme
sns.heatmap(
    test_correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    annot_kws={"size": 25},  # Adjust annotation font size here
)

# Rotate x-axis and y-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=25)  # Larger font size
plt.yticks(rotation=0, fontsize=25)  # Larger font size

# Set the title with a larger font size
plt.title("Correlation Matrix for Testing", fontsize=40)

# Adjust layout to fit everything neatly
plt.tight_layout()

# Display the plot
plt.show()


# Validate data
print(train_df.isnull().sum())
print(train_df.dtypes)

# Review the basic info and summary statistics of the dataset
print(train_df.info())
print(train_df.describe())


# Validate data
print(test_df.isnull().sum())
print(test_df.dtypes)

# Review the basic info and summary statistics of the dataset
print(test_df.info())
print(test_df.describe())


# Split train_df into train and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(train_df.drop(columns=['smoking']), train_df['smoking'], test_size=0.2, random_state=42)


# Ensure test_df has the same columns as train_df (except 'smoking')
train_features = train_df.drop(columns=['smoking']).columns
test_df_processed = test_df[train_features]


# Initialize dictionaries for storing results
results = {}
metrics = {}
losses = {}


# Helper function to calculate and store additional metrics
def store_metrics(model_name, y_true, y_pred):
    metrics[model_name] = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted')
    }


# Helper function to print classification report (if y_true is available)
def print_classification_report(y_true, y_pred, model_name):
    if y_true is not None:
        print(f"{model_name} Classification Report:")
        print(classification_report(y_true, y_pred))
    else:
        print(f"{model_name}: No true labels available for evaluation.")


# Decision Tree Classifier
dt_variations = [
    {'name': 'Depth_3', 'params': {'max_depth': 3}},
    {'name': 'Depth_5', 'params': {'max_depth': 5}},
    {'name': 'Depth_None', 'params': {'max_depth': None}},
]

for variation in dt_variations:
    model = DecisionTreeClassifier(random_state=42, **variation['params'])
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)  # Predict on the validation data
    results[f"Decision Tree {variation['name']}"] = predictions  # Store predictions
    store_metrics(f"Decision Tree {variation['name']}", y_val, predictions)  # Store metrics
    print_classification_report(y_val, predictions, f"Decision Tree {variation['name']}")  # Evaluate on validation set

    # Predict on test_df_processed (preprocessed test data)
    test_predictions = model.predict(test_df_processed)
    results[f"Decision Tree {variation['name']} - Test"] = test_predictions


# Logistic Regression
lr_variations = [
    {'name': 'liblinear', 'params': {'solver': 'liblinear', 'max_iter': 1000}},
    {'name': 'lbfgs', 'params': {'solver': 'lbfgs', 'max_iter': 1000}},
    {'name': 'saga', 'params': {'solver': 'saga', 'max_iter': 1000}},
]

for variation in lr_variations:
    model = LogisticRegression(random_state=42, **variation['params'])
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)  # Predict on the validation data
    results[f"Logistic Regression {variation['name']}"] = predictions  # Store predictions
    store_metrics(f"Logistic Regression {variation['name']}", y_val, predictions)  # Store metrics
    print_classification_report(y_val, predictions, f"Logistic Regression {variation['name']}")  # Evaluate on validation set

    # Predict on test_df_processed (preprocessed test data)
    test_predictions = model.predict(test_df_processed)
    results[f"Logistic Regression {variation['name']} - Test"] = test_predictions


# Neural Network (MLP Classifier) with Variations
nn_variations = [
    {'name': '64_32', 'params': {'hidden_layer_sizes': (64, 32), 'activation': 'relu', 'solver': 'adam'}},
    {'name': '128_64', 'params': {'hidden_layer_sizes': (128, 64), 'activation': 'relu', 'solver': 'adam'}},
    {'name': '32_16', 'params': {'hidden_layer_sizes': (32, 16), 'activation': 'relu', 'solver': 'adam'}},
]

for variation in nn_variations:
    print(f"Training Neural Network {variation['name']}...")
    model = MLPClassifier(max_iter=1000, random_state=42, **variation['params'])
    model.fit(X_train, y_train)

    # Save loss values
    losses[f"Neural Network {variation['name']}"] = model.loss_curve_

    # Print the number of iterations used in training
    print(f"Neural Network {variation['name']} completed in {model.n_iter_} iterations.")
    
    # Predict on validation set
    predictions = model.predict(X_val)
    results[f"Neural Network {variation['name']}"] = predictions  # Store predictions
    store_metrics(f"Neural Network {variation['name']}", y_val, predictions)  # Store metrics
    print_classification_report(y_val, predictions, f"Neural Network {variation['name']}")  # Evaluate on validation set

    # Predict on test_df_processed (preprocessed test data)
    test_predictions = model.predict(test_df_processed)
    results[f"Neural Network {variation['name']} - Test"] = test_predictions


# Compare and identify the best model for each algorithm
def find_best_model():
    best_models = {}

    # Identify the best Decision Tree model
    dt_metrics = {k: v for k, v in metrics.items() if k.startswith("Decision Tree")}
    best_dt_model = max(dt_metrics, key=lambda x: dt_metrics[x]['F1 Score'])
    best_models['Decision Tree'] = best_dt_model
    print(f"Best Decision Tree Model: {best_dt_model}")
    print(f"Metrics: {dt_metrics[best_dt_model]}\n")

    # Identify the best Logistic Regression model
    lr_metrics = {k: v for k, v in metrics.items() if k.startswith("Logistic Regression")}
    best_lr_model = max(lr_metrics, key=lambda x: lr_metrics[x]['F1 Score'])
    best_models['Logistic Regression'] = best_lr_model
    print(f"Best Logistic Regression Model: {best_lr_model}")
    print(f"Metrics: {lr_metrics[best_lr_model]}\n")

    # Identify the best Neural Network model
    nn_metrics = {k: v for k, v in metrics.items() if k.startswith("Neural Network")}
    best_nn_model = max(nn_metrics, key=lambda x: nn_metrics[x]['F1 Score'])
    best_models['Neural Network'] = best_nn_model
    print(f"Best Neural Network Model: {best_nn_model}")
    print(f"Metrics: {nn_metrics[best_nn_model]}\n")

    return best_models


# Call the function to find and display the best models
best_models = find_best_model()

# Example usage for evaluating test data predictions
for model_name in best_models.values():
    test_preds = results.get(f"{model_name} - Test")
    if test_preds is not None:
        print(f"Test Predictions for {model_name}:")
        print(test_preds)
    else:
        print(f"No test predictions available for {model_name}.")


# Check sizes of y_val and predictions
print(f"Size of y_val: {len(y_val)}")
for model_name, predictions in results.items():
    if predictions is not None:
        print(f"{model_name} predictions size: {len(predictions)}")

# Optionally, truncate predictions to match y_val length
for model_name, predictions in results.items():
    if predictions is not None and len(predictions) != len(y_val):
        results[model_name] = predictions[:len(y_val)]  # Adjust size to match y_val


# Function to plot metrics for the best models
def plot_metrics(metric_name, metrics_dict, best_models):
    plt.figure(figsize=(10, 6))
    values = [metrics_dict[model][metric_name] for model in best_models.values()]
    labels = list(best_models.values())
    plt.bar(labels, values, color='lightcoral')
    plt.title(f'Model Performance Comparison - {metric_name}')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Plot Accuracy
plot_metrics("Accuracy", metrics, best_models)

# Plot Precision
plot_metrics("Precision", metrics, best_models)

# Plot Recall
plot_metrics("Recall", metrics, best_models)

# Plot F1 Score
plot_metrics("F1 Score", metrics, best_models)

# Loss Visualization (if applicable, for Neural Networks only)
def plot_loss(loss_dict, best_models):
    plt.figure(figsize=(10, 6))
    for model_name in best_models.values():
        if model_name in loss_dict:
            plt.plot(loss_dict[model_name], label=model_name)
    plt.title("Loss Curves for Best Neural Network Models")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Assuming `losses` contains training losses for each neural network
# Example: losses = {'Neural Network 64_32': [0.5, 0.4, 0.35, ...], ...}
plot_loss(losses, best_models)


# Save all three best models
model_filenames = {}

# Prepare full training data (combine X_train and X_val for final training)
X_full = pd.concat([X_train, X_val], axis=0)
y_full = pd.concat([y_train, y_val], axis=0)

# Train and save the best Decision Tree model
best_dt_name = best_models['Decision Tree']
if "Depth_3" in best_dt_name:
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
elif "Depth_5" in best_dt_name:
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
else:
    dt_model = DecisionTreeClassifier(max_depth=None, random_state=42)

dt_model.fit(X_full, y_full)
dt_filename = 'best_decision_tree.pkl'
joblib.dump(dt_model, dt_filename)
model_filenames['Decision Tree'] = dt_filename
print(f"Decision Tree model saved as '{dt_filename}'")

# Train and save the best Logistic Regression model
best_lr_name = best_models['Logistic Regression']
if "liblinear" in best_lr_name:
    lr_model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
elif "lbfgs" in best_lr_name:
    lr_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
else:
    lr_model = LogisticRegression(solver='saga', max_iter=1000, random_state=42)

lr_model.fit(X_full, y_full)
lr_filename = 'best_logistic_regression.pkl'
joblib.dump(lr_model, lr_filename)
model_filenames['Logistic Regression'] = lr_filename
print(f"Logistic Regression model saved as '{lr_filename}'")

# Train and save the best Neural Network model
best_nn_name = best_models['Neural Network']
if "64_32" in best_nn_name:
    nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
elif "128_64" in best_nn_name:
    nn_model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=1000, random_state=42)
else:
    nn_model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=1000, random_state=42)

nn_model.fit(X_full, y_full)
nn_filename = 'best_neural_network.pkl'
joblib.dump(nn_model, nn_filename)
model_filenames['Neural Network'] = nn_filename
print(f"Neural Network model saved as '{nn_filename}'")

# Determine and save the overall best model
best_overall_model = max(metrics, key=lambda x: metrics[x]['F1 Score'])
print(f"\n{'='*50}")
print(f"Best Overall Model: {best_overall_model}")
print(f"Metrics: {metrics[best_overall_model]}")
print(f"{'='*50}\n")

# Save the overall best model separately
if best_overall_model.startswith("Decision Tree"):
    final_model = dt_model
    best_model_filename = dt_filename
elif best_overall_model.startswith("Logistic Regression"):
    final_model = lr_model
    best_model_filename = lr_filename
else:
    final_model = nn_model
    best_model_filename = nn_filename

# Also save as 'best_model.pkl' for easy reference
joblib.dump(final_model, 'best_model.pkl')
print(f"Best overall model also saved as 'best_model.pkl'")
print(f"\nAll models saved successfully!")
print(f"Model files: {model_filenames}")



# Use Kaggle API to automatically detect and increment version

# Get Kaggle username from Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    kaggle_username = user_secrets.get_secret("KAGGLE_USERNAME")
    print(f"Retrieved username from Kaggle Secrets: {kaggle_username}")
except Exception as e:
    # Fallback for local testing or if secret not set
    print(f"Could not access Kaggle Secrets ({e}), using fallback method")
    try:
        # Try to get username from Kaggle config
        config_result = subprocess.run(
            ['kaggle', 'config', 'view'],
            capture_output=True,
            text=True
        )
        if config_result.returncode == 0:
            for line in config_result.stdout.split('\n'):
                if 'username:' in line.lower():
                    kaggle_username = line.split(':')[1].strip()
                    print(f"Retrieved username from Kaggle config: {kaggle_username}")
                    break
            else:
                kaggle_username = os.environ.get('KAGGLE_USERNAME')
                print(f"Using username from environment variable: {kaggle_username}")
        else:
            kaggle_username = os.environ.get('KAGGLE_USERNAME')
            print(f"Using username from environment variable: {kaggle_username}")
    except Exception as fallback_error:
        print(f"Fallback methods failed: {fallback_error}")
        raise ValueError("Could not determine Kaggle username. Please set KAGGLE_USERNAME in Kaggle Secrets.")

# Build the model identifier
KAGGLE_MODEL = f"{kaggle_username}/model-for-binary-prediction-of-smoker-status"
print(f"Kaggle model: {KAGGLE_MODEL}")

try:
    # Query Kaggle API to get current version
    result = subprocess.run(
        ['kaggle', 'models', 'instances', 'versions', '-m', KAGGLE_MODEL, '--page-size', '1'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # Parse the output to get the latest version number
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) > 1:
            # Skip header, get first data line
            latest_version_line = output_lines[1].strip()
            # Extract version number (first column)
            latest_version = int(latest_version_line.split()[0])
            next_version = latest_version + 1
            print(f"Found Kaggle model version {latest_version}, creating version {next_version}")
        else:
            next_version = 1
            print("No existing versions found, starting with version 1")
    else:
        # Fallback if API call fails
        print("Kaggle API call failed, using fallback version detection")
        next_version = 6  # Set to your current version + 1
        print(f"Using fallback version: {next_version}")
        
except Exception as e:
    print(f"Error querying Kaggle API: {e}")
    print("Using fallback version detection")
    next_version = 6  # Set to your current version + 1
    print(f"Using fallback version: {next_version}")

# Create the version directory
base_dir = 'model-for-binary-prediction-of-smoker-status/scikitlearn/main'
model_dir = os.path.join(base_dir, str(next_version))
os.makedirs(model_dir, exist_ok=True)

print(f"Creating model version: {next_version}")

# Save the best model to the versioned directory
model_path = os.path.join(model_dir, 'best_model.pkl')
joblib.dump(final_model, model_path)
print(f"Model saved to '{model_path}'")
print(f"Best Model: {best_overall_model}")
print(f"Model type: {type(final_model).__name__}\n")

# Load the best model from the versioned directory
print(f"Loading the model from '{model_path}'...")
loaded_model = joblib.load(model_path)
print(f"Model loaded successfully!")
print(f"Model type: {type(loaded_model).__name__}\n")

# Generate predictions for test dataset using the loaded model
# Use the preprocessed test data (without 'id' column)
print("Generating predictions for test dataset...")
test_predictions = loaded_model.predict(test_df_processed)

# Create submission dataframe using the stored test IDs
submission_df = pd.DataFrame({
    'id': test_ids,
    'smoking': test_predictions
})

# Save to CSV
submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)
print(f"\nSubmission file saved as '{submission_filename}'")
print(f"\nFirst few rows of submission:")
print(submission_df.head())
print(f"\nTotal predictions: {len(submission_df)}")
print(f"\nValue counts of predictions:")
print(submission_df['smoking'].value_counts())


