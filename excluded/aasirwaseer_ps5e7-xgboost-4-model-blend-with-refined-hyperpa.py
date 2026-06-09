# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Define the file paths for the training, testing, and sample submission data.
# This makes it easy to change the paths if needed.
TRAIN_PATH = '/kaggle/input/playground-series-s5e7/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e7/test.csv'
SAMPLE_SUBMISSION_PATH = '/kaggle/input/playground-series-s5e7/sample_submission.csv'

# Load the training data from the CSV file into a pandas DataFrame.
# A DataFrame is a 2D labeled data structure with columns of potentially different types.
try:
    train_df = pd.read_csv(TRAIN_PATH)
    print("✅ Training data loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: The file at {TRAIN_PATH} was not found. Please check the file path.")
    # Exit or handle the error appropriately if the file is not found.
    train_df = None

# --- Initial Data Exploration ---
if train_df is not None:
    print("\n--- First 5 Rows of the Training Data ---")
    # Display the first 5 rows of the DataFrame to get a quick overview of the data.
    print(train_df.head())

    print("\n--- Dataframe Information ---")
    # .info() provides a concise summary of the DataFrame.
    # It shows the column names, the number of non-null values, and the data type of each column.
    # This is great for quickly spotting columns with missing data.
    train_df.info()

    print("\n--- Missing Values Count ---")
    # .isnull().sum() counts the number of missing (NaN) values in each column.
    # It's a crucial step to identify which columns might need cleaning or imputation.
    print(train_df.isnull().sum())

    print("\n--- Target Variable Distribution ---")
    # .value_counts() returns a Series containing counts of unique values for the 'Personality' column.
    # This helps us understand if the dataset is balanced or imbalanced between Introverts and Extroverts.
    if 'Personality' in train_df.columns:
        print(train_df['Personality'].value_counts())
    else:
        print("❌ 'Personality' column not found in the training data.")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
# It's good practice to load the data again in a new script or cell.
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    print("✅ Training and testing data loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Make sure the file paths are correct.")
    train_df = None
    test_df = None

if train_df is not None:
    # --- Data Visualization ---
    print("\n--- Visualizing Feature Distributions ---")

    # Separate numerical and categorical columns for easier processing
    numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    # Remove 'id' as it's just an identifier
    numerical_cols.remove('id')
    
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    target_col = 'Personality'

    # Set up the plotting area
    # Create a figure with a specific size for better readability
    plt.figure(figsize=(16, 12))
    plt.suptitle('Feature Distributions by Personality', fontsize=20)

    # Plot distributions for numerical features
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(3, 3, i)
        # Use a histogram with a kernel density estimate to see the shape of the distribution
        sns.histplot(data=train_df, x=col, hue=target_col, kde=True, multiple="stack")
        plt.title(f'Distribution of {col}')
        plt.xlabel('')
        plt.ylabel('')

    # Plot distributions for categorical features
    for i, col in enumerate(categorical_cols, len(numerical_cols) + 1):
        plt.subplot(3, 3, i)
        # A count plot is perfect for showing the frequency of categories
        sns.countplot(data=train_df, x=col, hue=target_col)
        plt.title(f'Distribution of {col}')
        plt.xlabel('')
        plt.ylabel('')
    
    # Adjust layout to prevent plots from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    # --- Data Preprocessing ---
    print("\n--- Starting Data Preprocessing ---")

    # Combine train and test for consistent preprocessing
    combined_df = pd.concat([train_df.drop(target_col, axis=1), test_df], ignore_index=True)

    # 1. Encode Binary Categorical Features
    # Convert 'Yes'/'No' to 1/0 for easier processing
    for col in categorical_cols:
        combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0})
    
    print("✅ Binary features encoded.")

    # 2. Impute Missing Values
    # For numerical columns, fill with the median
    for col in numerical_cols:
        median_val = combined_df[col].median()
        combined_df[col].fillna(median_val, inplace=True)
    
    # For categorical columns, fill with the mode (most frequent value)
    for col in categorical_cols:
        mode_val = combined_df[col].mode()[0]
        combined_df[col].fillna(mode_val, inplace=True)

    print("✅ Missing values imputed.")

    # 3. Encode Target Variable
    # Convert 'Extrovert'/'Introvert' to 1/0
    train_df[target_col] = train_df[target_col].map({'Extrovert': 1, 'Introvert': 0})
    print("✅ Target variable encoded.")

    # 4. Separate back into training and testing sets
    X = combined_df.iloc[:len(train_df)].drop('id', axis=1)
    X_test = combined_df.iloc[len(train_df):].drop('id', axis=1)
    y = train_df[target_col]

    print("\n--- Preprocessing Complete ---")
    print("Shape of training features (X):", X.shape)
    print("Shape of testing features (X_test):", X_test.shape)
    print("Shape of target (y):", y.shape)

    print("\n--- First 5 Rows of Processed Training Data (X) ---")
    print(X.head())



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- NOTE: This script assumes you have run the previous preprocessing script ---
# --- and have the variables X, y, and X_test in memory. ---
# --- For clarity, I'm including the preprocessing steps again. ---

# --- Load Data ---
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File paths are incorrect.")
    # Stop execution if files are not found
    exit()

# --- Preprocessing ---
print("\n--- Running Preprocessing ---")
target_col = 'Personality'
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
numerical_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64'] and col not in ['id', target_col]]

# Combine for consistent processing
combined_df = pd.concat([train_df.drop(target_col, axis=1), test_df], ignore_index=True)

# Encode binary features
for col in categorical_cols:
    combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0})

# Impute missing values
for col in numerical_cols:
    combined_df[col].fillna(combined_df[col].median(), inplace=True)
for col in categorical_cols:
    combined_df[col].fillna(combined_df[col].mode()[0], inplace=True)

# Encode target variable
train_df[target_col] = train_df[target_col].map({'Extrovert': 1, 'Introvert': 0})

# Separate back into train and test sets
X = combined_df.iloc[:len(train_df)].drop('id', axis=1)
X_test = combined_df.iloc[len(train_df):].drop('id', axis=1)
y = train_df[target_col]
print("✅ Preprocessing complete.")


# --- Model Training and Validation ---
print("\n--- Training and Validating Model ---")

# 1. Split data for validation
# We use a 80/20 split. stratify=y ensures the proportion of Introverts/Extroverts is the same in train and validation sets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")

# 2. Initialize and train the RandomForestClassifier
# random_state=42 ensures we get the same results every time we run this.
# n_jobs=-1 uses all available CPU cores to speed up training.
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model trained on the training subset.")

# 3. Make predictions and evaluate on the validation set
y_pred_val = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_val)

print(f"\nValidation Accuracy: {accuracy:.4f}")
print("\nValidation Classification Report:")
# This report gives us a detailed breakdown of performance for each class.
print(classification_report(y_val, y_pred_val, target_names=['Introvert', 'Extrovert']))


# --- Create Submission File ---
print("\n--- Generating Submission File ---")

# 1. Retrain the model on the ENTIRE training dataset (X, y)
# This ensures the model learns from all available data before predicting on the test set.
model.fit(X, y)
print("✅ Model retrained on the full dataset.")

# 2. Make predictions on the official test data
test_predictions = model.predict(X_test)

# 3. Format the predictions into the required submission format
# Convert numerical predictions back to string labels
test_predictions_labels = np.where(test_predictions == 1, 'Extrovert', 'Introvert')

# Create the submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': test_predictions_labels})

# 4. Save the submission file
submission_df.to_csv('submission.csv', index=False)
print("\n✅ Submission file 'submission.csv' created successfully!")
print("You can now submit this file to the Kaggle competition.")

# Display the first few rows of the submission file
print("\n--- First 5 Rows of Submission File ---")
print(submission_df.head())



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- NOTE: This script uses the same preprocessing as before. ---
# --- For clarity, I'm including the preprocessing steps again. ---

# --- Load Data ---
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File paths are incorrect.")
    exit()

# --- Preprocessing ---
print("\n--- Running Preprocessing ---")
target_col = 'Personality'
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
numerical_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64'] and col not in ['id', target_col]]

combined_df = pd.concat([train_df.drop(target_col, axis=1), test_df], ignore_index=True)

for col in categorical_cols:
    combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0})

for col in numerical_cols:
    combined_df[col].fillna(combined_df[col].median(), inplace=True)
for col in categorical_cols:
    combined_df[col].fillna(combined_df[col].mode()[0], inplace=True)

train_df[target_col] = train_df[target_col].map({'Extrovert': 1, 'Introvert': 0})

X = combined_df.iloc[:len(train_df)].drop('id', axis=1)
X_test = combined_df.iloc[len(train_df):].drop('id', axis=1)
y = train_df[target_col]
print("✅ Preprocessing complete.")


# --- Model Training with LightGBM and Early Stopping ---
# CORRECTED: Reverted to the standard, more reliable callback system with the fix for the ValueError.
print("\n--- Training and Validating LightGBM Model ---")

# 1. Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")

# 2. Initialize the LGBMClassifier
# Note: We do NOT specify the metric here.
lgbm = lgb.LGBMClassifier(objective='binary',
                          n_estimators=2000,
                          learning_rate=0.01,
                          num_leaves=20,
                          max_depth=5,
                          seed=42,
                          n_jobs=-1,
                          verbose=-1,
                          colsample_bytree=0.7,
                          subsample=0.7)

# 3. Train with Early Stopping
print("⏳ Training model with early stopping...")
lgbm.fit(X_train, y_train,
         eval_set=[(X_val, y_val)],
         # FIX: Specify the evaluation metric here to resolve the callback error.
         eval_metric='accuracy',
         callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)])

print("✅ Model training complete.")

# 4. Evaluate the model on the validation set
y_pred_val = lgbm.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_val)

print(f"\nValidation Accuracy: {accuracy:.4f}")
print("\nValidation Classification Report:")
print(classification_report(y_val, y_pred_val, target_names=['Introvert', 'Extrovert']))


# --- Create Submission File ---
print("\n--- Generating Submission File ---")

# 1. Retrain the model on the ENTIRE training dataset
# We use the best iteration found during early stopping.
best_iteration = lgbm.best_iteration_
print(f"Retraining on full dataset with {best_iteration} estimators...")
lgbm_full = lgb.LGBMClassifier(objective='binary',
                               n_estimators=best_iteration, # Use the optimal number of trees
                               learning_rate=0.01,
                               num_leaves=20,
                               max_depth=5,
                               seed=42,
                               n_jobs=-1,
                               verbose=-1,
                               colsample_bytree=0.7,
                               subsample=0.7)

lgbm_full.fit(X, y)
print("✅ Model retrained on the full dataset.")

# 2. Make predictions on the official test data
test_predictions = lgbm_full.predict(X_test)

# 3. Format the predictions into the required submission format
test_predictions_labels = np.where(test_predictions == 1, 'Extrovert', 'Introvert')
submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': test_predictions_labels})

# 4. Save the submission file
submission_df.to_csv('submission_lgbm.csv', index=False)
print("\n✅ Submission file 'submission_lgbm.csv' created successfully!")
print("Good luck!")

# Display the first few rows of the submission file
print("\n--- First 5 Rows of Submission File ---")
print(submission_df.head())



import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- NOTE: This script uses the same preprocessing as before. ---
# --- For clarity, I'm including the preprocessing steps again. ---

# --- Load Data ---
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File paths are incorrect.")
    exit()

# --- Preprocessing ---
print("\n--- Running Preprocessing ---")
target_col = 'Personality'
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
numerical_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64'] and col not in ['id', target_col]]

combined_df = pd.concat([train_df.drop(target_col, axis=1), test_df], ignore_index=True)

for col in categorical_cols:
    combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0})

for col in numerical_cols:
    combined_df[col].fillna(combined_df[col].median(), inplace=True)
for col in categorical_cols:
    combined_df[col].fillna(combined_df[col].mode()[0], inplace=True)

train_df[target_col] = train_df[target_col].map({'Extrovert': 1, 'Introvert': 0})

X = combined_df.iloc[:len(train_df)].drop('id', axis=1)
X_test = combined_df.iloc[len(train_df):].drop('id', axis=1)
y = train_df[target_col]
print("✅ Preprocessing complete.")


# --- Hyperparameter Tuning with Optuna ---
print("\n--- Starting Hyperparameter Tuning with Optuna ---")

# 1. Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Define the objective function for Optuna
def objective(trial):
    """
    This function takes a 'trial' object from Optuna, which suggests hyperparameters,
    trains a model, and returns its accuracy. Optuna's goal is to maximize this value.
    """
    # Define the search space for the hyperparameters
    params = {
        'objective': 'binary',
        # FIX: Removed 'metric' from here to avoid conflict with eval_metric in .fit()
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 10, 40),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # Train the model with the suggested hyperparameters
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='accuracy',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    # Make predictions and return the accuracy
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    return accuracy

# 3. Create an Optuna study and run the optimization
# We want to maximize accuracy, so the direction is 'maximize'.
study = optuna.create_study(direction='maximize')
# We'll run 50 trials. More trials can yield better results but take longer.
study.optimize(objective, n_trials=50)

print("\n✅ Tuning complete!")
print(f"Best trial accuracy: {study.best_value:.4f}")
print("Best hyperparameters found:")
print(study.best_params)


# --- Create Submission File with Best Hyperparameters ---
print("\n--- Generating Submission File with Tuned Parameters ---")

# 1. Get the best hyperparameters from the study
best_params = study.best_params
best_params['objective'] = 'binary'
# FIX: Removed 'metric' from here as well.
best_params['seed'] = 42
best_params['n_jobs'] = -1
best_params['verbose'] = -1

# 2. Train the final model on the full dataset
# First, find the optimal number of estimators with early stopping
temp_model = lgb.LGBMClassifier(n_estimators=2000, **best_params)
temp_model.fit(X_train, y_train,
               eval_set=[(X_val, y_val)],
               eval_metric='accuracy',
               callbacks=[lgb.early_stopping(50, verbose=False)])
best_iteration = temp_model.best_iteration_

print(f"Retraining on full dataset with {best_iteration} estimators...")
final_model = lgb.LGBMClassifier(n_estimators=best_iteration, **best_params)
final_model.fit(X, y)
print("✅ Final model trained.")

# 3. Make predictions on the test data
test_predictions = final_model.predict(X_test)

# 4. Format and save the submission file
test_predictions_labels = np.where(test_predictions == 1, 'Extrovert', 'Introvert')
submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': test_predictions_labels})
submission_df.to_csv('submission_optuna.csv', index=False)
print("\n✅ Submission file 'submission_optuna.csv' created successfully!")



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- NOTE: This script uses the same preprocessing as before. ---
# --- For clarity, I'm including the preprocessing steps again. ---

# --- Load Data ---
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File paths are incorrect.")
    exit()

# --- Preprocessing ---
print("\n--- Running Preprocessing ---")
target_col = 'Personality'
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
numerical_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64'] and col not in ['id', target_col]]

combined_df = pd.concat([train_df.drop(target_col, axis=1), test_df], ignore_index=True)

for col in categorical_cols:
    combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0})

for col in numerical_cols:
    combined_df[col].fillna(combined_df[col].median(), inplace=True)
for col in categorical_cols:
    combined_df[col].fillna(combined_df[col].mode()[0], inplace=True)

train_df[target_col] = train_df[target_col].map({'Extrovert': 1, 'Introvert': 0})

X = combined_df.iloc[:len(train_df)].drop('id', axis=1)
X_test = combined_df.iloc[len(train_df):].drop('id', axis=1)
y = train_df[target_col]
print("✅ Preprocessing complete.")


# --- Model Training and Ensembling ---
print("\n--- Training and Ensembling Models ---")

# 1. Define the RandomForest model (from Step 3)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 2. Define the tuned LightGBM model (using the best params from your Optuna run)
# NOTE: Replace these with the actual best params from your last run's output if they are different.
lgbm_params = {
    'learning_rate': 0.01298592358888913,
    'num_leaves': 37,
    'max_depth': 8,
    'subsample': 0.9829744398242469,
    'colsample_bytree': 0.6833179024034633,
    'reg_alpha': 0.008919868691515938,
    'reg_lambda': 0.7390159976395304,
    'objective': 'binary',
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'n_estimators': 439 # A good number from previous runs
}
lgbm_model = lgb.LGBMClassifier(**lgbm_params)

# 3. Train both models on the full training data
print("⏳ Training RandomForest model...")
rf_model.fit(X, y)
print("✅ RandomForest model trained.")

print("⏳ Training LightGBM model...")
lgbm_model.fit(X, y)
print("✅ LightGBM model trained.")

# 4. Get prediction probabilities from both models
# We need the probability of the positive class ('Extrovert', which is 1)
print("⏳ Generating predictions...")
rf_probs = rf_model.predict_proba(X_test)[:, 1]
lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]

# 5. Create the ensemble prediction by averaging the probabilities
# You can experiment with different weights, but 50/50 is a great start.
ensemble_probs = (rf_probs * 0.5) + (lgbm_probs * 0.5)

# 6. Convert probabilities to final predictions (0 or 1)
ensemble_preds = (ensemble_probs > 0.5).astype(int)
print("✅ Ensemble predictions created.")


# --- Create Submission File ---
print("\n--- Generating Final Submission File ---")

# 1. Format the predictions into the required submission format
test_predictions_labels = np.where(ensemble_preds == 1, 'Extrovert', 'Introvert')
submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': test_predictions_labels})

# 2. Save the submission file
submission_df.to_csv('submission_ensemble.csv', index=False)
print("\n✅ Submission file 'submission_ensemble.csv' created successfully!")
print("This is our best shot yet. Good luck!")

# Display the first few rows of the submission file
print("\n--- First 5 Rows of Submission File ---")
print(submission_df.head())



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# --- NOTE: This script uses the same preprocessing as before. ---
# --- For clarity, I'm including the preprocessing steps again. ---

# --- Load Data ---
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File paths are incorrect.")
    exit()

# --- Preprocessing ---
print("\n--- Running Preprocessing ---")
target_col = 'Personality'
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
numerical_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64'] and col not in ['id', target_col]]

combined_df = pd.concat([train_df.drop(target_col, axis=1), test_df], ignore_index=True)

for col in categorical_cols:
    combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0})

for col in numerical_cols:
    combined_df[col].fillna(combined_df[col].median(), inplace=True)
for col in categorical_cols:
    combined_df[col].fillna(combined_df[col].mode()[0], inplace=True)

train_df[target_col] = train_df[target_col].map({'Extrovert': 1, 'Introvert': 0})

X = combined_df.iloc[:len(train_df)].drop('id', axis=1)
X_test = combined_df.iloc[len(train_df):].drop('id', axis=1)
y = train_df[target_col]
print("✅ Preprocessing complete.")


# --- K-Fold Ensemble Training ---
print("\n--- Training Ensemble with 5-Fold Cross-Validation ---")

# 1. Define models with the same parameters as before
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
lgbm_params = {
    'learning_rate': 0.01298592358888913, 'num_leaves': 37, 'max_depth': 8,
    'subsample': 0.9829744398242469, 'colsample_bytree': 0.6833179024034633,
    'reg_alpha': 0.008919868691515938, 'reg_lambda': 0.7390159976395304,
    'objective': 'binary', 'seed': 42, 'n_jobs': -1, 'verbose': -1, 'n_estimators': 439
}
lgbm_model = lgb.LGBMClassifier(**lgbm_params)

# 2. Set up K-Fold Cross-Validation
# StratifiedKFold ensures each fold has the same proportion of Introverts/Extroverts as the whole dataset.
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# 3. Create arrays to store predictions for the test set from each fold
rf_test_preds = []
lgbm_test_preds = []

# 4. Loop through each fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    
    # Train both models on the training data for this fold
    print("⏳ Training RandomForest...")
    rf_model.fit(X_train, y_train)
    
    print("⏳ Training LightGBM...")
    lgbm_model.fit(X_train, y_train)
    
    # Make predictions on the full test set and store them
    print("⏳ Generating predictions for the test set...")
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
    
    rf_test_preds.append(rf_probs)
    lgbm_test_preds.append(lgbm_probs)

print("\n✅ All folds complete.")

# 5. Average the predictions from all folds
avg_rf_preds = np.mean(rf_test_preds, axis=0)
avg_lgbm_preds = np.mean(lgbm_test_preds, axis=0)

# 6. Create the final ensemble prediction
ensemble_probs = (avg_rf_preds * 0.5) + (avg_lgbm_preds * 0.5)
ensemble_preds = (ensemble_probs > 0.5).astype(int)
print("✅ Final ensemble predictions created by averaging across all folds.")


# --- Create Submission File ---
print("\n--- Generating Final Submission File ---")

test_predictions_labels = np.where(ensemble_preds == 1, 'Extrovert', 'Introvert')
submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': test_predictions_labels})
submission_df.to_csv('submission_kfold_ensemble.csv', index=False)

print("\n✅ Submission file 'submission_kfold_ensemble.csv' created successfully!")
print("This is the most robust model we've built. Let's see how it does!")



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

# --- NOTE: This script uses the same preprocessing as before. ---
# --- For clarity, I'm including the preprocessing steps again. ---

# --- Load Data ---
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File paths are incorrect.")
    exit()

# --- Preprocessing ---
print("\n--- Running Preprocessing ---")
target_col = 'Personality'
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
numerical_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64'] and col not in ['id', target_col]]

combined_df = pd.concat([train_df.drop(target_col, axis=1), test_df], ignore_index=True)

for col in categorical_cols:
    combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0})

for col in numerical_cols:
    combined_df[col].fillna(combined_df[col].median(), inplace=True)
for col in categorical_cols:
    combined_df[col].fillna(combined_df[col].mode()[0], inplace=True)

train_df[target_col] = train_df[target_col].map({'Extrovert': 1, 'Introvert': 0})

X = combined_df.iloc[:len(train_df)].drop('id', axis=1)
X_test = combined_df.iloc[len(train_df):].drop('id', axis=1)
y = train_df[target_col]
print("✅ Preprocessing complete.")


# --- Weighted Ensemble Training ---
print("\n--- Training Weighted Ensemble Models ---")

# 1. Define the RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 2. Define the tuned LightGBM model
# Using the same proven parameters from our previous best model
lgbm_params = {
    'learning_rate': 0.01298592358888913, 'num_leaves': 37, 'max_depth': 8,
    'subsample': 0.9829744398242469, 'colsample_bytree': 0.6833179024034633,
    'reg_alpha': 0.008919868691515938, 'reg_lambda': 0.7390159976395304,
    'objective': 'binary', 'seed': 42, 'n_jobs': -1, 'verbose': -1, 'n_estimators': 439
}
lgbm_model = lgb.LGBMClassifier(**lgbm_params)

# 3. Train both models on the full training data
print("⏳ Training RandomForest model...")
rf_model.fit(X, y)
print("✅ RandomForest model trained.")

print("⏳ Training LightGBM model...")
lgbm_model.fit(X, y)
print("✅ LightGBM model trained.")

# 4. Get prediction probabilities from both models
print("⏳ Generating predictions...")
rf_probs = rf_model.predict_proba(X_test)[:, 1]
lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]

# 5. Create the WEIGHTED ensemble prediction
# We give more weight to the stronger model (LGBM)
lgbm_weight = 0.65
rf_weight = 0.35
ensemble_probs = (lgbm_probs * lgbm_weight) + (rf_probs * rf_weight)

# 6. Convert probabilities to final predictions
ensemble_preds = (ensemble_probs > 0.5).astype(int)
print(f"✅ Weighted ensemble predictions created ({lgbm_weight*100}% LGBM, {rf_weight*100}% RF).")


# --- Create Submission File ---
print("\n--- Generating Final Submission File ---")

test_predictions_labels = np.where(ensemble_preds == 1, 'Extrovert', 'Introvert')
submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': test_predictions_labels})
submission_df.to_csv('submission_weighted_ensemble.csv', index=False)

print("\n✅ Submission file 'submission_weighted_ensemble.csv' created successfully!")
print("Let's see if this refinement gets us closer to the top!")



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

# --- NOTE: This script uses the same preprocessing as before. ---
# --- For clarity, I'm including the preprocessing steps again. ---

# --- Load Data ---
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File paths are incorrect.")
    exit()

# --- Preprocessing ---
print("\n--- Running Preprocessing ---")
target_col = 'Personality'
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
numerical_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64'] and col not in ['id', target_col]]

combined_df = pd.concat([train_df.drop(target_col, axis=1), test_df], ignore_index=True)

for col in categorical_cols:
    combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0})

# Impute missing values BEFORE creating new features
for col in numerical_cols:
    combined_df[col].fillna(combined_df[col].median(), inplace=True)
for col in categorical_cols:
    combined_df[col].fillna(combined_df[col].mode()[0], inplace=True)

# --- Feature Engineering ---
print("\n--- Creating New Features ---")

# 1. Social Engagement Score
combined_df['Social_Engagement'] = combined_df['Social_event_attendance'] + combined_df['Going_outside'] + combined_df['Post_frequency']

# 2. Alone vs. Social Ratio
# Add 1 to the denominator to avoid division by zero errors
combined_df['Alone_Ratio'] = combined_df['Time_spent_Alone'] / (combined_df['Social_event_attendance'] + combined_df['Going_outside'] + 1)

# 3. Social Burnout Index
combined_df['Social_Burnout'] = combined_df['Drained_after_socializing'] * combined_df['Social_event_attendance']

print("✅ New features created.")


# --- Final Data Preparation ---
train_df[target_col] = train_df[target_col].map({'Extrovert': 1, 'Introvert': 0})

X = combined_df.iloc[:len(train_df)].drop('id', axis=1)
X_test = combined_df.iloc[len(train_df):].drop('id', axis=1)
y = train_df[target_col]
print("✅ Preprocessing and feature engineering complete.")


# --- Simple Ensemble Training (Our Best Model Architecture) ---
print("\n--- Training Simple Ensemble on Enriched Data ---")

# 1. Define the models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
lgbm_params = {
    'learning_rate': 0.01298592358888913, 'num_leaves': 37, 'max_depth': 8,
    'subsample': 0.9829744398242469, 'colsample_bytree': 0.6833179024034633,
    'reg_alpha': 0.008919868691515938, 'reg_lambda': 0.7390159976395304,
    'objective': 'binary', 'seed': 42, 'n_jobs': -1, 'verbose': -1, 'n_estimators': 439
}
lgbm_model = lgb.LGBMClassifier(**lgbm_params)

# 2. Train both models on the full, enriched training data
print("⏳ Training RandomForest model...")
rf_model.fit(X, y)
print("✅ RandomForest model trained.")

print("⏳ Training LightGBM model...")
lgbm_model.fit(X, y)
print("✅ LightGBM model trained.")

# 3. Get prediction probabilities from both models
print("⏳ Generating predictions...")
rf_probs = rf_model.predict_proba(X_test)[:, 1]
lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]

# 4. Create the 50/50 ensemble prediction
ensemble_probs = (rf_probs * 0.5) + (lgbm_probs * 0.5)
ensemble_preds = (ensemble_probs > 0.5).astype(int)
print("✅ Ensemble predictions created.")


# --- Create Submission File ---
print("\n--- Generating Final Submission File ---")

test_predictions_labels = np.where(ensemble_preds == 1, 'Extrovert', 'Introvert')
submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': test_predictions_labels})
submission_df.to_csv('submission_feature_engineered.csv', index=False)

print("\n✅ Submission file 'submission_feature_engineered.csv' created successfully!")
print("Let's see if our new features pushed us to the top!")



import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# --- Load Data & Preprocessing ---
# (Using the same proven preprocessing steps)
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File paths are incorrect.")
    exit()

print("\n--- Running Preprocessing ---")
target_col = 'Personality'
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
numerical_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64'] and col not in ['id', target_col]]
combined_df = pd.concat([train_df.drop(target_col, axis=1), test_df], ignore_index=True)
for col in categorical_cols:
    combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0})
for col in numerical_cols:
    combined_df[col].fillna(combined_df[col].median(), inplace=True)
for col in categorical_cols:
    combined_df[col].fillna(combined_df[col].mode()[0], inplace=True)
train_df[target_col] = train_df[target_col].map({'Extrovert': 1, 'Introvert': 0})
X = combined_df.iloc[:len(train_df)].drop('id', axis=1)
X_test = combined_df.iloc[len(train_df):].drop('id', axis=1)
y = train_df[target_col]
print("✅ Preprocessing complete.")


# --- Stacking Ensemble Training with 3 Models ---
print("\n--- Training 3-Model Stacking Ensemble ---")

# 1. Define the base models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

lgbm_params = {
    'learning_rate': 0.0129, 'num_leaves': 37, 'max_depth': 8,
    'subsample': 0.9829, 'colsample_bytree': 0.6833,
    'reg_alpha': 0.0089, 'reg_lambda': 0.7390,
    'objective': 'binary', 'seed': 42, 'n_jobs': -1, 'verbose': -1, 'n_estimators': 439
}
lgbm_model = lgb.LGBMClassifier(**lgbm_params)

# Add XGBoost, inspired by the high-scoring notebook
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              use_label_encoder=False, seed=42, n_estimators=500,
                              learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8)

# Define the meta-model
meta_model = LogisticRegression(solver='liblinear')

# 2. Set up K-Fold Cross-Validation
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Create arrays to store predictions for all three models
oof_preds_rf = np.zeros(len(X))
oof_preds_lgbm = np.zeros(len(X))
oof_preds_xgb = np.zeros(len(X))
test_preds_rf = np.zeros(len(X_test))
test_preds_lgbm = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

# 3. Loop through each fold to generate the meta-features
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Train and predict with each base model
    print("Training RF...")
    rf_model.fit(X_train, y_train)
    oof_preds_rf[val_idx] = rf_model.predict_proba(X_val)[:, 1]
    test_preds_rf += rf_model.predict_proba(X_test)[:, 1] / N_SPLITS

    print("Training LGBM...")
    lgbm_model.fit(X_train, y_train)
    oof_preds_lgbm[val_idx] = lgbm_model.predict_proba(X_val)[:, 1]
    test_preds_lgbm += lgbm_model.predict_proba(X_test)[:, 1] / N_SPLITS

    print("Training XGB...")
    xgb_model.fit(X_train, y_train)
    oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_preds_xgb += xgb_model.predict_proba(X_test)[:, 1] / N_SPLITS

print("\n✅ Meta-feature generation complete.")

# 4. Create the new training data for the meta-model
X_meta_train = pd.DataFrame({
    'rf_pred': oof_preds_rf,
    'lgbm_pred': oof_preds_lgbm,
    'xgb_pred': oof_preds_xgb
})

# 5. Train the meta-model
print("⏳ Training meta-model...")
meta_model.fit(X_meta_train, y)
print("✅ Meta-model trained.")

# 6. Create the new test data for the meta-model
X_meta_test = pd.DataFrame({
    'rf_pred': test_preds_rf,
    'lgbm_pred': test_preds_lgbm,
    'xgb_pred': test_preds_xgb
})

# 7. Make final predictions
final_predictions = meta_model.predict(X_meta_test)


# --- Create Submission File ---
print("\n--- Generating Final Submission File ---")

test_predictions_labels = np.where(final_predictions == 1, 'Extrovert', 'Introvert')
submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': test_predictions_labels})
submission_df.to_csv('submission_stacking_3model.csv', index=False)

print("\n✅ Submission file 'submission_stacking_3model.csv' created successfully!")
print("This is our strongest play yet, combining our work with insights from the top notebook. Good luck!")



import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# --- Load Data & Preprocessing ---
# (Using the same proven preprocessing steps)
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File paths are incorrect.")
    exit()

print("\n--- Running Preprocessing ---")
target_col = 'Personality'
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
numerical_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64'] and col not in ['id', target_col]]
combined_df = pd.concat([train_df.drop(target_col, axis=1), test_df], ignore_index=True)
for col in categorical_cols:
    combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0})
for col in numerical_cols:
    combined_df[col].fillna(combined_df[col].median(), inplace=True)
for col in categorical_cols:
    combined_df[col].fillna(combined_df[col].mode()[0], inplace=True)
train_df[target_col] = train_df[target_col].map({'Extrovert': 1, 'Introvert': 0})
X = combined_df.iloc[:len(train_df)].drop('id', axis=1)
X_test = combined_df.iloc[len(train_df):].drop('id', axis=1)
y = train_df[target_col]
print("✅ Preprocessing complete.")


# --- 4-Model Blended Ensemble Training ---
print("\n--- Training 4-Model Blended Ensemble ---")

# 1. Define the four base models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

lgbm_params = {
    'learning_rate': 0.0129, 'num_leaves': 37, 'max_depth': 8,
    'subsample': 0.9829, 'colsample_bytree': 0.6833,
    'reg_alpha': 0.0089, 'reg_lambda': 0.7390,
    'objective': 'binary', 'seed': 42, 'n_jobs': -1, 'verbose': -1, 'n_estimators': 439
}
lgbm_model = lgb.LGBMClassifier(**lgbm_params)

xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              use_label_encoder=False, seed=42, n_estimators=500,
                              learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8)

# Add the ExtraTreesClassifier
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 2. Train all four models on the full dataset
print("⏳ Training RandomForest...")
rf_model.fit(X, y)
print("✅ RF trained.")

print("⏳ Training LightGBM...")
lgbm_model.fit(X, y)
print("✅ LGBM trained.")

print("⏳ Training XGBoost...")
xgb_model.fit(X, y)
print("✅ XGB trained.")

print("⏳ Training ExtraTrees...")
et_model.fit(X, y)
print("✅ ET trained.")

# 3. Get prediction probabilities from all models
print("\n⏳ Generating predictions from all models...")
rf_probs = rf_model.predict_proba(X_test)[:, 1]
lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
et_probs = et_model.predict_proba(X_test)[:, 1]

# 4. Create the final blend by averaging the probabilities
# Each model gets an equal 25% vote
ensemble_probs = (rf_probs + lgbm_probs + xgb_probs + et_probs) / 4.0

# 5. Convert probabilities to final predictions
final_predictions = (ensemble_probs > 0.5).astype(int)
print("✅ Final predictions created.")


# --- Create Submission File ---
print("\n--- Generating Final Submission File ---")

test_predictions_labels = np.where(final_predictions == 1, 'Extrovert', 'Introvert')
submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': test_predictions_labels})
submission_df.to_csv('submission_4_model_blend.csv', index=False)

print("\n✅ Submission file 'submission_4_model_blend.csv' created successfully!")
print("This is our strongest and most diverse blend yet. Let's see if it takes the top spot!")



import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# --- Load Data & Preprocessing ---
# (Using the same proven preprocessing steps)
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File paths are incorrect.")
    exit()

print("\n--- Running Preprocessing ---")
target_col = 'Personality'
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
numerical_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64'] and col not in ['id', target_col]]
combined_df = pd.concat([train_df.drop(target_col, axis=1), test_df], ignore_index=True)
for col in categorical_cols:
    combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0})
for col in numerical_cols:
    combined_df[col].fillna(combined_df[col].median(), inplace=True)
for col in categorical_cols:
    combined_df[col].fillna(combined_df[col].mode()[0], inplace=True)
train_df[target_col] = train_df[target_col].map({'Extrovert': 1, 'Introvert': 0})
X = combined_df.iloc[:len(train_df)].drop('id', axis=1)
X_test = combined_df.iloc[len(train_df):].drop('id', axis=1)
y = train_df[target_col]
print("✅ Preprocessing complete.")


# --- 4-Model Blended Ensemble with Refined Hyperparameters ---
print("\n--- Training 4-Model Blended Ensemble with Refined Hyperparameters ---")

# 1. Define the four base models with parameters inspired by the high-scoring notebook
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                  max_depth=10, min_samples_leaf=2)

lgbm_params = {
    'objective': 'binary', 'metric': 'accuracy', 'random_state': 42,
    'n_estimators': 500, 'learning_rate': 0.01, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0.1,
    'lambda_l2': 0.1, 'num_leaves': 31, 'verbose': -1, 'n_jobs': -1
}
lgbm_model = lgb.LGBMClassifier(**lgbm_params)

# CORRECTED: Fixed the syntax error in the keyword arguments
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              use_label_encoder=False, seed=42, n_estimators=500,
                              learning_rate=0.01, max_depth=5,
                              subsample=0.8, colsample_bytree=0.8)

et_model = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                max_depth=10, min_samples_leaf=2)

# 2. Train all four models on the full dataset
print("⏳ Training RandomForest...")
rf_model.fit(X, y)
print("✅ RF trained.")

print("⏳ Training LightGBM...")
lgbm_model.fit(X, y)
print("✅ LGBM trained.")

print("⏳ Training XGBoost...")
xgb_model.fit(X, y)
print("✅ XGB trained.")

print("⏳ Training ExtraTrees...")
et_model.fit(X, y)
print("✅ ET trained.")

# 3. Get prediction probabilities from all models
print("\n⏳ Generating predictions from all models...")
rf_probs = rf_model.predict_proba(X_test)[:, 1]
lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
et_probs = et_model.predict_proba(X_test)[:, 1]

# 4. Create the final blend by averaging the probabilities
ensemble_probs = (rf_probs + lgbm_probs + xgb_probs + et_probs) / 4.0

# 5. Convert probabilities to final predictions
final_predictions = (ensemble_probs > 0.5).astype(int)
print("✅ Final predictions created.")


# --- Create Submission File ---
print("\n--- Generating Final Submission File ---")

test_predictions_labels = np.where(final_predictions == 1, 'Extrovert', 'Introvert')
submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': test_predictions_labels})
submission_df.to_csv('submission_refined_4_blend.csv', index=False)

print("\n✅ Submission file 'submission_refined_4_blend.csv' created successfully!")
print("Let's see if these new parameters make the difference!")



import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# --- Load Data & Preprocessing ---
# (Using the same proven preprocessing steps)
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File paths are incorrect.")
    exit()

print("\n--- Running Preprocessing ---")
target_col = 'Personality'
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
numerical_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64'] and col not in ['id', target_col]]
combined_df = pd.concat([train_df.drop(target_col, axis=1), test_df], ignore_index=True)
for col in categorical_cols:
    combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0})
for col in numerical_cols:
    combined_df[col].fillna(combined_df[col].median(), inplace=True)
for col in categorical_cols:
    combined_df[col].fillna(combined_df[col].mode()[0], inplace=True)
train_df[target_col] = train_df[target_col].map({'Extrovert': 1, 'Introvert': 0})
X = combined_df.iloc[:len(train_df)].drop('id', axis=1)
X_test = combined_df.iloc[len(train_df):].drop('id', axis=1)
y = train_df[target_col]
print("✅ Preprocessing complete.")


# --- Optimized Weighted 4-Model Blend ---
print("\n--- Training Optimized Weighted 4-Model Blend ---")

# 1. Define the four base models using the parameters from our best simple blend
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

lgbm_params = {
    'learning_rate': 0.0129, 'num_leaves': 37, 'max_depth': 8,
    'subsample': 0.9829, 'colsample_bytree': 0.6833,
    'reg_alpha': 0.0089, 'reg_lambda': 0.7390,
    'objective': 'binary', 'seed': 42, 'n_jobs': -1, 'verbose': -1, 'n_estimators': 439
}
lgbm_model = lgb.LGBMClassifier(**lgbm_params)

xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              use_label_encoder=False, seed=42, n_estimators=500,
                              learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8)

et_model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)


# 2. Train all four models on the full dataset
print("⏳ Training RandomForest...")
rf_model.fit(X, y)
print("✅ RF trained.")

print("⏳ Training LightGBM...")
lgbm_model.fit(X, y)
print("✅ LGBM trained.")

print("⏳ Training XGBoost...")
xgb_model.fit(X, y)
print("✅ XGB trained.")

print("⏳ Training ExtraTrees...")
et_model.fit(X, y)
print("✅ ET trained.")

# 3. Get prediction probabilities from all models
print("\n⏳ Generating predictions from all models...")
rf_probs = rf_model.predict_proba(X_test)[:, 1]
lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
et_probs = et_model.predict_proba(X_test)[:, 1]

# 4. Create the final blend using optimized weights
# Give more weight to the gradient boosting models
weights = {'lgbm': 0.3, 'xgb': 0.3, 'rf': 0.2, 'et': 0.2}
ensemble_probs = (lgbm_probs * weights['lgbm'] +
                  xgb_probs * weights['xgb'] +
                  rf_probs * weights['rf'] +
                  et_probs * weights['et'])

# 5. Convert probabilities to final predictions
final_predictions = (ensemble_probs > 0.5).astype(int)
print("✅ Final predictions created with optimized weights.")


# --- Create Submission File ---
print("\n--- Generating Final Submission File ---")

test_predictions_labels = np.where(final_predictions == 1, 'Extrovert', 'Introvert')
submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': test_predictions_labels})
submission_df.to_csv('submission_optimized_weighted_blend.csv', index=False)

print("\n✅ Submission file 'submission_optimized_weighted_blend.csv' created successfully!")
print("This is our final refinement. Let's see if it pays off!")


