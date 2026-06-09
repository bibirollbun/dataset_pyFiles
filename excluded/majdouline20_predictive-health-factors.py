# %% Imports and Configurations
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# H2O for modeling
import h2o
from h2o.estimators import H2OGradientBoostingEstimator, H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch

# Configure pandas to display all columns
pd.set_option('display.max_columns', None)

# Set random seed for reproducibility
RANDOM_SEED = 111

# Aesthetic configurations (colors for plots)
COLOR_PRIMARY   = 'darkblue'
COLOR_SECONDARY = 'darkgreen'
COLOR_TERTIARY  = 'darkred'



# %% File Listing and Data Loading
# Define the input directory and list files for verification
input_dir = '/kaggle/input/exploring-predictive-health-factors'
files = os.listdir(input_dir)
print("Files in input directory:", files)

# Define file paths
train_path      = os.path.join(input_dir, 'train.csv')
test_path       = os.path.join(input_dir, 'test.csv')
submission_path = os.path.join(input_dir, 'sample_submission.csv')

# Load datasets
df_train = pd.read_csv(train_path, low_memory=False)
df_test  = pd.read_csv(test_path, low_memory=False)
df_sub   = pd.read_csv(submission_path)



# %% Data Overview and Minimal Preprocessing
# Display data summaries and check for missing values
print("Training Data Info:")
df_train.info(show_counts=True)
print("\nTest Data Info:")
df_test.info(show_counts=True)

# Summarize missing values in training data
missing_train = df_train.isnull().sum()
print("\nMissing Values in Training Data:")
print(missing_train[missing_train > 0])

# Standardize categorical text (e.g., strip extra spaces and convert to lowercase)
categorical_columns = df_train.select_dtypes(include='object').columns.tolist()
for col in categorical_columns:
    df_train[col] = df_train[col].str.strip().str.lower()
    if col in df_test.columns:
        df_test[col] = df_test[col].str.strip().str.lower()


# %% Feature Engineering
# Create new features or transform existing ones.
# For example: extract numeric midpoints from age ranges (e.g., '20-25' -> 22.5)

def extract_age_midpoint(age_str):
    try:
        parts = age_str.split('-')
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2.0
    except Exception:
        return np.nan

if 'age' in df_train.columns:
    df_train['age_midpoint'] = df_train['age'].apply(extract_age_midpoint)
    if 'age' in df_test.columns:
        df_test['age_midpoint'] = df_test['age'].apply(extract_age_midpoint)


# %% H2O Initialization and Data Conversion
# Initialize H2O cluster and convert pandas DataFrames to H2O Frames
h2o.init(max_mem_size='16G', nthreads=4)

train_h2o = h2o.H2OFrame(df_train)
test_h2o  = h2o.H2OFrame(df_test)

# Define target and predictors.
# We assume the target column is "pcos" (converted to lowercase during preprocessing).
target = 'PCOS'
# Exclude the target and any identifier columns (e.g., 'id') from predictors.
exclude_columns = ['id', target]
predictors = [col for col in train_h2o.col_names if col not in exclude_columns]

# Force the target variable to be categorical
train_h2o[target] = train_h2o[target].asfactor()

print("Predictors used for modeling:", predictors)


# %% Hyperparameter Tuning and Model Training

# Define a hyperparameter grid for the GBM model.
hyper_params = {
    'ntrees': [10, 20, 30],
    'max_depth': [3, 4, 5],
    'learn_rate': [0.05, 0.1, 0.2]
}

# Define search criteria.
search_criteria = {'strategy': "Cartesian"}

# Initialize the base GBM model with desired fixed parameters, including the seed.
base_gbm = H2OGradientBoostingEstimator(
    seed=12345,  # Set the random seed here
    stopping_rounds=5,
    stopping_metric='auc',
    stopping_tolerance=0.0001,
    score_each_iteration=True
)

# Initialize grid search without the seed (and nfolds) arguments.
grid = H2OGridSearch(
    model=base_gbm,
    hyper_params=hyper_params,
    search_criteria=search_criteria,
    grid_id='gbm_grid'
)

# Train the grid search, passing nfolds as an argument to train().
import time
start_time = time.time()
grid.train(x=predictors, y=target, training_frame=train_h2o, nfolds=5)
elapsed_time = time.time() - start_time
print(f"Grid search training completed in {elapsed_time:.2f} seconds")

# Retrieve the best model based on cross-validation AUC.
sorted_grid = grid.get_grid(sort_by='auc', decreasing=True)
best_model = sorted_grid.models[0]
print("Best model hyperparameters:", best_model.params)


# %% Model Evaluation
# Evaluate the performance of the best model from the grid search

print("Best Model Summary:")
best_model.summary()

print("\nCross-Validation Metrics Summary:")
print(best_model.cross_validation_metrics_summary())

# Plot variable importance (top 20 features)
best_model.varimp_plot(20)
plt.tight_layout()
plt.show()

# Optionally, plot the scoring history for each CV fold
cv_models = best_model.cross_validation_models()
if cv_models:
    for i, cv_model in enumerate(cv_models):
        cv_score_history = cv_model.score_history()
        plt.figure(figsize=(8, 4))
        plt.scatter(cv_score_history['number_of_trees'], cv_score_history['training_auc'],
                    color='blue', label='Training AUC')
        plt.scatter(cv_score_history['number_of_trees'], cv_score_history['validation_auc'],
                    color='darkorange', label='Validation AUC')
        plt.title(f"CV Fold {i+1} - Scoring History [AUC]")
        plt.xlabel("Number of Trees")
        plt.ylabel("AUC")
        plt.ylim(0.7, 1.0)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# %% Predictions and Submission Generation
# Generate predictions on the test set using the best GBM model

test_predictions = best_model.predict(test_h2o).as_data_frame()

# Append prediction probabilities and binary predictions to the test DataFrame.
# Here, we assume the positive class is labeled "Yes" (adjust if necessary).
df_test['pred_GBM'] = test_predictions['yes']
df_test['pred_GBM_binary'] = test_predictions['predict']

# Visualize the distribution of predicted probabilities for the positive class
plt.figure(figsize=(8, 3))
df_test['pred_GBM'].plot(kind='hist', bins=15, color=COLOR_SECONDARY, edgecolor='black')
plt.title("Best GBM Model - Test Predictions (Probability of 'yes')")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Create the submission file using the sample submission template
df_submission = df_sub.copy()
df_submission[target] = df_test['pred_GBM']
submission_file = 'submission_best_gbm.csv'
df_submission.to_csv(submission_file, index=False)
print(f"Submission file saved as: {submission_file}")
print("Submission preview:")
print(df_submission.head(10))

