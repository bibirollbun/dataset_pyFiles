import pandas as pd

# Define the path to the dataset
path_to_ds = '/kaggle/input/dig4bio-v1/'

# Load the submission files
subm_0 = pd.read_csv(path_to_ds + "submission_1_ridge_stack.csv")
subm_1 = pd.read_csv(path_to_ds + "submission_2_linear_stack.csv") # Score: 0.60906
subm_2 = pd.read_csv(path_to_ds + "submission_3_rf_stack.csv")
subm_3 = pd.read_csv(path_to_ds + "submission 0.60906.csv") # Note: Application does not match!??
subm_4 = pd.read_csv(path_to_ds + "submission 0.37957.csv")
subm_5 = pd.read_csv(path_to_ds + "submission 0.26361.csv")

# Load the sample submission file
subm = pd.read_csv("/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/sample_submission.csv")

# Combine predictions using weighted averages/
# Experiment with different weights to find the best ensemble performance

# # Example weight combination (commented out)
subm['Glucose'] = 0.0100 * subm_0['Glucose'] + 0.0100 * subm_1['Glucose'] + 0.0100 * subm_2['Glucose'] + 0.9500 * subm_3['Glucose'] + 0.0100 * subm_4['Glucose'] + 0.0100 * subm_5['Glucose']
subm['Sodium Acetate'] = 0.0100 * subm_0['Sodium Acetate'] + 0.0100 * subm_1['Sodium Acetate'] + 0.0100 * subm_2['Sodium Acetate'] + 0.9500 * subm_3['Sodium Acetate'] + 0.0100 * subm_4['Sodium Acetate'] + 0.0100 * subm_5['Sodium Acetate']
subm['Magnesium Sulfate'] = 0.0100 * subm_0['Magnesium Sulfate'] + 0.0100 * subm_1['Magnesium Sulfate'] + 0.0100 * subm_2['Magnesium Sulfate'] + 0.9500 * subm_3['Magnesium Sulfate'] + 0.0100 * subm_4['Magnesium Sulfate'] + 0.0100 * subm_5['Magnesium Sulfate']

# Final weight combination
subm['Glucose'] = 1.000 * subm_1['Glucose'] - 0.0017 * subm_4['Glucose'] + 0.0017 * subm_5['Glucose']
subm['Sodium Acetate'] = 1.000 * subm_1['Sodium Acetate'] - 0.0017 * subm_4['Sodium Acetate'] + 0.0017 * subm_5['Sodium Acetate']
subm['Magnesium Sulfate'] = 1.000 * subm_1['Magnesium Sulfate'] - 0.0017 * subm_4['Magnesium Sulfate'] + 0.0017 * subm_5['Magnesium Sulfate']

# Save the final submission to a CSV file
subm.to_csv('submission.csv', index=False)

# Display the first 8 rows of the final submission
print(subm.head(8))

