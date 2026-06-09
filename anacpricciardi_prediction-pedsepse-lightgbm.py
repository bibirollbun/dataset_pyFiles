# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, f1_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
path = "/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/"

# Load individual datasets
# Each dataset represents different aspects of patient data (devices, drug exposure, lab measurements, etc.)
devices_train = pd.read_csv(path + "devices_train.csv")
drugexposure_train = pd.read_csv(path + "drugsexposure_train.csv")
measurements_lab_train = pd.read_csv(path + "measurement_lab_train.csv")
observations_train = pd.read_csv(path + "observation_train.csv")
person_demographics = pd.read_csv(path + "person_demographics_episode_train.csv")
outcome_train = pd.read_csv(path + "SepsisLabel_train.csv")

# Helper Functions
def aggregate_time_series(data, time_col, group_col, agg_features):
    """Aggregate time-series data for feature generation."""
    # Ensure numeric columns are properly converted
    numeric_cols = [col for col, agg in agg_features.items() if 'mean' in agg or 'std' in agg or 'min' in agg or 'max']
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    agg_data = data.groupby(group_col).agg(agg_features)
    agg_data.columns = ["_".join(col).strip() for col in agg_data.columns.values]  # Flatten MultiIndex
    return agg_data.reset_index()




# Feature Engineering
# Define aggregation dictionary for observations
agg_dict_observations = {
    'valuefilled': ['mean', 'std', 'min', 'max'],  # Statistics on measured values
    'observation_concept_id': ['nunique']  # Count of unique observation types
}

# Convert 'valuefilled' column to numeric and handle non-numeric values
observations_train['valuefilled'] = pd.to_numeric(observations_train['valuefilled'], errors='coerce')

# Aggregate time-series observations at patient level
observations_agg = aggregate_time_series(observations_train, 'observation_datetime', 'person_id', agg_dict_observations)

# Process demographic data (e.g., age, gender)
demographics_numeric = person_demographics.select_dtypes(include=[np.number])
demographics_numeric['person_id'] = person_demographics['person_id']
demographics_agg = demographics_numeric.groupby('person_id').mean().reset_index()

# Select outcome labels (sepsis or not)
outcome_data = outcome_train[['person_id', 'SepsisLabel']]

# Merge datasets into a single DataFrame for training
combined_data = outcome_data.merge(observations_agg, on='person_id', how='left')
combined_data = combined_data.merge(demographics_agg, on='person_id', how='left')
combined_data.fillna(0, inplace=True)  # Fill missing values with 0


# Train-Test Split
# Separate features (X) and labels (y)
X = combined_data.drop(columns=['person_id', 'SepsisLabel'])
y = combined_data['SepsisLabel']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
# Initialize LightGBM model with basic parameters
model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metric='pr_auc', n_estimators=100, random_state=42)

# Train the model with early stopping to prevent overfitting
model.fit(
    X_train, 
    y_train, 
    eval_set=[(X_val, y_val)],  # Use validation set for evaluation
    eval_metric='auc',  # Metric to evaluate (AUC)
    callbacks=[lgb.early_stopping(stopping_rounds=10)]  # Stop if performance doesn't improve in 10 rounds
)


# Predictions and Metrics
# Generate predictions (probabilities) for the validation set
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)

# Calculate area under the precision-recall curve
pr_auc = auc(recall, precision)

# Calculate additional metrics
accuracy = accuracy_score(y_val, (y_pred_proba > 0.5).astype(int))  # Accuracy at threshold 0.5
f1 = f1_score(y_val, (y_pred_proba > 0.5).astype(int))  # F1 Score

# Print evaluation metrics
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")




# Visualizations
# Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()

# Feature Importance
# Plot feature importance to interpret model
lgb.plot_importance(model, max_num_features=3, importance_type='gain', figsize=(10, 6))
plt.title("TopFeature Importances")
plt.show()


