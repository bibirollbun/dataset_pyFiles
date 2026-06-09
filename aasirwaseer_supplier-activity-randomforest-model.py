# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load the training data
train_df = pd.read_csv('/kaggle/input/desafo-clasificacin-de-proveedores-activos/train_202201-202402.csv')

# Display the first few rows
print(train_df.head())

# Check the shape of the data
print(f"Shape of the training data: {train_df.shape}")

# Check data types and missing values
print(train_df.info())

# Check basic statistics
print(train_df.describe())



# Load the sample submission file
sample_submission = pd.read_csv('/kaggle/input/desafo-clasificacin-de-proveedores-activos/sample_submission.csv')

# Display the first few rows
print(sample_submission.head())

# Check the shape of the submission file
print(f"Shape of the sample submission: {sample_submission.shape}")



# Convert date columns to datetime
train_df['FECHA_FORMALIZACIÓN'] = pd.to_datetime(train_df['FECHA_FORMALIZACIÓN'])
train_df['FECHA_ÚLTIMO_ESTADO'] = pd.to_datetime(train_df['FECHA_ÚLTIMO_ESTADO'])

# Extract month and year from FECHA_FORMALIZACIÓN to create CODMES
train_df['CODMES'] = train_df['FECHA_FORMALIZACIÓN'].dt.strftime('%Y%m')

# Create the CODMES_RUC_PROVEEDOR column as in the submission format
train_df['CODMES_RUC_PROVEEDOR'] = train_df['CODMES'] + '_' + train_df['RUC_PROVEEDOR'].astype(str)

# Check the unique months in the dataset
print("Unique months in the dataset:")
print(train_df['CODMES'].unique())

# Count orders by month
monthly_orders = train_df.groupby('CODMES').size()
print("\nNumber of orders by month:")
print(monthly_orders)

# Count unique suppliers by month
monthly_suppliers = train_df.groupby('CODMES')['RUC_PROVEEDOR'].nunique()
print("\nNumber of unique suppliers by month:")
print(monthly_suppliers)


# Create a dataframe with unique supplier-month combinations
supplier_month_df = train_df.groupby(['CODMES', 'RUC_PROVEEDOR']).size().reset_index(name='order_count')

# Create a pivot table to see which suppliers received orders in which months
pivot_df = supplier_month_df.pivot_table(
    index='RUC_PROVEEDOR', 
    columns='CODMES', 
    values='order_count',
    fill_value=0
)

# Convert to binary (1 if supplier received at least one order in that month, 0 otherwise)
pivot_binary_df = (pivot_df > 0).astype(int)

# Display the first few rows
print("\nSupplier-Month Order Matrix (1 = received order, 0 = no order):")
print(pivot_binary_df.head())

# Get all unique suppliers
all_suppliers = train_df['RUC_PROVEEDOR'].unique()
print(f"\nTotal number of unique suppliers: {len(all_suppliers)}")

# Get all unique months
all_months = sorted(train_df['CODMES'].unique())
print(f"\nTotal number of unique months: {len(all_months)}")


# Calculate how many months each supplier was active
supplier_activity = pivot_binary_df.sum(axis=1).reset_index()
supplier_activity.columns = ['RUC_PROVEEDOR', 'active_months']

print("\nSupplier activity distribution:")
print(supplier_activity['active_months'].value_counts().sort_index())

# Plot the distribution
plt.figure(figsize=(12, 6))
sns.histplot(supplier_activity['active_months'], bins=len(all_months), kde=False)
plt.title('Distribution of Supplier Activity (Number of Active Months)')
plt.xlabel('Number of Active Months')
plt.ylabel('Number of Suppliers')
plt.xticks(range(1, len(all_months)+1))
plt.grid(True, alpha=0.3)
plt.show()


# Function to create features for a given month
def create_features_for_month(target_month, history_months, pivot_df):
    # Convert the pivot table to a DataFrame for easier manipulation
    activity_df = pivot_df.copy().reset_index()
    
    features = []
    
    for supplier in activity_df['RUC_PROVEEDOR']:
        supplier_data = activity_df[activity_df['RUC_PROVEEDOR'] == supplier]
        
        # Basic features
        row = {
            'RUC_PROVEEDOR': supplier,
            'target_month': target_month
        }
        
        # Activity in previous months
        for i, month in enumerate(history_months):
            if month in pivot_df.columns:
                row[f'active_{i+1}_months_ago'] = supplier_data[month].values[0]
            else:
                row[f'active_{i+1}_months_ago'] = 0
        
        # Calculate activity patterns
        history_values = [supplier_data[month].values[0] if month in pivot_df.columns else 0 for month in history_months]
        
        # Total active months in history
        row['total_active_months'] = sum(history_values)
        
        # Active in last month
        row['active_last_month'] = history_values[0] if history_values else 0
        
        # Active in any of last 3 months
        row['active_in_last_3_months'] = 1 if sum(history_values[:3]) > 0 else 0
        
        # Percentage of active months
        row['active_month_ratio'] = sum(history_values) / len(history_values) if history_values else 0
        
        # Trend: increasing or decreasing activity
        if len(history_values) >= 3:
            recent = sum(history_values[:3])
            older = sum(history_values[3:6]) if len(history_values) >= 6 else sum(history_values[3:])
            row['activity_trend'] = recent - older
        else:
            row['activity_trend'] = 0
            
        features.append(row)
    
    return pd.DataFrame(features)

# Let's test this for one month
# We'll use February 2024 (202402) as our target month and use previous months as history
all_months = sorted(train_df['CODMES'].unique())
print("All months in chronological order:")
print(all_months)

# Choose a test month
test_month = '202402'  # February 2024
# Get previous months for history (up to 6 months)
history_months = [m for m in all_months if m < test_month][-6:]
history_months.reverse()  # Most recent first

print(f"\nTarget month: {test_month}")
print(f"History months (most recent first): {history_months}")

# Create features for the test month
test_features = create_features_for_month(test_month, history_months, pivot_binary_df)
print("\nFeatures for prediction (first few rows):")
print(test_features.head())


# Load the sample submission to get the required suppliers for March and April 2024
sample_sub = pd.read_csv('/kaggle/input/desafo-clasificacin-de-proveedores-activos/sample_submission.csv')

# Extract the CODMES and RUC_PROVEEDOR from the CODMES_RUC_PROVEEDOR column
sample_sub[['CODMES', 'RUC_PROVEEDOR']] = sample_sub['CODMES_RUC_PROVEEDOR'].str.split('_', expand=True)
sample_sub['RUC_PROVEEDOR'] = sample_sub['RUC_PROVEEDOR'].astype(int)

# Get unique months in the submission file
submission_months = sample_sub['CODMES'].unique()
print(f"\nMonths to predict: {submission_months}")

# Get unique suppliers in the submission file
submission_suppliers = sample_sub['RUC_PROVEEDOR'].unique()
print(f"Number of suppliers to predict: {len(submission_suppliers)}")

# Check if all submission suppliers are in our training data
suppliers_in_train = set(train_df['RUC_PROVEEDOR'].unique())
suppliers_in_submission = set(submission_suppliers)
missing_suppliers = suppliers_in_submission - suppliers_in_train

print(f"Number of suppliers in submission not found in training data: {len(missing_suppliers)}")
if len(missing_suppliers) > 0:
    print("First few missing suppliers:", list(missing_suppliers)[:5])



# Function to prepare data for all months
def prepare_all_data(pivot_binary_df, history_window=6):
    all_months = sorted(pivot_binary_df.columns)
    all_data = []
    
    # Start from the 7th month to have enough history
    for i in range(history_window, len(all_months)):
        target_month = all_months[i]
        history_months = all_months[i-history_window:i]
        history_months.reverse()  # Most recent first
        
        # Create features
        month_features = create_features_for_month(target_month, history_months, pivot_binary_df)
        
        # Add target (whether supplier was active in the target month)
        month_features['TARGET'] = month_features.apply(
            lambda row: pivot_binary_df.loc[row['RUC_PROVEEDOR'], target_month], 
            axis=1
        )
        
        all_data.append(month_features)
    
    return pd.concat(all_data, ignore_index=True)

# Prepare all data
all_data = prepare_all_data(pivot_binary_df)

print("Prepared data shape:", all_data.shape)
print("First few rows:")
print(all_data.head())

# Check class distribution
print("\nTarget distribution:")
print(all_data['TARGET'].value_counts(normalize=True))

# Split the data into features and target
X = all_data.drop(['RUC_PROVEEDOR', 'target_month', 'TARGET'], axis=1)
y = all_data['TARGET']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_val_scaled)

# Evaluate the model
print("\nRandom Forest Model Evaluation:")
print("F1 Score:", f1_score(y_val, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_val, y_pred_rf))

# Train a Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_gb = gb_model.predict(X_val_scaled)

# Evaluate the model
print("\nGradient Boosting Model Evaluation:")
print("F1 Score:", f1_score(y_val, y_pred_gb))
print("\nClassification Report:")
print(classification_report(y_val, y_pred_gb))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)



# Function to prepare prediction data for a specific month
def prepare_prediction_data(target_month, suppliers, pivot_df):
    # Get the 6 most recent months before the target month
    all_months = sorted(pivot_df.columns)
    history_months = [m for m in all_months if m < target_month][-6:]
    history_months.reverse()  # Most recent first
    
    # Create features
    features = []
    
    for supplier in suppliers:
        # Basic features
        row = {
            'RUC_PROVEEDOR': supplier,
            'target_month': target_month
        }
        
        # Activity in previous months
        for i, month in enumerate(history_months):
            if month in pivot_df.columns and supplier in pivot_df.index:
                row[f'active_{i+1}_months_ago'] = pivot_df.loc[supplier, month]
            else:
                row[f'active_{i+1}_months_ago'] = 0
        
        # Calculate activity patterns
        history_values = []
        for month in history_months:
            if month in pivot_df.columns and supplier in pivot_df.index:
                history_values.append(pivot_df.loc[supplier, month])
            else:
                history_values.append(0)
        
        # Total active months in history
        row['total_active_months'] = sum(history_values)
        
        # Active in last month
        row['active_last_month'] = history_values[0] if history_values else 0
        
        # Active in any of last 3 months
        row['active_in_last_3_months'] = 1 if sum(history_values[:3]) > 0 else 0
        
        # Percentage of active months
        row['active_month_ratio'] = sum(history_values) / len(history_values) if history_values else 0
        
        # Trend: increasing or decreasing activity
        if len(history_values) >= 3:
            recent = sum(history_values[:3])
            older = sum(history_values[3:6]) if len(history_values) >= 6 else sum(history_values[3:])
            row['activity_trend'] = recent - older
        else:
            row['activity_trend'] = 0
            
        features.append(row)
    
    return pd.DataFrame(features)

# Get unique suppliers from the sample submission
sample_sub = pd.read_csv('/kaggle/input/desafo-clasificacin-de-proveedores-activos/sample_submission.csv')
sample_sub[['CODMES', 'RUC_PROVEEDOR']] = sample_sub['CODMES_RUC_PROVEEDOR'].str.split('_', expand=True)
sample_sub['RUC_PROVEEDOR'] = sample_sub['RUC_PROVEEDOR'].astype(int)

# Get unique suppliers and months
submission_suppliers = sample_sub['RUC_PROVEEDOR'].unique()
submission_months = sample_sub['CODMES'].unique()

# Prepare prediction data for March 2024
march_data = prepare_prediction_data('202403', submission_suppliers, pivot_binary_df)
print("March 2024 prediction data shape:", march_data.shape)

# Prepare prediction data for April 2024
april_data = prepare_prediction_data('202404', submission_suppliers, pivot_binary_df)
print("April 2024 prediction data shape:", april_data.shape)

# Scale the features
X_march = march_data.drop(['RUC_PROVEEDOR', 'target_month'], axis=1)
X_april = april_data.drop(['RUC_PROVEEDOR', 'target_month'], axis=1)

X_march_scaled = scaler.transform(X_march)
X_april_scaled = scaler.transform(X_april)

# Make predictions
march_preds = rf_model.predict(X_march_scaled)
april_preds = rf_model.predict(X_april_scaled)

# Create submission dataframe
march_submission = pd.DataFrame({
    'CODMES_RUC_PROVEEDOR': '202403_' + march_data['RUC_PROVEEDOR'].astype(str),
    'TARGET': march_preds
})

april_submission = pd.DataFrame({
    'CODMES_RUC_PROVEEDOR': '202404_' + april_data['RUC_PROVEEDOR'].astype(str),
    'TARGET': april_preds
})

# Combine the predictions
final_submission = pd.concat([march_submission, april_submission], ignore_index=True)

# Check the submission
print("\nFinal submission shape:", final_submission.shape)
print("First few rows:")
print(final_submission.head())

# Check the distribution of predictions
print("\nPrediction distribution:")
print(final_submission['TARGET'].value_counts())

# Save the submission file
final_submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved as 'submission.csv'")



from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Create the grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

# Fit the grid search
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print("\nBest parameters:", grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred_best = best_model.predict(X_val_scaled)

# Evaluate the best model
print("\nBest Model Evaluation:")
print("F1 Score:", f1_score(y_val, y_pred_best))
print("\nClassification Report:")
print(classification_report(y_val, y_pred_best))

# Make predictions for March and April with the best model
march_preds_best = best_model.predict(X_march_scaled)
april_preds_best = best_model.predict(X_april_scaled)

# Create submission dataframe with the best model
march_submission_best = pd.DataFrame({
    'CODMES_RUC_PROVEEDOR': '202403_' + march_data['RUC_PROVEEDOR'].astype(str),
    'TARGET': march_preds_best
})

april_submission_best = pd.DataFrame({
    'CODMES_RUC_PROVEEDOR': '202404_' + april_data['RUC_PROVEEDOR'].astype(str),
    'TARGET': april_preds_best
})

# Combine the predictions
final_submission_best = pd.concat([march_submission_best, april_submission_best], ignore_index=True)

# Save the submission file
final_submission_best.to_csv('submission_best.csv', index=False)
print("\nBest model submission file saved as 'submission_best.csv'")


