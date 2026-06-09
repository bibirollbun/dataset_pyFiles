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


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import json
import optuna


from sklearn.model_selection import KFold


# Set some display options for better visualization
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')


df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


# Display the first few rows of the training data
print("Training Data Head:")
df_train.head()


df_train.columns


df_train.shape


# Get a concise summary of the dataframe
print("\nTraining Data Info:")
df_train.info()


# Check for missing values
print("\nMissing Values in Train Data:")
print(df_train.isnull().sum())


# Check for missing values
print("\nMissing Values in Test Data:")
print(df_test.isnull().sum())


# Descriptive statistics for numerical columns
df_train.describe()


# Distribution of the target variable 'accident_risk'
plt.figure(figsize=(10, 6))
sns.countplot(x='loan_paid_back', data=df_train, palette='pastel', edgecolor='black')
plt.title('Distribution of Loan Payback')
plt.xlabel('Loan Payback')
plt.ylabel('Count')
plt.show()


categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_features)



# A more compact view of categorical features vs the target
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
axes = axes.flatten()
cmap = plt.get_cmap('magma')
colors = cmap([0.9, 0.66, 0.33])
target = 'loan_paid_back'

for i, col in enumerate(categorical_features):
    grouped = df_train.groupby(col)[target].mean()
    axes[i].bar(grouped.index.astype(str), grouped.values, color=colors)
    axes[i].set_ylabel(f'Mean {target}')
    axes[i].set_title(f'{col} vs {target}')
    axes[i].tick_params(axis='x', rotation=45)
    
plt.tight_layout()
plt.show()


# Correlation heatmap for numerical features
numerical_features = df_train.select_dtypes(include=np.number)
# id is not important ashas no information for correlation matrix
numerical_features.drop(columns=['id'], errors='ignore', inplace=True)
plt.figure(figsize=(12, 10))
sns.heatmap(numerical_features.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# Combine train and test for consistent encoding
combined_df = pd.concat([df_train.drop('loan_paid_back', axis=1), df_test], ignore_index=True)

# Drop the id column as it's not a feature
train_ids = df_train['id']
test_ids = df_test['id']
combined_df = combined_df.drop('id', axis=1)


# Example engineered features
#combined_df['income_to_loan_ratio'] = combined_df['annual_income'] / combined_df['loan_amount']
#combined_df['interest_per_income'] = combined_df['interest_rate'] / combined_df['annual_income']
#combined_df['loan_to_credit'] = combined_df['loan_amount'] / combined_df['credit_score']



"""Create interaction and derived numerical features"""
#df_new = df.copy()
        
# Income-based ratios
combined_df['income_to_loan_ratio'] = combined_df['annual_income'] / (combined_df['loan_amount'] + 1)
combined_df['loan_to_income_pct'] = (combined_df['loan_amount'] / combined_df['annual_income']) * 100
        
# Credit utilization proxy
combined_df['debt_amount'] = combined_df['annual_income'] * combined_df['debt_to_income_ratio']
combined_df['total_debt_with_loan'] = combined_df['debt_amount'] + combined_df['loan_amount']
combined_df['total_debt_to_income'] = combined_df['total_debt_with_loan'] / combined_df['annual_income']
        
# Interest burden
combined_df['annual_interest_payment'] = (combined_df['loan_amount'] * combined_df['interest_rate']) / 100
combined_df['interest_to_income_ratio'] = combined_df['annual_interest_payment'] / combined_df['annual_income']
        
# Credit score interactions
combined_df['credit_score_x_income'] = combined_df['credit_score'] * combined_df['annual_income'] / 100000
combined_df['credit_score_x_dti'] = combined_df['credit_score'] * (1 - combined_df['debt_to_income_ratio'])
        
# Risk indicators
combined_df['high_dti_low_credit'] = ((combined_df['debt_to_income_ratio'] > 0.4) & 
                                    (combined_df['credit_score'] < 650)).astype(int)
combined_df['low_income_high_loan'] = ((combined_df['annual_income'] < 50000) & 
                                    (combined_df['loan_amount'] > 20000)).astype(int)
        
# Binned features (useful for tree models)
combined_df['income_bracket'] = pd.cut(combined_df['annual_income'], 
                                    bins=[0, 30000, 50000, 75000, 100000, np.inf],
                                    labels=[1, 2, 3, 4, 5]).astype(float)
        
combined_df['credit_tier'] = pd.cut(combined_df['credit_score'],
                                bins=[0, 580, 670, 740, 800, np.inf],
                                labels=[1, 2, 3, 4, 5]).astype(float)
        
combined_df['dti_bracket'] = pd.cut(combined_df['debt_to_income_ratio'],
                                bins=[0, 0.2, 0.35, 0.5, np.inf],
                                labels=[1, 2, 3, 4]).astype(float)
        
# Polynomial features for key variables
combined_df['credit_score_squared'] = combined_df['credit_score'] ** 2
combined_df['income_log'] = np.log1p(combined_df['annual_income'])
combined_df['loan_amount_log'] = np.log1p(combined_df['loan_amount'])


# Correlation heatmap for numerical features
numerical_features = combined_df.select_dtypes(include=np.number)
# id is not important ashas no information for correlation matrix
numerical_features.drop(columns=['id'], errors='ignore', inplace=True)
plt.figure(figsize=(12, 10))
sns.heatmap(numerical_features.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical + Engineered Features')
plt.show()


import category_encoders as ce

# Recreate the combined_df (without target)
#combined_df = pd.concat([df_train.drop('loan_paid_back', axis=1), df_test], ignore_index=True)

# Store target separately
y = df_train['loan_paid_back']

# Initialize target encoder
target_encoder = ce.TargetEncoder(cols=categorical_features)

# !!!IMPORTANT !!
# Fit only on training data, transform both train and test
target_encoder.fit(df_train[categorical_features], y)

# Apply same learned mapping to both train and test
combined_df[categorical_features] = target_encoder.transform(combined_df[categorical_features])


# 5. === Remove highly correlated features ===
corr_matrix = combined_df.select_dtypes(include=[np.number]).corr().abs()

# Replace NaNs with 0 (no correlation)
corr_matrix = corr_matrix.fillna(0)

# Take upper triangle
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Define threshold (you can adjust between 0.85–0.9)
threshold = 0.85

# Find features to drop
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print(f"Number of features to drop due to correlation > {threshold}: {len(to_drop)}")
print("Dropped features:", to_drop)

# Drop them from the dataset
combined_df_filtered = combined_df.drop(columns=to_drop)

# 6. Separate back into training and testing sets
X = combined_df_filtered.iloc[:len(df_train)]
X_test = combined_df_filtered.iloc[len(df_train):]
#y = df_train['loan_paid_back']

print("Shape of processed training data (X):", X.shape)
print("Shape of processed test data (X_test):", X_test.shape)



# Correlation heatmap for numerical features
numerical_features = combined_df_filtered.select_dtypes(include=np.number)
# id is not important ashas no information for correlation matrix
numerical_features.drop(columns=['id'], errors='ignore', inplace=True)
plt.figure(figsize=(12, 10))
sns.heatmap(numerical_features.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of AFTER Threshold and Target Encoding')
plt.show()


# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# XGBoost parameters for classification
xgb_params = {
    'objective': 'binary:logistic',     # for probability output between 0 and 1
    'eval_metric': 'auc',               # AUC metric directly in XGBoost
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'early_stopping_rounds': 50,
    'n_jobs': -1,
    'tree_method': 'hist'               # use 'gpu_hist' if GPU available
}

# Train the model
xgb_model = xgb.XGBClassifier(**xgb_params)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Predict probabilities (for ROC AUC, we need probabilities, not labels)
val_preds_proba = xgb_model.predict_proba(X_val)[:, 1]

# Compute ROC-AUC
auc_score = roc_auc_score(y_val, val_preds_proba)
print(f"XGBoost Validation AUC: {auc_score:.5f}")


# Plot feature importance using the correct model variable: xgb_model
xgb.plot_importance(xgb_model, max_num_features=20)

# Customize the plot
plt.title('XGBoost Feature Importance')
plt.gcf().set_size_inches(12, 10) # A good way to set figure size
plt.tight_layout()
plt.show()


 final_xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators' : 2000,
        'early_stopping_rounds': 100,
        'tree_method': 'hist',
        'n_jobs': -1,
        'random_state': 42,
        'learning_rate': 0.10168576005909855,
        'max_depth': 4,
        'subsample': 0.9933104859166163,
        'colsample_bytree': 0.6757368270740294,
        'gamma': 8.900539690103504e-06,
        'lambda': 0.0035807311955684974,
        'alpha': 0.02091505227850534   
}


# import optuna
# import xgboost as xgb
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# import numpy as np




# # 1. Define the objective function for Optuna
# def objective(trial):
#     """
#     This function takes a trial, creates a set of parameters,
#     trains an XGBoost model, and returns the average validation RMSE
#     from a 3-fold cross-validation.
#     """
    
#     # Define the search space for the hyperparameters
#     params = {
#         'objective': 'binary:logistic',
#         'eval_metric': 'auc',
#         'n_estimators': 1000, # We use early stopping, so this can be a high number
#         'early_stopping_rounds' : 50, # Use early stopping
#         'tree_method': 'hist',
#         'n_jobs': -1,
#         'random_state': 42,
        
#         # Parameters to be tuned by Optuna
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'max_depth': trial.suggest_int('max_depth', 4, 10),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
#         'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True), # L2 regularization
#         'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),   # L1 regularization
#     }

#     # Use cross-validation within the trial for a robust score
#     N_SPLITS_TRIAL = 3
#     kf = KFold(n_splits=N_SPLITS_TRIAL, shuffle=True, random_state=42)
    
#     scores = []
  
#     for train_index, val_index in kf.split(X, y):
#         X_train, X_val = X.iloc[train_index], X.iloc[val_index]
#         y_train, y_val = y.iloc[train_index], y.iloc[val_index]

#         model = xgb.XGBClassifier(**params)
#         model.fit(X_train, y_train,
#                   eval_set=[(X_val, y_val)],
#                   verbose=False)
        
#         # Predict probabilities (for ROC AUC, we need probabilities, not labels)
#         val_preds_proba = model.predict_proba(X_val)[:, 1]

#         # Compute ROC-AUC
#         auc_score = roc_auc_score(y_val, val_preds_proba)
        
        
#         #preds = model.predict(X_val)
#         #rmse = np.sqrt(mean_squared_error(y_val, preds))
#         scores.append(auc_score)

#     # Return the average RMSE across the folds
#     return np.mean(scores)


# # 2. Create a study and run the optimization
# # The 'direction' is 'maximize' because we want the higgest AUC
# study = optuna.create_study(direction='maximize')

# # n_trials is the number of different parameter combinations Optuna will try.
# study.optimize(objective, n_trials=30)


# # 3. Print the results
# print("Number of finished trials: ", len(study.trials))
# print("Best trial:")
# trial = study.best_trial

# print("  Value (AUC): ", trial.value)
# print("  Params: ")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")

# # You can now get the dictionary of the best parameters
# best_params = trial.params


# import json
# ## NEW: Save the Best Parameters to a JSON file ---
# best_params = trial.params
# file_path = 'best_xgb_params.json'
# with open(file_path, 'w') as f:
#     json.dump(best_params, f, indent=4)

# print(f"\nBest parameters saved to {file_path}")


# # Define the path to your saved parameters file
# params_file = 'best_xgb_params.json'
# final_xgb_params = {}

# # Load the parameters from the file
# try:
#     with open(params_file, 'r') as f:
#         best_params_from_study = json.load(f)
#         print("Successfully loaded parameters from file.")
        
#     # Combine with your other fixed parameters for the final model
#     final_xgb_params = {
#         'objective': 'binary:logistic',
#         'eval_metric': 'auc',
#         'n_estimators' : 2000,
#         'early_stopping_rounds': 100,
#         'tree_method': 'hist',
#         'n_jobs': -1,
#         'random_state': 42,
#         **best_params_from_study # Unpacks the loaded dictionary
#     }

# except FileNotFoundError:
#     print(f"Error: Parameter file not found at {params_file}. Please run the tuning script first.")
#     # You might want to fall back to default parameters here if the file doesn't exist


# Setup K-Fold Cross-Validation
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_predictions = np.zeros(X.shape[0])
test_predictions = np.zeros(X_test.shape[0])
models = []
oof_auc_scores = []

# Loop through each fold
for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"===== FOLD {fold+1} =====")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]


    xgb_model = xgb.XGBClassifier(**final_xgb_params)
    
    xgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose = False   )

    # Store predictions

    # Predict probabilities (for ROC AUC, we need probabilities, not labels)
    val_preds_proba = xgb_model.predict_proba(X_val)[:, 1]
    oof_predictions[val_index] = val_preds_proba
    fold_test_preds = xgb_model.predict_proba(X_test)[:, 1]
    test_predictions += fold_test_preds / N_SPLITS
    
    # Compute ROC-AUC
    auc_score = roc_auc_score(y_val, val_preds_proba)
    oof_auc_scores.append(auc_score)
    

    print(f"Fold {fold+1} AUC: {auc_score}")
    
    models.append(xgb_model)

print(f"\nAverage CV AUC: {np.mean(oof_auc_scores):.5f} (+/- {np.std(oof_auc_scores):.5f})")




# Ensure predictions are within the [0, 1] range
test_predictions = np.clip(test_predictions, 0, 1)

# Create the submission file
submission_df = pd.DataFrame({'id': test_ids, 'loan_paid_back': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
submission_df.head()

