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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Load the datasets
train_df = pd.read_csv('/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/train.csv')
test_df = pd.read_csv('/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/test.csv')
sample_submission = pd.read_csv('/kaggle/input/binary-classification-with-a-bank-churn-dataset-1/sample_submission.csv')

# Display the first few rows of the training data
print(train_df.head())


# Check for missing values
print("Missing values in the training set:")
print(train_df.isnull().sum())

# Summary statistics of numerical features
print("\nSummary statistics of numerical features:")
print(train_df.describe())

# Distribution of the target variable 'Exited'
plt.figure(figsize=(6, 4))
sns.countplot(x='Exited', data=train_df)
plt.title('Distribution of Exited (Churn)')
plt.xlabel('Exited (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = train_df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# Handle categorical variables using one-hot encoding
train_df = pd.get_dummies(train_df, columns=['Geography', 'Gender'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Geography', 'Gender'], drop_first=True)

# Separate features and target
X = train_df.drop(['id', 'CustomerId', 'Surname', 'Exited'], axis=1)  # Drop non-relevant columns
y = train_df['Exited']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Scale the test set
test_df_scaled = scaler.transform(test_df.drop(['id', 'CustomerId', 'Surname'], axis=1))


# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model
model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = model.predict_proba(X_val)[:, 1]  # Probabilities for the positive class (Exited = 1)

# Evaluate the model using ROC AUC score
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f'Validation ROC AUC Score: {roc_auc}')


# Predict on the test set
test_preds = model.predict_proba(test_df_scaled)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({'id': test_df['id'], 'Exited': test_preds})
submission.to_csv('submission1.csv', index=False)

# Display the first few rows of the submission file
print(submission.head())


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'scale_pos_weight': [len(y_train) - sum(y_train) / sum(y_train)]  # Adjust for class imbalance
}

# Initialize the XGBoost model
xgb_model = XGBClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best ROC AUC Score:", grid_search.best_score_)


# Create new features for the training set
train_df['BalancePerProduct'] = train_df['Balance'] / (train_df['NumOfProducts'] + 1e-6)
train_df['AgeTimesTenure'] = train_df['Age'] * train_df['Tenure']
train_df['CreditScoreGroup'] = pd.cut(train_df['CreditScore'], bins=[0, 600, 700, 850], labels=[0, 1, 2])

# Create new features for the test set
test_df['BalancePerProduct'] = test_df['Balance'] / (test_df['NumOfProducts'] + 1e-6)
test_df['AgeTimesTenure'] = test_df['Age'] * test_df['Tenure']
test_df['CreditScoreGroup'] = pd.cut(test_df['CreditScore'], bins=[0, 600, 700, 850], labels=[0, 1, 2])

# Update the feature set
X = train_df.drop(['id', 'CustomerId', 'Surname', 'Exited'], axis=1)
X_test = test_df.drop(['id', 'CustomerId', 'Surname'], axis=1)

# One-hot encode new categorical features
X = pd.get_dummies(X, columns=['CreditScoreGroup'], drop_first=True)
X_test = pd.get_dummies(X_test, columns=['CreditScoreGroup'], drop_first=True)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Initialize the XGBoost model with the best parameters
best_params = grid_search.best_params_
xgb_model_tuned = XGBClassifier(**best_params, random_state=42)

# Train the model
xgb_model_tuned.fit(X_train, y_train)

# Predict on the validation set
y_val_pred_tuned = xgb_model_tuned.predict_proba(X_val)[:, 1]

# Evaluate the model using ROC AUC score
roc_auc_tuned = roc_auc_score(y_val, y_val_pred_tuned)
print(f'Tuned XGBoost Validation ROC AUC Score: {roc_auc_tuned}')


from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Initialize the RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Create an ensemble of XGBoost and RandomForest
ensemble = VotingClassifier(estimators=[
    ('xgb', xgb_model_tuned),
    ('rf', rf_model)
], voting='soft')

# Train the ensemble model
ensemble.fit(X_train, y_train)

# Predict on the validation set
y_val_pred_ensemble = ensemble.predict_proba(X_val)[:, 1]

# Evaluate the ensemble model using ROC AUC score
roc_auc_ensemble = roc_auc_score(y_val, y_val_pred_ensemble)
print(f'Ensemble Validation ROC AUC Score: {roc_auc_ensemble}')


# Predict on the test set using the ensemble model
test_preds_ensemble = ensemble.predict_proba(X_test_scaled)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({'id': test_df['id'], 'Exited': test_preds_ensemble})

# Save the submission file
submission.to_csv('submission_ensemble.csv', index=False)

# Display the first few rows of the submission file
print(submission.head())


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# Define base models
base_models = [
    ('xgb', xgb_model_tuned),
    ('rf', rf_model)
]

# Define meta-model
meta_model = LogisticRegression()

# Create stacking ensemble
stacking_ensemble = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Train the stacking ensemble
stacking_ensemble.fit(X_train, y_train)

# Predict on the validation set
y_val_pred_stacking = stacking_ensemble.predict_proba(X_val)[:, 1]

# Evaluate the stacking ensemble
roc_auc_stacking = roc_auc_score(y_val, y_val_pred_stacking)
print(f'Stacking Ensemble Validation ROC AUC Score: {roc_auc_stacking}')


# Predict on the test set using the stacking ensemble model
test_preds_stacking_ensemble = stacking_ensemble.predict_proba(X_test_scaled)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({'id': test_df['id'], 'Exited': test_preds_stacking_ensemble})

# Save the submission file
submission.to_csv('submission_stacking_ensemble.csv', index=False)

# Display the first few rows of the submission file
print(submission.head())


from lightgbm import LGBMClassifier

# Initialize LightGBM model
lgbm_model = LGBMClassifier(random_state=42, class_weight='balanced')

# Train the model
lgbm_model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred_lgbm = lgbm_model.predict_proba(X_val)[:, 1]

# Evaluate the model
roc_auc_lgbm = roc_auc_score(y_val, y_val_pred_lgbm)
print(f'LightGBM Validation ROC AUC Score: {roc_auc_lgbm}')


# Predict on the test set using the LGBM model
test_preds_lgbm_model = lgbm_model.predict_proba(X_test_scaled)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({'id': test_df['id'], 'Exited': test_preds_lgbm_model})

# Save the submission file
submission.to_csv('submission_lgbm_model.csv', index=False)

# Display the first few rows of the submission file
print(submission.head())


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Define base models with better parameters
base_models = [
    ('xgb', xgb_model_tuned),
    ('rf', rf_model),
    ('lgbm', LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=32,
        random_state=42,
        class_weight='balanced'
    ))
]

# Define meta-model with balanced class weights
meta_model = LogisticRegression(class_weight='balanced', max_iter=1000)

# Create stacking ensemble with proper CV
stacking_ensemble = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1,
    passthrough=True  # Include original features
)



def create_advanced_features(df):
    # Convert to DataFrame if input is numpy array
    df = pd.DataFrame(df) if isinstance(df, np.ndarray) else df.copy()
    
    # Convert all column names to strings
    df.columns = df.columns.astype(str)
    
    # Create numerical features
    df['CLV'] = df.iloc[:, 5] * df.iloc[:, 3]
    df['Age_Balance_Ratio'] = df.iloc[:, 1] / (df.iloc[:, 5] + 1)
    df['Balance_per_Product'] = df.iloc[:, 5] / (df.iloc[:, 2] + 1)
    
    # Create CreditScore bins as numerical values instead of categories
    df['CreditScore_Bin'] = pd.qcut(df.iloc[:, 0], q=5, labels=[0, 1, 2, 3, 4])
    
    return df

# Update XGBoost parameters in the stacking ensemble
xgb_model_tuned.set_params(enable_categorical=True)

# Apply feature engineering and fit the model
X_train_enhanced = create_advanced_features(X_train)
X_val_enhanced = create_advanced_features(X_val)

# Fit and evaluate
stacking_ensemble.fit(X_train_enhanced, y_train)
y_val_pred_stacking = stacking_ensemble.predict_proba(X_val_enhanced)[:, 1]
roc_auc_stacking = roc_auc_score(y_val, y_val_pred_stacking)
print(f'Stacking Ensemble Validation ROC AUC Score: {roc_auc_stacking:.4f}')



from sklearn.model_selection import GridSearchCV
import warnings  
warnings.filterwarnings("ignore")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [31, 50, 100],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(LGBMClassifier(random_state=42), param_grid, scoring='roc_auc', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best ROC AUC Score:", grid_search.best_score_)


best_params = grid_search.best_params_
lgbm_model_tuned = LGBMClassifier(**best_params, random_state=42)

# Train the model
lgbm_model_tuned.fit(X_train, y_train)

# Predict on the validation set
y_val_pred_lgbm_tuned = lgbm_model_tuned.predict_proba(X_val)[:, 1]

# Evaluate the model
roc_auc_lgbm_tuned = roc_auc_score(y_val, y_val_pred_lgbm_tuned)
print(f'Tuned LightGBM Validation ROC AUC Score: {roc_auc_lgbm_tuned}')


# Predict on the test set
test_preds_lgbm_tuned = lgbm_model_tuned.predict_proba(X_test_scaled)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({'id': test_df['id'], 'Exited': test_preds_lgbm_tuned})

# Save the submission file
submission.to_csv('submission.csv', index=False)

# Display the first few rows of the submission file
print(submission.head())

