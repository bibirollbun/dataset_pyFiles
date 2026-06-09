import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer

import optuna
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from catboost import CatBoostClassifier


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load the datasets
train = pd.read_csv('/kaggle/input/gvu-spring-2025-data-454-project-2/train.csv')
test = pd.read_csv('/kaggle/input/gvu-spring-2025-data-454-project-2/test.csv')

train.head()


train.isna().sum()


test.isna().sum()


train.describe()


# Convert dates to datetime format
train['orderDate'] = pd.to_datetime(train['orderDate'], errors='coerce')
train['deliveryDate'] = pd.to_datetime(train['deliveryDate'], errors='coerce')
train['creationDate'] = pd.to_datetime(train['creationDate'], errors='coerce')
train['dateOfBirth'] = pd.to_datetime(train['dateOfBirth'], errors='coerce')

# Calculate derived features
train['deliveryTime'] = (train['deliveryDate'] - train['orderDate']).dt.days
train['accountAge'] = (train['orderDate'] - train['creationDate']).dt.days
train['customerAge'] = (train['orderDate'] - train['dateOfBirth']).dt.days // 365

# Fill missing values
train['deliveryTime'] = train['deliveryTime'].fillna(train['deliveryTime'].median())
train['accountAge'] = train['accountAge'].fillna(train['accountAge'].median())
train['customerAge'] = train['customerAge'].fillna(train['customerAge'].median())
train['size'] = train['size'].fillna('Unknown')
train['color'] = train['color'].fillna('Unknown')


# Encode categorical variables
categorical_cols = ['size', 'color', 'salutation', 'state']
encoder = LabelEncoder()
for col in categorical_cols:
    train[col] = encoder.fit_transform(train[col].astype(str))


# calculate the age of customers 
train['age'] = (pd.to_datetime('today') - pd.to_datetime(train['dateOfBirth'])).dt.days // 365
train['age'].hist()


train['price'].hist()
train.boxplot(column='price', by='returnShipment')


# Prepare X and y
X = train.drop(['returnShipment'], axis=1)
y = train['returnShipment']

# Convert datetime columns into numeric features or drop them
X['orderDate'] = pd.to_datetime(X['orderDate'], errors='coerce')
X['deliveryDate'] = pd.to_datetime(X['deliveryDate'], errors='coerce')
X['creationDate'] = pd.to_datetime(X['creationDate'], errors='coerce')
X['dateOfBirth'] = pd.to_datetime(X['dateOfBirth'], errors='coerce')

# Feature engineering for datetime columns
X['deliveryTime'] = (X['deliveryDate'] - X['orderDate']).dt.days
X['accountAge'] = (X['orderDate'] - X['creationDate']).dt.days
X['customerAge'] = (X['orderDate'] - X['dateOfBirth']).dt.days // 365

# Drop original datetime columns
X = X.drop(['orderDate', 'deliveryDate', 'creationDate', 'dateOfBirth'], axis=1)

# Fill missing values
X = X.fillna(0)

# Apply pd.get_dummies to categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Prepare test data
test['orderDate'] = pd.to_datetime(test['orderDate'], errors='coerce')
test['deliveryDate'] = pd.to_datetime(test['deliveryDate'], errors='coerce')
test['creationDate'] = pd.to_datetime(test['creationDate'], errors='coerce')
test['dateOfBirth'] = pd.to_datetime(test['dateOfBirth'], errors='coerce')

# Feature engineering for test data
test['deliveryTime'] = (test['deliveryDate'] - test['orderDate']).dt.days
test['accountAge'] = (test['orderDate'] - test['creationDate']).dt.days
test['customerAge'] = (test['orderDate'] - test['dateOfBirth']).dt.days // 365

# Drop datetime columns from test data
test = test.drop(['orderDate', 'deliveryDate', 'creationDate', 'dateOfBirth'], axis=1)

# Fill missing values in test data
test = test.fillna(0)

# Select columns for one-hot encoding (low cardinality)
low_cardinality_cols = ['size', 'color', 'salutation', 'state']

# Apply pd.get_dummies to low-cardinality columns only
test_low_cardinality = pd.get_dummies(test[low_cardinality_cols], drop_first=True)

# Use LabelEncoder for high-cardinality columns
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in ['itemID', 'customerID', 'manufacturerID']:
    test[col] = label_encoder.fit_transform(test[col].astype(str))

# Combine low-cardinality and high-cardinality features
test_processed = pd.concat([test.drop(low_cardinality_cols, axis=1), test_low_cardinality], axis=1)

# Align test columns with train data
test_processed = test_processed.reindex(columns=X_train.columns, fill_value=0)


# Inspect available columns
print("Train Columns:", train.columns)
print("Test Columns:", test.columns)


# Check for datetime columns
datetime_cols = ['orderDate', 'deliveryDate', 'creationDate', 'dateOfBirth']
missing_datetime_cols = [col for col in datetime_cols if col not in train.columns]

if not missing_datetime_cols:
    # If datetime columns exist, process them
    train['orderDate'] = pd.to_datetime(train['orderDate'], errors='coerce')
    train['deliveryDate'] = pd.to_datetime(train['deliveryDate'], errors='coerce')
    train['creationDate'] = pd.to_datetime(train['creationDate'], errors='coerce')
    train['dateOfBirth'] = pd.to_datetime(train['dateOfBirth'], errors='coerce')

    # Feature engineering
    train['deliveryTime'] = (train['deliveryDate'] - train['orderDate']).dt.days
    train['accountAge'] = (train['orderDate'] - train['creationDate']).dt.days
    train['customerAge'] = (train['orderDate'] - train['dateOfBirth']).dt.days // 365

    # Fill missing values for these features
    train.fillna({'deliveryTime': train['deliveryTime'].median(),
                  'accountAge': train['accountAge'].median(),
                  'customerAge': train['customerAge'].median()}, inplace=True)

# Fill missing values for categorical columns
train['size'] = train['size'].fillna('Unknown')
train['color'] = train['color'].fillna('Unknown')

# Encode categorical variables
categorical_cols = ['size', 'color', 'salutation', 'state', 'itemID', 'customerID', 'manufacturerID']
encoder = LabelEncoder()
for col in categorical_cols:
    if col in train.columns:
        train[col] = encoder.fit_transform(train[col].astype(str))


# Prepare features and target variable
X = train.drop(['returnShipment'] + datetime_cols, axis=1, errors='ignore')  # Drop datetime columns if they exist
y = train['returnShipment']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess test dataset
for col in datetime_cols:
    if col in test.columns:
        test[col] = pd.to_datetime(test[col], errors='coerce')

# Feature engineering for test data
if 'deliveryDate' in test.columns and 'orderDate' in test.columns:
    test['deliveryTime'] = (test['deliveryDate'] - test['orderDate']).dt.days
if 'orderDate' in test.columns and 'creationDate' in test.columns:
    test['accountAge'] = (test['orderDate'] - test['creationDate']).dt.days
if 'orderDate' in test.columns and 'dateOfBirth' in test.columns:
    test['customerAge'] = (test['orderDate'] - test['dateOfBirth']).dt.days // 365

# Fill missing values in test data
test.fillna(0, inplace=True)

# Encode categorical variables in the test dataset
for col in categorical_cols:
    if col in test.columns:
        test[col] = encoder.fit_transform(test[col].astype(str))

# Align test columns with train data
test_processed = test.reindex(columns=X_train.columns, fill_value=0)


# # Handle Missing Values
# imputer = SimpleImputer(strategy="mean")  # Replace NaNs with mean values
# X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
# test_processed = pd.DataFrame(imputer.transform(test_processed), columns=test_processed.columns)

# # Define the objective function for Optuna
# def objective(trial):
#     # Hyperparameters to tune
#     n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)  # Number of trees
#     max_depth = trial.suggest_int("max_depth", 3, 20)  # Tree depth
#     min_samples_split = trial.suggest_int("min_samples_split", 2, 20)  # Min samples to split a node

#     # Initialize RepeatedStratifiedKFold
#     skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

#     brier_scores = []

#     for train_idx, test_idx in skf.split(X_train, y_train):
#         # Split the data into train and test for this fold
#         X_fold_train, X_fold_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
#         y_fold_train, y_fold_test = y_train.iloc[train_idx], y_train.iloc[test_idx]

#         # Initialize and train the Random Forest model
#         model = RandomForestClassifier(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             random_state=42,
#             n_jobs=-1  # Use all available CPU cores
#         )
#         model.fit(X_fold_train, y_fold_train)

#         # Predict probabilities for test fold
#         y_pred = model.predict_proba(X_fold_test)[:, 1]

#         # Calculate Brier score
#         brier_score = brier_score_loss(y_fold_test, y_pred)
#         brier_scores.append(brier_score)

#     # Return the mean Brier score across all folds
#     return np.mean(brier_scores)

# # Run Optuna optimization
# study = optuna.create_study(direction="minimize")  # Minimize Brier score
# study.optimize(objective, n_trials=15)  # Run 20 trials for tuning

# # Print the best hyperparameters
# print("Best hyperparameters:", study.best_params)

# # Train the final model using the best parameters
# best_params = study.best_params
# final_model = RandomForestClassifier(
#     n_estimators=best_params["n_estimators"],
#     max_depth=best_params["max_depth"],
#     min_samples_split=best_params["min_samples_split"],
#     random_state=42,
#     n_jobs=-1  # Use all available CPU cores
# )

# final_model.fit(X_train, y_train)

# # Predict on the test dataset
# final_test_predictions = final_model.predict_proba(test_processed)[:, 1]


from sklearn.model_selection import StratifiedKFold

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

brier_scores = []

for train_idx, val_idx in skf.split(X_train, y_train):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Train CatBoost Model
    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="BrierScore",
        random_seed=42,
        verbose=0
    )
    model.fit(X_fold_train, y_fold_train)

    # Predict probabilities
    y_fold_pred_proba = model.predict_proba(X_fold_val)[:, 1]

    # Compute Brier Score
    brier_score = brier_score_loss(y_fold_val, y_fold_pred_proba)
    brier_scores.append(brier_score)

# Average Brier Score across all folds
final_brier_score = np.mean(brier_scores)
print(f"Cross-Validated Brier Score: {final_brier_score}")


# Prepare submission file in the required format
submission = pd.DataFrame({
    'id': test['id'],  # Ensure test IDs are included
    'returnShipment': test_predictions  # Predictions as probabilities
})

# Save the submission file in the current working directory
submission.to_csv('submission.csv', index=False)

print("submission.csv has been created successfully in the current directory.")


# # Define parameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2', None],
#     'bootstrap': [True, False]
# }

# # Initialize GridSearchCV
# grid_search = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=42),
#     param_grid=param_grid,
#     cv=5,
#     scoring='neg_brier_score',  # Using Brier Score for evaluation
#     n_jobs=-1
# )

# # Fit GridSearchCV to the training data
# grid_search.fit(X_train, y_train)

# # Extract the best model
# best_model = grid_search.best_estimator_
# print(f"Best Parameters: {grid_search.best_params_}")

# # Predict probabilities on validation set
# y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]

# # Evaluate Brier Score
# brier_score = brier_score_loss(y_val, y_val_pred_proba)
# print(f"Brier Score with tuned model: {brier_score}")


# # Predict probabilities on test data using the best model
# test_prediction_rf = best_model.predict_proba(test_processed)[:, 1]

# # Prepare submission file
# submission_rf = pd.DataFrame({
#     'id': test['id'],  # Ensure test IDs are included
#     'returnShipment': test_prediction_rf  # Predictions as probabilities
# })

# # Save the submission file in the current working directory
# submission_rf.to_csv('submission_rf.csv', index=False)

# print("submission_rf.csv has been created successfully in the current directory.")


# # Base Model
# model_et = ExtraTreesClassifier(random_state=42, n_estimators=100)
# model_et.fit(X_train, y_train)

# # Predict probabilities
# y_val_pred_proba_et = model_et.predict_proba(X_val)[:, 1]

# # Evaluate Brier Score
# brier_score_et = brier_score_loss(y_val, y_val_pred_proba_et)
# print(f"Brier Score: {brier_score_et}")


# # Predict probabilities on test data
# test_predictions_et = model_et.predict_proba(test_processed)[:, 1]

# # Prepare submission file in the required format
# submission_et = pd.DataFrame({
#     'id': test['id'],  # Ensure test IDs are included
#     'returnShipment': test_predictions_et  # Predictions as probabilities
# })

# # Save the submission file in the current working directory
# submission_et.to_csv('submission_et.csv', index=False)

# print("submission_et.csv has been created successfully in the current directory.")


# # Base Model
# model_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# model_xgb.fit(X_train, y_train)

# # Predict probabilities
# y_val_pred_proba_xgb = model_xgb.predict_proba(X_val)[:, 1]

# # Evaluate Brier Score
# brier_score_xgb = brier_score_loss(y_val, y_val_pred_proba_xgb)
# print(f"Brier Score: {brier_score_xgb}")


# # Predict probabilities on test data
# test_predictions_xgb = model_xgb.predict_proba(test_processed)[:, 1]

# # Prepare submission file in the required format
# submission_xgb = pd.DataFrame({
#     'id': test['id'],  # Ensure test IDs are included
#     'returnShipment': test_predictions_xgb  # Predictions as probabilities
# })

# # Save the submission file in the current working directory
# submission_xgb.to_csv('submission_xgb.csv', index=False)

# print("submission_xgb.csv has been created successfully in the current directory.")




