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


# Step 1.1: Import required libraries
import pandas as pd  # pandas is used for reading CSV files and handling tabular data

# Step 1.2: Define file paths (you can adjust paths as needed)
train_path = "/kaggle/input/playground-series-s5e12/train.csv"  # path to the training dataset
test_path = "/kaggle/input/playground-series-s5e12/test.csv"    # path to the test dataset

# Step 1.3: Load the datasets into pandas DataFrames
train_df = pd.read_csv(train_path)  # loads the training data containing features and the target column
test_df = pd.read_csv(test_path)    # loads the test data containing features but no target column

# Step 1.4: Display shapes of the datasets
print("Train dataset shape:", train_df.shape)  # shows number of rows and columns, helps verify file integrity
print("Test dataset shape:", test_df.shape)    # helps confirm consistency with the expected test size

# Step 1.5: Preview the first few rows of the train dataset
print("\nTrain dataset preview:")
print(train_df.head())  # helps us understand what the data looks like and what features exist

# Step 1.6: Preview the first few rows of the test dataset
print("\nTest dataset preview:")
print(test_df.head())  # ensures test dataset has same feature structure as train (except target column)

# Step 1.7: Check for missing values in training dataset
print("\nMissing values in the training dataset:")
print(train_df.isnull().sum())  # counts missing values in each column, important for preprocessing

# Step 1.8: Check for missing values in the test dataset
print("\nMissing values in the test dataset:")
print(test_df.isnull().sum())  # helps identify if test data requires special handling

# Step 1.9: Display data types for each column in the training dataset
print("\nTraining dataset column information:")
print(train_df.info())  # provides column types and non-null counts, useful for detecting numerical/categorical variables

# Step 1.10: Display data types for each column in the test dataset
print("\nTest dataset column information:")
print(test_df.info())  # helps ensure both datasets share the same structure (except target column)



# Step 2.1: Import visualization libraries
import matplotlib.pyplot as plt  # used for general plotting
import seaborn as sns            # used for more complex visualizations

# Configure default plot style
sns.set(style="whitegrid")  # sets a clean background for better readability




plt.figure(figsize=(6,4))  # sets plot size
sns.countplot(x=train_df['diagnosed_diabetes'])  # count of 0 vs 1
plt.title("Distribution of Target Variable: diagnosed_diabetes")  # plot title
plt.xlabel("Diagnosed Diabetes (0 = No, 1 = Yes)")  # x-axis label
plt.ylabel("Count")  # y-axis label
plt.show()  # display the plot

# Calculate proportion of positive and negative classes
target_counts = train_df['diagnosed_diabetes'].value_counts(normalize=True)
print("Target class proportions:")
print(target_counts)




numerical_features = [
    'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
    'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides'
]

# Plot histograms for numerical features
train_df[numerical_features].hist(figsize=(18, 12), bins=30)
plt.suptitle("Distribution of Numerical Features")
plt.show()




categorical_features = [
    'gender', 'ethnicity', 'education_level',
    'income_level', 'smoking_status', 'employment_status'
]

# Plot countplots for each categorical feature
plt.figure(figsize=(10, 12))  # large figure size for multiple subplots

for i, col in enumerate(categorical_features, 1):
    plt.subplot(2, 3, i)  # create a 2x3 grid of subplots
    sns.countplot(x=train_df[col])
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)  # rotate labels for readability

plt.tight_layout()
plt.show()




plt.figure(figsize=(12, 10))  # set figure size
corr_matrix = train_df.corr(numeric_only=True)  # compute correlation only for numeric columns
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)  # heatmap of correlations
plt.title("Correlation Heatmap of Numerical Features")
plt.show()




target_correlations = corr_matrix['diagnosed_diabetes'].sort_values(ascending=False)
print("Correlation of Numerical Features with Target (diagnosed_diabetes):")
print(target_correlations)




X = train_df.drop(columns=['diagnosed_diabetes'])  # all input features
y = train_df['diagnosed_diabetes']                 # target variable

# The test dataset contains only features
X_test = test_df.copy()                            # store test features separately





# Numerical features (already inspected during EDA)
numerical_features = [
    'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
    'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides'
]

# Binary integer features (0 or 1)
binary_features = [
    'family_history_diabetes', 'hypertension_history', 'cardiovascular_history'
]

# Categorical features (strings that require encoding)
categorical_features = [
    'gender', 'ethnicity', 'education_level',
    'income_level', 'smoking_status', 'employment_status'
]




from sklearn.preprocessing import StandardScaler, OneHotEncoder  # scaling + encoding tools
from sklearn.compose import ColumnTransformer                    # for combining multiple transformers
from sklearn.pipeline import Pipeline                            # building a preprocessing pipeline





# Transformer for numerical features: Standardization
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # scales features to mean=0, std=1
])

# Transformer for categorical features: One-hot encoding
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # handles new/unseen categories safely
])

# Transformer for binary features: pass them through without changes
binary_transformer = 'passthrough'  # binary values are already numeric (0/1)





preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),     # apply scaling to numerical columns
        ('cat', categorical_transformer, categorical_features),  # apply one-hot encoding
        ('bin', binary_transformer, binary_features)             # keep binary columns unchanged
    ],
    remainder='drop'  # drop id column and any other unused columns
)



# Step 3.6: Fit the preprocessing pipeline and transform the training features
X_preprocessed = preprocessor.fit_transform(X)  # fit on training data and transform it

# Step 3.7: Transform test features using the same pipeline
X_test_preprocessed = preprocessor.transform(X_test)  # no fit here, only transform



# Step 3.8: Print shape of transformed data for verification
print("Shape of preprocessed training data:", X_preprocessed.shape)
print("Shape of preprocessed test data:", X_test_preprocessed.shape)

# This confirms that the preprocessing was applied correctly and generates the final feature matrix.



# Step 4.1: Import required libraries

from sklearn.model_selection import train_test_split   # used for splitting data
from sklearn.linear_model import LogisticRegression     # baseline classification model
from sklearn.metrics import roc_auc_score               # competition metric: AUC
import numpy as np                                      # numerical operations


# Step 4.2: Split the dataset into training and validation sets

# test_size=0.2 => 80% training, 20% validation
# random_state ensures reproducibility
X_train, X_val, y_train, y_val = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42
)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)



# Step 4.3: Train a baseline Logistic Regression model

# logistic regression needs solver="liblinear" or "lbfgs" depending on dataset size
log_reg = LogisticRegression(max_iter=500)  # increase max_iter to ensure convergence

# Train on the training set
log_reg.fit(X_train, y_train)



# Step 4.4: Predict probabilities on the validation set

# predict_proba returns two columns: [probability of class 0, probability of class 1]
# We take probability of class 1
y_val_pred_proba = log_reg.predict_proba(X_val)[:, 1]



# Step 4.5: Evaluate model using AUC (Area Under the ROC Curve)

auc_log_reg = roc_auc_score(y_val, y_val_pred_proba)
print("Baseline Logistic Regression AUC:", auc_log_reg)



# Step 5.1: Import required libraries
from sklearn.model_selection import KFold, cross_val_score              # for cross-validation
from sklearn.linear_model import LogisticRegression                     # baseline model
from sklearn.ensemble import RandomForestClassifier                     # tree-based model
from sklearn.metrics import roc_auc_score                               # evaluation metric
from xgboost import XGBClassifier                                       # powerful gradient boosting model
import numpy as np                                                      # numerical operations



# Step 5.2: Define K-Fold cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)  
# n_splits=5: 5 folds
# shuffle=True ensures randomness
# random_state=42 ensures reproducibility



# Step 5.3: Logistic Regression Cross-Validation

log_reg = LogisticRegression(max_iter=500)

log_reg_cv_scores = cross_val_score(
    log_reg, 
    X_preprocessed, 
    y,
    scoring="roc_auc",   # AUC score
    cv=kf                # 5-fold CV
)

print("Logistic Regression CV AUC (per fold):", log_reg_cv_scores)
print("Logistic Regression Mean CV AUC:", log_reg_cv_scores.mean())



# Step 5.4: Random Forest Cross-Validation

rf_clf = RandomForestClassifier(
    n_estimators=200,       # number of trees
    random_state=42,
    n_jobs=-1               # use all CPU cores
)

rf_cv_scores = cross_val_score(
    rf_clf, 
    X_preprocessed, 
    y,
    scoring="roc_auc",
    cv=kf
)

print("Random Forest CV AUC (per fold):", rf_cv_scores)
print("Random Forest Mean CV AUC:", rf_cv_scores.mean())



# Step 5.5: XGBoost Cross-Validation

xgb_clf = XGBClassifier(
    n_estimators=500,          # number of boosting rounds
    learning_rate=0.05,        # step size
    max_depth=6,               # depth of trees
    subsample=0.8,             # use 80% of samples per tree
    colsample_bytree=0.8,      # use 80% of features per tree
    eval_metric='auc',         # required for AUC
    random_state=42,
    n_jobs=-1
)

xgb_cv_scores = cross_val_score(
    xgb_clf, 
    X_preprocessed, 
    y,
    scoring="roc_auc",
    cv=kf
)

print("XGBoost CV AUC (per fold):", xgb_cv_scores)
print("XGBoost Mean CV AUC:", xgb_cv_scores.mean())



# Step 6.1: Store Mean CV AUC scores for all models (computed dynamically)

log_reg_auc = log_reg_cv_scores.mean()        # Mean AUC for Logistic Regression
rf_auc = rf_cv_scores.mean()                  # Mean AUC for Random Forest
xgb_auc = xgb_cv_scores.mean()                # Mean AUC for XGBoost

# Create a result dictionary
model_results = {
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Mean CV AUC": [log_reg_auc, rf_auc, xgb_auc]
}

# Convert to DataFrame
results_df = pd.DataFrame(model_results)

# Sort models by AUC in descending order
results_df_sorted = results_df.sort_values(by="Mean CV AUC", ascending=False)

results_df_sorted



# Step 7.1: Import required tools
from sklearn.model_selection import RandomizedSearchCV   # efficient hyperparameter search
from xgboost import XGBClassifier                        # gradient boosting model
import numpy as np


# Step 7.2: Define the XGBoost model (base)
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    n_jobs=-1,
    random_state=42
)



# Step 7.3: Define the hyperparameter search space

param_grid = {
    "n_estimators": [300, 500, 700, 900],            # number of boosting rounds
    "max_depth": [3, 4, 5, 6, 7],                    # tree depth
    "learning_rate": [0.01, 0.03, 0.05, 0.1],        # shrinkage rate
    "subsample": [0.6, 0.7, 0.8, 1.0],               # row sampling
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],        # feature sampling
    "gamma": [0, 1, 5],                              # minimum loss reduction
    "min_child_weight": [1, 3, 5],                   # minimum sum of instance weight
}



# Step 7.4: Set up RandomizedSearchCV

xgb_random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=20,                      # try 20 random combinations
    scoring="roc_auc",              # AUC metric
    cv=3,                           # 3-fold CV for speed
    verbose=2,                      # print progress
    random_state=42,
    n_jobs=-1                       # use all CPU cores
)



# Step 7.5: Run the hyperparameter search
xgb_random_search.fit(X_preprocessed, y)



# Step 7.6: Show the best parameters found
print("Best Parameters:", xgb_random_search.best_params_)
print("Best AUC Score from CV:", xgb_random_search.best_score_)



# Step 8.1: Import XGBoost
from xgboost import XGBClassifier



# Step 8.2: Define the final model using the best parameters

best_xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=700,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=1,
    gamma=1,
    n_jobs=-1,
    random_state=42
)



# Step 8.3: Train the final XGBoost model on the full dataset
best_xgb.fit(X_preprocessed, y)



# Step 8.3 (new): Train on training split and compute AUC
best_xgb.fit(X_train, y_train)

# Predict on validation split
y_val_pred = best_xgb.predict_proba(X_val)[:, 1]

# Compute AUC
final_auc = roc_auc_score(y_val, y_val_pred)
print("Final Validation AUC (Tuned XGBoost):", final_auc)



# Step 8.4: Predict probabilities on the test dataset
test_pred_prob = best_xgb.predict_proba(X_test_preprocessed)[:, 1]



# Step 8.5: Create the submission DataFrame
submission_df = pd.DataFrame({
    "id": test_df["id"],
    "diagnosed_diabetes": test_pred_prob
})



# Step 8.6: Save the submission file
submission_df.to_csv("submission.csv", index=False)

print("Submission file created successfully!")
submission_df.head()





