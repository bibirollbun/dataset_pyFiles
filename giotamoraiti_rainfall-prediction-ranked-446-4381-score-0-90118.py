import os
print(os.listdir("/kaggle/input"))

import warnings
# Suppress specific FutureWarnings related to inf handling in Seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


# Adjust the filename based on the competition dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# Display dataframe
train_df


# See Features
print(train_df.columns)
print("Number of useful features", len(train_df.columns)-2) # Exclude instance id and target


# Print the smallest and largest value of each feature
print("\nSmallest and Largest Values of Each Feature:")

# Iterate through each column and print min and max values
for column in train_df.columns:
    min_value = train_df[column].min()
    max_value = train_df[column].max()
    print(f"{column}: Min = {min_value}, Max = {max_value}")


# Check for NaN or infinite values in the DataFrame
print(train_df.isna().sum())  # Count of NaN values per column


# Check if balanced
print("Number of total instances instances are: ", len(train_df))
# print("Number of non rainfall instances are: ", (train_df["rainfall"] == 0).sum())
# print("Number of rainfall instances are: ", (train_df["rainfall"] == 1).sum())

train_df["rainfall"].value_counts()


plt.figure(figsize=(6, 4))
sns.countplot(x='rainfall', data=train_df, palette='Set1')
plt.title('Rainfall Class Distribution')
plt.xlabel('Rainfall')
plt.ylabel('Count')
plt.show()


# Histograms for Distribution of Numerical Features
train_df.hist(figsize=(12, 10), bins=30)
plt.suptitle('Distribution of Numerical Features')
plt.tight_layout()
plt.show()


# Scatterplot Matrix - Visualize the pairwise relationships between features to understand any correlation
sns.pairplot(train_df[['pressure', 'maxtemp', 'temparature', 'humidity', 'windspeed']])
plt.suptitle('Pairplot for Selected Features', y=1.02)
plt.show()


# Correlation Heatmap to understand the linear relationships between numerical variables
plt.figure(figsize=(10, 8))
corr = train_df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# Rainfall vs Cloud and Humidity
plt.figure(figsize=(10, 6))
sns.scatterplot(x='cloud', y='humidity', hue='rainfall', data=train_df, palette='coolwarm')
plt.title('Rainfall vs Cloud and Humidity')
plt.show()


# # Rainfall vs Cloud and Sunshine
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sunshine', y='cloud', hue='rainfall', data=train_df, palette='coolwarm')
plt.title('Rainfall vs Cloud and Sunshine')
plt.show()


# Plotting Temperature Over Time using Day Numbers as X-axis
plt.figure(figsize=(12, 6))
plt.scatter(train_df['day'], train_df['temparature'], color='blue', alpha=0.7)
plt.title('Temperature Over Time (Day-wise)')
plt.xlabel('Day Number')
plt.ylabel('Temperature')
plt.tight_layout()  # Ensure the plot fits well
plt.show()


# Separate features
features = train_df.iloc[:, 1:-1] # All columns except the first and last

# Separate targets
target = train_df['rainfall']


test_df


# Check for NaN or infinite values in the DataFrame
print(test_df.isna().sum())  # Count of NaN values per column


# Fill missing values with the mean of the 'winddirection' column
value = test_df['winddirection'].mean()
test_df['winddirection'] = test_df['winddirection'].fillna(value)
value


test_features = test_df.iloc[:, 1:] # All columns except the first and last
test_features


# Scale features
scaler = StandardScaler()

# Fit the scaler on training data and transform both train and test data
features_scaled = scaler.fit_transform(features)
test_features_scaled = scaler.transform(test_features)


# Define cross-validation strategy
random_state=26
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)  # 5-Fold CV


# Create a Random Forest Regressor model
rf_model = RandomForestClassifier(n_estimators=70, random_state=random_state)

# Perform cross-validation
acc = cross_val_score(rf_model, features, target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(rf_model, features, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Default Parameters
rf_model = RandomForestClassifier(random_state=random_state)
print(rf_model.get_params())


# Define parameter grid
param_grid = {
    'n_estimators': [40, 50, 60, 70],
    'max_depth': [None, 3, 5],
    'min_samples_split': [2, 5, 7],
    'class_weight': ['balanced', {0: 4, 1: 1}, {0: 2, 1: 1}]
}

rf_model = RandomForestClassifier(random_state=random_state)

# Perform Grid Search
grid_search = GridSearchCV(rf_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(features, target)

# Print best parameters and accuracy
print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", round(grid_search.best_score_, 4))


# Create a Random Forest Regressor model
rf_model = RandomForestClassifier(n_estimators=70, max_depth=5, min_samples_split=2, class_weight='balanced', random_state=random_state)

# Perform cross-validation
acc = cross_val_score(rf_model, features, target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(rf_model, features, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Create the XGBoost Classifier model
xgb_model = xgb.XGBClassifier(n_estimators=70, random_state=random_state)

# Perform cross-validation
acc = cross_val_score(xgb_model, features, target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(xgb_model, features, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Default Parameters
xgb_model = xgb.XGBClassifier(random_state=random_state)
print(xgb_model.get_params())


# Define parameter grid
param_grid = {
    'n_estimators': [50, 70], 
    'max_depth': [3, 5],    
    'learning_rate': [0.05, 0.1],  
    'subsample': [0.7, 0.8, 1.0],     
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_lambda': [1, 5, 10],  
    'scale_pos_weight': [1, 2, 4]
}

xgb_model = xgb.XGBClassifier(random_state=random_state)

# Perform Grid Search
grid_search = GridSearchCV(xgb_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(features, target)

# Print best parameters and accuracy
print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", round(grid_search.best_score_, 4))


# Create the XGBoost Classifier model
xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, colsample_bytree=0.7, learning_rate=0.1, 
                              reg_lambda=5, scale_pos_weight=2, subsample=0.7, random_state=random_state)

# Perform cross-validation
acc = cross_val_score(xgb_model, features, target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(xgb_model, features, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Initialize Support Vector Classifier (SVC)
svm_model = SVC(kernel='linear', C=0.35, gamma='scale', random_state=random_state)

# Perform cross-validation
acc = cross_val_score(svm_model, features_scaled, target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(svm_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Default Parameters
svm_model = SVC(random_state=random_state)
print(svm_model.get_params())


# Define the Logistic Regression model
svm_model = SVC(random_state=random_state)

param_grid = {
    'C': [0.01, 0.1, 0.35, 1, 10, 100],              # Regularization parameter
    'gamma': ['scale', 'auto'],                      # Kernel coefficient for ‘rbf’, ‘poly’, and ‘sigmoid’ kernels
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Linear kernel
    'class_weight': [None, 'balanced'],              # Handle class imbalance
}


# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(svm_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(features_scaled, target)

# Print best parameters and accuracy
print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", round(grid_search.best_score_, 4))


# Initialize Support Vector Classifier (SVC)
svm_model = SVC(kernel='linear', C=0.01, gamma='scale', random_state=random_state)

# Perform cross-validation
acc = cross_val_score(svm_model, features_scaled, target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(svm_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Define the Logistic Regression model
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state)

# Perform cross-validation
acc = cross_val_score(log_reg_model, features_scaled, target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(log_reg_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Default Parameters
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state)
print(log_reg_model.get_params())


# Define the Logistic Regression model
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state)

# Define the parameter grid
param_grid = {
    'C': [0.01, 0.1, 0.5, 0.7, 1, 1.5],  # More values for regularization strength
    'penalty': ['l2'],  # L1 = Lasso, L2 = Ridge
    'solver': ['liblinear', 'newton-cg', 'lbfgs', 'newton-cholesky'],
    'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 2, 1: 1}],
     'max_iter': [100, 250, 500, 1000, 5000, 10000]
}

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(log_reg_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(features_scaled, target)

# Print best parameters and accuracy
print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", round(grid_search.best_score_, 4))


# Define the Logistic Regression model
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state)

param_grid = [
    # 1) L1 or L2 penalty with liblinear solver
    {
        'penalty': ['l1', 'l2'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear'],  # liblinear supports only l1 or l2
        'class_weight': [None, 'balanced'],
        'max_iter': [50, 100, 300, 500]
    },
    # 2) L2 penalty with lbfgs, sag, saga
    {
        'penalty': ['l2'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'sag', 'saga', 'newton-cg'],  # these solvers handle l2
        'class_weight': [None, 'balanced'],
        'max_iter': [10, 20, 30, 40, 50, 100, 300, 500]
    },
    # 3) Elasticnet penalty with saga solver (only combo that supports elasticnet)
    {
        'penalty': ['elasticnet'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['saga'],
        'class_weight': [None, 'balanced'],
        'max_iter': [50, 100, 300, 500],
        'l1_ratio': [0, 0.5, 1]  # only relevant for elasticnet
    }
]

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(log_reg_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(features_scaled, target)

# Print best parameters and accuracy
print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", round(grid_search.best_score_, 4))


# Define the Logistic Regression model
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state, C=0.01, max_iter=100, penalty='l2', class_weight={0: 1, 1: 2})

# Perform cross-validation
acc = cross_val_score(log_reg_model, features_scaled, target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(log_reg_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Define the Logistic Regression model
log_reg_model = LogisticRegression(solver='saga', random_state=random_state, C=0.1, max_iter=50, penalty='elasticnet', l1_ratio=0.5, class_weight=None)

# Perform cross-validation
acc = cross_val_score(log_reg_model, features_scaled, target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(log_reg_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Create the CatBoost Classifier model
catboost_model = CatBoostClassifier(iterations=70, random_state=random_state, learning_rate=0.1, depth=6, verbose=0)

# Perform cross-validation for accuracy
acc = cross_val_score(catboost_model, features, target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(), 4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(catboost_model, features, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(), 4))


# Create the CatBoost Classifier model
catboost_model = CatBoostClassifier(random_state=random_state, verbose=0)

# Define the parameter grid to search over
param_grid = {
    'iterations': [70, 100, 120],  # Number of trees
    'learning_rate': [0.05, 0.1, 0.2],  # Learning rate
    'depth': [4, 6, 8],  # Depth of trees
}

# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=catboost_model, param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)

# Fit GridSearchCV on the data
grid_search.fit(features, target)

# Get the best parameters and best score
print("Best hyperparameters:", grid_search.best_params_)
print("Best AUC-ROC score:", grid_search.best_score_)


# Create the CatBoost Classifier model
catboost_model = CatBoostClassifier(iterations=100, random_state=random_state, learning_rate=0.1, depth=4, verbose=0)

# Perform cross-validation for accuracy
acc = cross_val_score(catboost_model, features, target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(), 4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(catboost_model, features, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(), 4))


# Create models
rf_model = RandomForestClassifier(n_estimators=70, max_depth=5, min_samples_split=2, class_weight='balanced', random_state=random_state)
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state)
svm_model = SVC(kernel='linear', C=0.01, gamma='scale', probability=True, random_state=random_state)
xgb_model = xgb.XGBClassifier(n_estimators=70, random_state=random_state)
catboost_model = CatBoostClassifier(iterations=70, random_state=random_state, learning_rate=0.1, depth=6, verbose=0, l2_leaf_reg=1.5)


# Create a VotingClassifier (using soft voting, which averages the predicted probabilities)
voting_model = VotingClassifier(estimators=[('rf', rf_model), ('log_reg', log_reg_model), ('svm', svm_model)], voting='soft')

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(voting_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Create a VotingClassifier (using soft voting, which averages the predicted probabilities)
voting_model = VotingClassifier(estimators=[('log_reg', log_reg_model), ('svm', svm_model)], voting='soft')

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(voting_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Create a VotingClassifier (using soft voting, which averages the predicted probabilities)
voting_model = VotingClassifier(estimators=[('rf', rf_model), ('svm', svm_model)], voting='soft')

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(voting_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Create a VotingClassifier (using soft voting, which averages the predicted probabilities)
voting_model = VotingClassifier(estimators=[('rf', rf_model), ('log_reg', log_reg_model)], voting='soft')

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(voting_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Create a VotingClassifier (using soft voting, which averages the predicted probabilities)
voting_model = VotingClassifier(estimators=[('rf', rf_model), ('log_reg', log_reg_model), ('svm', svm_model), ('xgb', xgb_model)], voting='soft')

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(voting_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Create a VotingClassifier (using soft voting, which averages the predicted probabilities)
voting_model = VotingClassifier(estimators=[('rf', rf_model), ('log_reg', log_reg_model), ('svm', svm_model), ('xgb', xgb_model), ('cb', catboost_model)], voting='soft')

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(voting_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


rf_model = RandomForestClassifier(n_estimators=110, max_depth=8, min_samples_split=2, random_state=random_state)
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state)

# Create a VotingClassifier (using soft voting, which averages the predicted probabilities)
voting_model = VotingClassifier(estimators=[('rf', rf_model), ('log_reg', log_reg_model)], voting='soft')

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(voting_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Meta-model is typically a simple model like logistic regression
stack_model = StackingClassifier(
    estimators=[('rf', rf_model), ('log_reg', log_reg_model), ('svm', svm_model)],
    final_estimator=LogisticRegression()
)

auc_scores = cross_val_score(stack_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score (stacking):", round(auc_scores.mean(), 4))


# Meta-model is typically a simple model like logistic regression
stack_model = StackingClassifier(
    estimators=[('rf', rf_model), ('log_reg', log_reg_model)],
    final_estimator=LogisticRegression()
)

auc_scores = cross_val_score(stack_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score (stacking):", round(auc_scores.mean(), 4))


# Meta-model is typically a simple model like logistic regression
stack_model = StackingClassifier(
    estimators=[('rf', rf_model), ('log_reg', log_reg_model), ('svm', svm_model), ('xgb', xgb_model)],
    final_estimator=LogisticRegression()
)

auc_scores = cross_val_score(stack_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score (stacking):", round(auc_scores.mean(), 4))


# Meta-model is typically a simple model like logistic regression
stack_model = StackingClassifier(
    estimators=[('svm', svm_model), ('log_reg', log_reg_model)],
    final_estimator=LogisticRegression()
)

auc_scores = cross_val_score(stack_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score (stacking):", round(auc_scores.mean(), 4))


# Meta-model is typically a simple model like logistic regression
stack_model = StackingClassifier(
    estimators=[('rf', rf_model), ('log_reg', log_reg_model), ('svm', svm_model), ('xgb', xgb_model), ('cb', catboost_model)],
    final_estimator=LogisticRegression()
)

auc_scores = cross_val_score(stack_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score (stacking):", round(auc_scores.mean(), 4))


# some new features
features_engin = features.copy()
test_features_engin = test_features.copy()

features_engin['humidity_cloud_interaction'] = features['humidity'] * features['cloud']
features_engin['humidity_sunshine_interaction'] = features['humidity'] * features['sunshine']
features_engin['cloud_sunshine_ratio'] = features['cloud'] / (features['sunshine'] + 1e-5)
features_engin['relative_dryness'] = 100 - features['humidity']
features_engin['sunshine_percentage'] = features['sunshine'] / (features['sunshine'] + features['cloud'] + 1e-5)
features_engin['weather_index'] = (0.4 * features['humidity']) + (0.3 * features['cloud']) - (0.3 * features['sunshine'])

test_features_engin['humidity_cloud_interaction'] = test_features_engin['humidity'] * test_features_engin['cloud']
test_features_engin['humidity_sunshine_interaction'] = test_features_engin['humidity'] * test_features_engin['sunshine']
test_features_engin['cloud_sunshine_ratio'] = test_features_engin['cloud'] / (test_features_engin['sunshine'] + 1e-5)
test_features_engin['relative_dryness'] = 100 - test_features_engin['humidity']
test_features_engin['sunshine_percentage'] = test_features_engin['sunshine'] / (test_features_engin['sunshine'] + test_features_engin['cloud'] + 1e-5)
test_features_engin['weather_index'] = (0.4 * test_features_engin['humidity']) + (0.3 * test_features_engin['cloud']) - (0.3 * test_features_engin['sunshine'])


features_engin


# Define the Logistic Regression model
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state)

param_grid = [
    # 1) L1 or L2 penalty with liblinear solver
    {
        'penalty': ['l1', 'l2'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear'],  # liblinear supports only l1 or l2
        'class_weight': [None, 'balanced'],
        'max_iter': [50, 100, 300, 500]
    },
    # 2) L2 penalty with lbfgs, sag, saga
    {
        'penalty': ['l2'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'sag', 'saga', 'newton-cg'],  # these solvers handle l2
        'class_weight': [None, 'balanced'],
        'max_iter': [10, 20, 30, 40, 50, 100, 300, 500]
    },
    # 3) Elasticnet penalty with saga solver (only combo that supports elasticnet)
    {
        'penalty': ['elasticnet'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['saga'],
        'class_weight': [None, 'balanced'],
        'max_iter': [50, 100, 300, 500],
        'l1_ratio': [0, 0.5, 1]  # only relevant for elasticnet
    }
]

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(log_reg_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(features_engin, target)

# Print best parameters and accuracy
print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", round(grid_search.best_score_, 4))


# Scale features
scaler = StandardScaler()

# Fit the scaler on training data and transform both train and test data
features_scaled = scaler.fit_transform(features_engin)
test_features_scaled = scaler.transform(test_features_engin)


# Define the Logistic Regression model
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state, C=0.1, max_iter=50, penalty='l1', class_weight=None)

# Perform cross-validation
acc = cross_val_score(log_reg_model, features_scaled, target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(log_reg_model, features_scaled, target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Adjust the filename based on the competition dataset
new_df = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
new_df


new_feat1 = new_df.iloc[:, :8]
new_feat2= new_df.iloc[:, 9:]

# Merge the two DataFrames along columns (axis=1)
merged_df = pd.concat([new_feat1, new_feat2], axis=1)
merged_df


new_target = new_df.iloc[:, 8]

# Convert categorical values to binary (e.g., 'yes' -> 1, 'no' -> 0)
new_target_binary = new_target.map({'yes': 1, 'no': 0})
new_target_binary


features


# Check the column names of both DataFrames
print(merged_df.columns)
print(features.columns)


# Strip spaces from column names in both DataFrames
merged_df.columns = merged_df.columns.str.strip()
features.columns = features.columns.str.strip()

# Now merge the DataFrames vertically
extended_features = pd.concat([merged_df, features], axis=0, ignore_index=True)

# Display the merged result
extended_features


# Now merge the DataFrames vertically
extended_target = pd.concat([new_target_binary, target], axis=0, ignore_index=True)
extended_target


# Check for NaN or infinite values in the DataFrame
print(extended_features.isna().sum())  # Count of NaN values per column


# Fill missing values with the mean of the 'winddirection' column
value = extended_features['winddirection'].mean()
extended_features['winddirection'] = extended_features['winddirection'].fillna(value)

value = extended_features['windspeed'].mean()
extended_features['windspeed'] = extended_features['windspeed'].fillna(value)


# some new features
features_engin = extended_features.copy()
test_features_engin = test_features.copy()

features_engin['humidity_cloud_interaction'] = extended_features['humidity'] * extended_features['cloud']
features_engin['humidity_sunshine_interaction'] = extended_features['humidity'] * extended_features['sunshine']
features_engin['cloud_sunshine_ratio'] = extended_features['cloud'] / (extended_features['sunshine'] + 1e-5)
features_engin['relative_dryness'] = 100 - extended_features['humidity']
features_engin['sunshine_percentage'] = extended_features['sunshine'] / (extended_features['sunshine'] + extended_features['cloud'] + 1e-5)
features_engin['weather_index'] = (0.4 * extended_features['humidity']) + (0.3 * extended_features['cloud']) - (0.3 * extended_features['sunshine'])

test_features_engin['humidity_cloud_interaction'] = test_features_engin['humidity'] * test_features_engin['cloud']
test_features_engin['humidity_sunshine_interaction'] = test_features_engin['humidity'] * test_features_engin['sunshine']
test_features_engin['cloud_sunshine_ratio'] = test_features_engin['cloud'] / (test_features_engin['sunshine'] + 1e-5)
test_features_engin['relative_dryness'] = 100 - test_features_engin['humidity']
test_features_engin['sunshine_percentage'] = test_features_engin['sunshine'] / (test_features_engin['sunshine'] + test_features_engin['cloud'] + 1e-5)
test_features_engin['weather_index'] = (0.4 * test_features_engin['humidity']) + (0.3 * test_features_engin['cloud']) - (0.3 * test_features_engin['sunshine'])


# Scale features
scaler = StandardScaler()

# Fit the scaler on training data and transform both train and test data
ext_features_scaled = scaler.fit_transform(features_engin)
ext_features_scaled_test = scaler.fit_transform(test_features_engin)


# Define the Logistic Regression model
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state)

# Perform cross-validation
acc = cross_val_score(log_reg_model, ext_features_scaled, extended_target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(log_reg_model, ext_features_scaled, extended_target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Define the parameter grid
param_grid = {
    'C': [0.01, 0.1, 0.5, 0.7, 1, 1.5],  # More values for regularization strength
    'penalty': ['l2'],  # L1 = Lasso, L2 = Ridge
    'solver': ['liblinear', 'newton-cg', 'lbfgs', 'newton-cholesky'],
    'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 2, 1: 1}],
     'max_iter': [100, 250, 500, 1000, 5000, 10000]
}

log_reg_model = LogisticRegression(random_state=random_state)

# Perform Grid Search
grid_search = GridSearchCV(log_reg_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(ext_features_scaled, extended_target)

# Print best parameters and accuracy
print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", round(grid_search.best_score_, 4))


# Create a Random Forest Regressor model
rf_model = RandomForestClassifier(n_estimators=110, max_depth=8, min_samples_split=2, random_state=random_state)

# Perform cross-validation
acc = cross_val_score(rf_model, ext_features_scaled, extended_target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(rf_model, ext_features_scaled, extended_target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


param_grid = {
    'n_estimators': [90, 100, 110, 120, 130, 140, 150],
    'max_depth': [None, 6, 7, 8, 9, 10]
}

rf_model = RandomForestClassifier(random_state=random_state)

# Perform Grid Search
grid_search = GridSearchCV(rf_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(extended_features, extended_target)

# Print best parameters and accuracy
print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", round(grid_search.best_score_, 4))


# Create the XGBoost Classifier model
xgb_model = xgb.XGBClassifier(n_estimators=70, random_state=random_state)

# Perform cross-validation
acc = cross_val_score(xgb_model, ext_features_scaled, extended_target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(xgb_model, ext_features_scaled, extended_target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Create the CatBoost Classifier model
catboost_model = CatBoostClassifier(iterations=70, random_state=random_state, learning_rate=0.1, depth=6, verbose=0, l2_leaf_reg=1.5)

# Perform cross-validation for accuracy
acc = cross_val_score(catboost_model, ext_features_scaled, extended_target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(), 4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(catboost_model, extended_features, extended_target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(), 4))


# Define the Logistic Regression model
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state)

param_grid = [
    # 1) L1 or L2 penalty with liblinear solver
    {
        'penalty': ['l1', 'l2'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear'],  # liblinear supports only l1 or l2
        'class_weight': [None, 'balanced'],
        'max_iter': [50, 100, 300, 500]
    },
    # 2) L2 penalty with lbfgs, sag, saga
    {
        'penalty': ['l2'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'sag', 'saga', 'newton-cg'],  # these solvers handle l2
        'class_weight': [None, 'balanced'],
        'max_iter': [10, 20, 30, 40, 50, 100, 300, 500]
    },
    # 3) Elasticnet penalty with saga solver (only combo that supports elasticnet)
    {
        'penalty': ['elasticnet'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['saga'],
        'class_weight': [None, 'balanced'],
        'max_iter': [50, 100, 300, 500],
        'l1_ratio': [0, 0.5, 1]  # only relevant for elasticnet
    }
]

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(log_reg_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(ext_features_scaled, extended_target)

# Print best parameters and accuracy
print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", round(grid_search.best_score_, 4))


# Define the Logistic Regression model
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state, C=0.1, max_iter=50, penalty='l2', class_weight=None)

# Perform cross-validation
acc = cross_val_score(log_reg_model, ext_features_scaled, extended_target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(log_reg_model, ext_features_scaled, extended_target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


# Define the Logistic Regression model
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state, C=0.1, max_iter=50, penalty='l1', class_weight=None)

# Perform cross-validation
acc = cross_val_score(log_reg_model, ext_features_scaled, extended_target, cv=cv, scoring='accuracy')
print("Mean accuracy is:", round(acc.mean(),4))

# Perform cross-validation with AUC-ROC scoring
auc_scores = cross_val_score(log_reg_model, ext_features_scaled, extended_target, cv=cv, scoring='roc_auc')
print("Mean roc auc score is:", round(auc_scores.mean(),4))


rf_model = RandomForestClassifier(n_estimators=110, max_depth=8, min_samples_split=2, random_state=random_state)
rf_model.fit(extended_features, extended_target)

# Generate predictions
y_pred = rf_model.predict(test_features)
y_prob = rf_model.predict_proba(test_features)[:, 1]


xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, colsample_bytree=0.7, learning_rate=0.1, 
                              reg_lambda=5, scale_pos_weight=2, subsample=0.7, random_state=random_state)
xgb_model.fit(features, target)

# Generate predictions
y_pred = xgb_model.predict(test_features)
y_prob = xgb_model.predict_proba(test_features)[:, 1]


# Create the CatBoost Classifier model
catboost_model = CatBoostClassifier(iterations=70, random_state=random_state, learning_rate=0.1, depth=6, verbose=0, l2_leaf_reg=1.5)
catboost_model.fit(features, target)

# Generate predictions
y_pred = catboost_model.predict(test_features)
y_prob = catboost_model.predict_proba(test_features)[:, 1]


# Create a Linear Regression Model model
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state, C=0.1, max_iter=50, penalty='l1', class_weight=None)
log_reg_model.fit(features_scaled, target)

# Generate probabilities for each class (class 0 and class 1)
y_prob = log_reg_model.predict_proba(test_features_scaled)
y_prob = y_prob[:, 1]


# Create a Linear Regression Model model
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state, C=0.1, max_iter=50, penalty='l1', class_weight=None)
log_reg_model.fit(ext_features_scaled, extended_target)

# Generate probabilities for each class (class 0 and class 1)
y_prob = log_reg_model.predict_proba(test_features_scaled)
y_prob = y_prob[:, 1]


# Create models
rf_model = RandomForestClassifier(n_estimators=110, max_depth=8, min_samples_split=2, random_state=random_state)
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state)
svm_model = SVC(kernel='linear', C=0.01, gamma='scale', probability=True, random_state=random_state)
xgb_model = xgb.XGBClassifier(random_state=random_state)
catboost_model = CatBoostClassifier(random_state=random_state, verbose=0)


# Create a VotingClassifier (using soft voting, which averages the predicted probabilities)
voting_model = VotingClassifier(estimators=[('rf', rf_model), ('log_reg', log_reg_model), ('svm', svm_model), ('xgb', xgb_model), ('cb', catboost_model)], voting='soft')
voting_model.fit(ext_features_scaled, extended_target)

# Generate probabilities 
y_prob = voting_model.predict_proba(test_features_scaled)
y_prob = y_prob[:, 1]


from fastai.tabular.all import *
from fastai.metrics import RocAucBinary


df = pd.concat([extended_features, extended_target], axis=1)
df


df = pd.concat([features_engin, extended_target], axis=1)
df


# Create TabularPandas object
procs = [Categorify, FillMissing, Normalize]

# Define dependent variable and column types
dep_var = 'rainfall'  # Target
cont_names = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 
              'dewpoint', 'humidity', 'cloud', 'sunshine', 
              'winddirection', 'windspeed', 'humidity_cloud_interaction', 'humidity_sunshine_interaction', 'cloud_sunshine_ratio', 'relative_dryness', 'sunshine_percentage', 'weather_index']  # All continuous columns
cat_names = []  # No categorical columns
splits = RandomSplitter(valid_pct=0.2, seed=26)(range_of(df))

# Create TabularPandas object
to = TabularPandas(df, procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=dep_var, y_block=CategoryBlock(), splits=splits)

# Convert to DataLoaders
dls = to.dataloaders(bs=64)


# Create a TabularLearner
learn = tabular_learner(
    dls,
    layers=[128, 256, 512],  
    metrics=[accuracy, RocAucBinary(), Precision(), Recall()],
    loss_func=CrossEntropyLossFlat(),
    config={'ps': 0.5}  # 50% dropout in hidden layers
)

learn.summary()


# Find optimal lr
suggest_funcs = (minimum, steep, valley, slide)
lrs = learn.lr_find(suggest_funcs=suggest_funcs)

slice(lrs.valley)


# Train
learn.fit_one_cycle(20, lr_max=slice(lrs.valley))
learn.recorder.plot_loss()


# Confusion matrix
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# Generate predictions
test_dl = dls.test_dl(test_features_engin)
preds, _ = learn.get_preds(dl=test_dl)
preds = preds[:, 1]


# Create a DataFrame with 'id' and the predicted rainfall values
predictions_df = pd.DataFrame({
    'id': test_df['id'],  # The 'id' column from the test DataFrame
    'rainfall': preds    # The predicted values
})

# Save the DataFrame to a CSV file with headers
predictions_df.to_csv('predictions_tablearn20_model_extdata_newfeats.csv', index=False)

# If you want to check the output
print(predictions_df.head())


# Create a Linear Regression Model model
log_reg_model = LogisticRegression(solver='liblinear', random_state=random_state, C=0.1, max_iter=50, penalty='l1', class_weight=None)
log_reg_model.fit(ext_features_scaled, extended_target)

# Generate probabilities for each class (class 0 and class 1)
y_prob = log_reg_model.predict_proba(test_features_scaled)
y_prob = y_prob[:, 1]


# Merge predictions using voting
preds_array = preds.detach().cpu().numpy()
soft_voted_preds = (y_prob + preds_array) / 2


# Create a DataFrame with 'id' and the predicted rainfall values
predictions_df = pd.DataFrame({
    'id': test_df['id'],  # The 'id' column from the test DataFrame
    'rainfall': soft_voted_preds    # The predicted values
})

# Save the DataFrame to a CSV file with headers
predictions_df.to_csv('predictions_tablearn20_logreg_ensemble_model_extdata_newfeats.csv', index=False)

# If you want to check the output
print(predictions_df.head())

