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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier



# Load the dataset
train_pre = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_pre = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# Display basic information about the dataset
print("Training Data Info:")
print(train_pre.info())
print("\nTest Data Info:")
print(test_pre.info())



# Check for missing values
print("\nMissing Values in Training Data:")
print(train_pre.isnull().sum())
print("\nMissing Values in Test Data:")
print(test_pre.isnull().sum())



# Separate features and target in the training data
X_train = train_pre.drop(columns=['rainfall'])  # Features
y_train = train_pre['rainfall']  # Target

# Handle missing values using KNN Imputer
imputer = KNNImputer(n_neighbors=5)
train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
test = pd.DataFrame(imputer.transform(test_pre), columns=test_pre.columns)

# Add the target variable back to the training data
train['rainfall'] = y_train

# Check for missing values
print("\nMissing Values in Training Data:")
print(train.isnull().sum())
print("\nMissing Values in Test Data:")
print(test.isnull().sum())



# Distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='rainfall', data=train)
plt.title("Distribution of Rainfall (Target Variable)")
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
corr = train.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Distribution of numerical features
numerical_features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine']
train[numerical_features].hist(bins=20, figsize=(15, 10))
plt.suptitle("Distribution of Numerical Features")
plt.show()



# Create new features from 'day'
def create_season(day):
    if day <= 91:
        return 1  # Winter
    elif day <= 182:
        return 2  # Spring
    elif day <= 273:
        return 3  # Summer
    else:
        return 4  # Fall

train['season'] = train['day'].apply(create_season)
test['season'] = test['day'].apply(create_season)

# Standardize numerical features
scaler = StandardScaler()
train[numerical_features] = scaler.fit_transform(train[numerical_features])
test[numerical_features] = scaler.transform(test[numerical_features])

# Feature importance using Mutual Information
X = train.drop(columns=['rainfall', 'id'])
y = train['rainfall']
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores = pd.Series(mi_scores, index=X.columns, name="MI Scores")
print("\nFeature Importance (Mutual Information Scores):")
print(mi_scores.sort_values(ascending=False))



# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)



# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train a Random Forest Classifier with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
print(f"\nBest Parameters: {best_params}")
best_model = grid_search.best_estimator_



# Evaluate the model on the validation set
y_pred = best_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print(f"\nValidation AUC: {auc:.4f}")

# Cross-validation score
cv_scores = cross_val_score(best_model, X_res, y_res, cv=5, scoring='roc_auc')
print(f"\nCross-Validation AUC: {np.mean(cv_scores):.4f}")



# Generate predictions for the test set
test_features = test.drop(columns=['id'])
test_preds = best_model.predict_proba(test_features)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({'id': test['id'], 'rainfall': test_preds})
submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as 'submission.csv'.")


