# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 06:00:54 2025

Kaggle Playground Competition: Season 5 Episode 3

@author: Punit Bandi
"""
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# Plot visualizations for trends in train dataset
# Plot_Visual = 0: No plotting
# Plot_Visual = 1: Yes
Plot_Visuals = 0

###############################################################################
#                 READ TRAINING & TEST DATASET
###############################################################################
# Read train data from 'train.csv' file
rows_train = []
with open('train.csv', 'r') as file:
    reader = csv.DictReader(file)  # Uses the first row as column names
    for row in reader:
        rows_train.append(row)

# Rear test data from 'test.csv' file
rows_test = []
with open('test.csv', 'r') as file_test:
    reader = csv.DictReader(file_test)
    for row in reader:
        rows_test.append(row)

# Convert to DataFrame
train_original = pd.DataFrame(rows_train)
test_original = pd.DataFrame(rows_test)

# Use Sine and Cosine Encoding for cyclic feature 'day'
# Ensure 'day' column is numeric
train_original['day'] = pd.to_numeric(train_original['day'], errors='coerce')
train_original['sin_day'] = np.sin(2 * np.pi * train_original['day'] / 365)
train_original['cos_day'] = np.cos(2 * np.pi * train_original['day'] / 365)

test_original['day'] = pd.to_numeric(test_original['day'], errors='coerce')
test_original['sin_day'] = np.sin(2 * np.pi * test_original['day'] / 365)
test_original['cos_day'] = np.cos(2 * np.pi * test_original['day'] / 365)

print(f"Total number of rows in train set: {len(train_original)}")
print(f"Total number of rows in test set: {len(test_original)}")

###############################################################################
#       IDENTIFY NULL & MISSING DATA IN TRAINING & TEST DATASET
###############################################################################
# Identify rows with one or more empty strings
rows_with_empty_strings = (train_original == "").any(axis=1)
rows_with_empty_strings_test = (test_original == "").any(axis=1)

# Count rows with empty strings
count_rows_with_empty_strings = rows_with_empty_strings.sum()
Perc_rows_with_empty_strings = ((count_rows_with_empty_strings/len(train_original))*100).round(2)
print(f"Number of rows with missing data in train set: {count_rows_with_empty_strings} ({Perc_rows_with_empty_strings}%)")

count_rows_with_empty_strings_test = rows_with_empty_strings_test.sum()
Perc_rows_with_missing_strings_test = ((count_rows_with_empty_strings_test/len(test_original))*100).round(2)
print(f"Number of rows with missing data in test set: {count_rows_with_empty_strings_test} ({Perc_rows_with_missing_strings_test}%)")

empty_string_count = (train_original == '').sum()
empty_string_percentage = (empty_string_count / len(train_original)) * 100

if count_rows_with_empty_strings > 0:
    # Plot the bar chart to visualize missing value spread across various inputs
    plt.figure(figsize=(10, 6))  # Set the figure size
    empty_string_percentage.plot(kind='bar', color='skyblue')
    # Add labels and title
    plt.xlabel('Column Headers')
    plt.ylabel('Percentage of Missing Values')
    plt.title('Bar chart for percentage of missing values in train dataset')
    plt.xticks(rotation=90)  # Rotate x-axis labels if needed
    # Show the plot
    plt.tight_layout()
    plt.show()
###############################################################################
#                 VISUALIZE TRENDS IN TRAINING DATASET
###############################################################################
if Plot_Visuals==1:
    # Violin plot showing distribution and density of "num_sold" for each "country"
    sns.violinplot(x=train_original['rainfall'], y=train_original['pressure'])
#    plt.yscale('log')
    plt.title("Violin Plot of pressure vs. rainfall")
    plt.ylabel("pressure")
    plt.show()
    
    plt.figure(figsize=(8, 5))
    sns.histplot(train_original, x='maxtemp', hue='rainfall', kde=True, bins=30, alpha=0.5)
    plt.title('Distribution of maxtemp by rainfall')
    plt.show()
###############################################################################
#                 PREPARE DATA FOR TRAINING AND TESTING
###############################################################################
# split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(train_original.drop(columns=['id','day','rainfall']), train_original['rainfall'], test_size=0.20, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
###############################################################################
#                 LOGISTIC REGRESSION MODEL TRAINING
###############################################################################
# Hyperparameter Tuning
param_grid = {'C': [0.001, 0.005, 0.01, 0.1, 1, 10]}

grid = GridSearchCV(LogisticRegression(solver='liblinear', penalty='l2', random_state=42),
                    param_grid, scoring='roc_auc', cv=5)
grid.fit(X_train_scaled, y_train)

print("Best C:", grid.best_params_['C'])

logreg = LogisticRegression(
    solver='liblinear',  # 'saga' for large datasets
    penalty='l2',  # Use L2 regularization
    C=grid.best_params_['C'],  # Optimize this using GridSearchCV
    max_iter=500,  # Increase if convergence warning occurs
    class_weight='balanced',  # Use for imbalanced datasets
    random_state=42
)

# fit the model with data
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)

# Model Evaluation using Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cnf_matrix)

# Generate a classification report
print(classification_report(y_test, y_pred))

# ROC Curve
# Compute predicted probabilities
y_pred_proba = logreg.predict_proba(X_test_scaled)[::,1]
# Compute ROC curve
fpr, tpr, _ = metrics.roc_curve(y_test.astype(int),  y_pred_proba)
# Compute AUC
auc = metrics.roc_auc_score(y_test.astype(int), y_pred_proba)
# Plot ROC curve
plt.plot(fpr,tpr,label="AUC = "+str(round(auc, 3)))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc=4)
plt.show()

# Plotting Feature Importance
# Get feature names (if using PolynomialFeatures, get transformed feature names)
feature_names = X_train.columns  # Use X_train_poly.columns if using polynomial features

# Get absolute coefficients
feature_importance = np.abs(logreg.coef_[0])

# Sort features by importance
sorted_idx = np.argsort(feature_importance)[::-1]  # Sort in descending order

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_idx)), feature_importance[sorted_idx], align="center")
plt.xticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx], rotation=90)
plt.xlabel("Features")
plt.ylabel("Coefficient Magnitude")
plt.title("Feature Sensitivity in Logistic Regression")
plt.show()
###############################################################################
#                    PREDICTION FOR TEST DATASET
###############################################################################
# Predict for test data

# Convert all feature columns to numeric
X_submission = test_original.drop(columns=['id', 'day']).apply(pd.to_numeric, errors='coerce')

# Fill missing values
X_submission.fillna(X_submission['winddirection'].median(), inplace=True)

# Scale features
X_submission_scaled = scaler.fit_transform(X_submission)

y_test_pred = logreg.predict_proba(X_submission_scaled)[:,1] # Get probability of class 1

# Convert predictions to DataFrame and save
submit_df = pd.DataFrame({
    'id': test_original['id'],
    'rainfall': y_test_pred
})

submit_df.to_csv('Submission_S503_Punit_Bandi.csv', index=False)

print("Submission file saved as 'Submission_S503_Punit_Bandi.csv'")

