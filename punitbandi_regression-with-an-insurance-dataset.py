# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 00:41:27 2024

Kaggle Playground Competition

@author: Punit Bandi
"""
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool, cv

# Plot visualizations for trends in train dataset
# Plot_Visual = 0: No plotting
# Plot_Visual = 1: Yes
Plot_Visuals = 1

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

columns_to_convert = ['id','Age','Annual Income','Number of Dependents','Health Score',
               'Previous Claims','Vehicle Age','Credit Score','Insurance Duration',
               'Premium Amount']

columns_to_convert_test = ['id','Age','Annual Income','Number of Dependents','Health Score',
               'Previous Claims','Vehicle Age','Credit Score','Insurance Duration']

# Handle Policy Start Date Input Variable
# Convert to datetime
train_original['Policy Start Date'] = pd.to_datetime(train_original['Policy Start Date'])
test_original['Policy Start Date'] = pd.to_datetime(test_original['Policy Start Date'])

# Extract features from date time entries
# The time on each entry seems to be same so only extracting year, month and day
train_original['PS_year'] = train_original['Policy Start Date'].dt.year
train_original['PS_month'] = train_original['Policy Start Date'].dt.month
train_original['PS_day'] = train_original['Policy Start Date'].dt.day

test_original['PS_year'] = test_original['Policy Start Date'].dt.year
test_original['PS_month'] = test_original['Policy Start Date'].dt.month
test_original['PS_day'] = test_original['Policy Start Date'].dt.day

# Drop 'Policy Start Date' column from dataframe
train_original = train_original.drop(columns=['Policy Start Date'])
test_original = test_original.drop(columns=['Policy Start Date'])

###############################################################################
#                 VISUALIZE TRENDS IN TRAINING DATASET
###############################################################################
if Plot_Visuals==1:
    # Violin plot showing distribution and density of "Premium Amount" for each "Gender"
    sns.violinplot(x='Gender', y='Premium Amount', data=train_original)
    plt.yscale('log')
    plt.title("Violin Plot of Gender vs. Premium Amount")
    plt.ylabel("Premium Amount (Log Scale)")
    plt.show()
    # Violin plot showing distribution and density of "Premium Amount" for each "Marital Status"
    sns.violinplot(x='Marital Status', y='Premium Amount', data=train_original)
    plt.yscale('log')
    plt.title("Violin Plot of Marital Status vs. Premium Amount")
    plt.ylabel("Premium Amount (Log Scale)")
    plt.show()
    # Violin plot showing distribution and density of "Premium Amount" for each "Education Level"
    sns.violinplot(x='Education Level', y='Premium Amount', data=train_original)
    plt.yscale('log')
    plt.title("Violin Plot of Education Level vs. Premium Amount")
    plt.ylabel("Premium Amount (Log Scale)")
    plt.show()
    # Violin plot showing distribution and density of "Premium Amount" for each "Occupation"
    sns.violinplot(x='Occupation', y='Premium Amount', data=train_original)
    plt.yscale('log')
    plt.title("Violin Plot of Occupation vs. Premium Amount")
    plt.ylabel("Premium Amount (Log Scale)")
    plt.show()
    # Violin plot showing distribution and density of "Premium Amount" for each "Location"
    sns.violinplot(x='Location', y='Premium Amount', data=train_original)
    plt.yscale('log')
    plt.title("Violin Plot of Location vs. Premium Amount")
    plt.ylabel("Premium Amount (Log Scale)")
    plt.show()
    # Violin plot showing distribution and density of "Premium Amount" for each "Policy Type"
    sns.violinplot(x='Policy Type', y='Premium Amount', data=train_original)
    plt.yscale('log')
    plt.title("Violin Plot of Policy Type vs. Premium Amount")
    plt.ylabel("Premium Amount (Log Scale)")
    plt.show()
    # Violin plot showing distribution and density of "Premium Amount" for each "Customer Feedback"
    sns.violinplot(x='Customer Feedback', y='Premium Amount', data=train_original)
    plt.yscale('log')
    plt.title("Violin Plot of Customer Feedback vs. Premium Amount")
    plt.ylabel("Premium Amount (Log Scale)")
    plt.show()
    # Violin plot showing distribution and density of "Premium Amount" for each "Smoking Status"
    sns.violinplot(x='Smoking Status', y='Premium Amount', data=train_original)
    plt.yscale('log')
    plt.title("Violin Plot of Smoking Status vs. Premium Amount")
    plt.ylabel("Premium Amount (Log Scale)")
    plt.show()
    # Violin plot showing distribution and density of "Premium Amount" for each "Exercise Frequency"
    sns.violinplot(x='Exercise Frequency', y='Premium Amount', data=train_original)
    plt.yscale('log')
    plt.title("Violin Plot of Exercise Frequency vs. Premium Amount")
    plt.ylabel("Premium Amount (Log Scale)")
    plt.show()
    # Violin plot showing distribution and density of "Premium Amount" for each "Propert Type"
    sns.violinplot(x='Property Type', y='Premium Amount', data=train_original)
    plt.yscale('log')
    plt.title("Violin Plot of Property Type vs. Premium Amount")
    plt.ylabel("Premium Amount (Log Scale)")
    plt.show()
    # Hexbin plot of Age vs. Premium Amount
    plt.hexbin(
    x=pd.to_numeric(train_original['Age'], errors='coerce'),
    y=pd.to_numeric(train_original['Premium Amount'], errors='coerce'),
    cmap="Blues", gridsize=30
    )
    plt.title('Hexbin Plot of Age vs Premium Amount')
    plt.xlabel('Age')
    plt.ylabel('Premium Amount')
    plt.show()
    # Hexbin plot of Annual Income vs. Premium Amount
    plt.hexbin(
    x=pd.to_numeric(train_original['Annual Income'], errors='coerce'),
    y=pd.to_numeric(train_original['Premium Amount'], errors='coerce'),
    cmap="Blues", gridsize=30
    )
    plt.title('Hexbin Plot of Annual vs Premium Amount')
    plt.xlabel('Annual Income')
    plt.ylabel('Premium Amount')
    plt.show()
    # Hexbin plot of Previous Claims vs. Premium Amount
    plt.hexbin(
        x=pd.to_numeric(train_original['Previous Claims'], errors='coerce'),
        y=pd.to_numeric(train_original['Premium Amount'], errors='coerce'),
        cmap="Blues", gridsize=30
        )
    plt.title('Hexbin Plot of Previous Claims vs Premium Amount')
    plt.xlabel('Previous Claims')
    plt.ylabel('Premium Amount')
    plt.show()
    # Hexbin plot of Credit Score vs. Premium Amount
    plt.hexbin(
        x=pd.to_numeric(train_original['Credit Score'], errors='coerce'),
        y=pd.to_numeric(train_original['Premium Amount'], errors='coerce'),
        cmap="Blues", gridsize=30
        )
    plt.title('Hexbin Plot of Credit Score vs Premium Amount')
    plt.xlabel('Credit Score')
    plt.ylabel('Premium Amount')
    plt.show()
    # Histogram for Premium Amount
    plt.hist(train_original['Premium Amount'].sort_values(), bins=20, edgecolor='black')  # `bins` controls the number of bins
    plt.title('Histogram of Premium Amount')
    plt.xlabel('Premium Value')
    plt.ylabel('Frequency')
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Maximum 8 ticks on the x-axis
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))  # Round to nearest integer
    # Rotate labels for better visibility
    plt.xticks(rotation=90)
    plt.show()

###############################################################################
#       IDENTIFY NULL & MISSING DATA IN TRAINING & TEST DATASET
###############################################################################
# Identify rows with one or more empty strings
rows_with_empty_strings = (train_original == "").any(axis=1)
rows_with_empty_strings_test = (test_original == "").any(axis=1)

# Count rows with empty strings
count_rows_with_empty_strings = rows_with_empty_strings.sum()
print(f"Number of rows with missing data in train set: {count_rows_with_empty_strings}")

count_rows_with_empty_strings_test = rows_with_empty_strings_test.sum()
print(f"Number of rows with missing data in test set: {count_rows_with_empty_strings_test}")

empty_string_count = (train_original == '').sum()
empty_string_percentage = (empty_string_count / len(train_original)) * 100

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
#                 PREPARE DATA FOR TRAINING AND TESTING
###############################################################################
categorical_cols = ['Gender', 'Marital Status', 'Education Level', 'Occupation',
                    'Location', 'Policy Type', 'Customer Feedback', 'Smoking Status', 
                    'Exercise Frequency', 'Property Type']
numerical_cols = ['id','Age','Annual Income','Number of Dependents','Health Score',
                  'Previous Claims','Vehicle Age','Credit Score','Insurance Duration',
                  'PS_year','PS_month','PS_day','Premium Amount']

columns_to_exclude = ['id','Premium Amount']
X = train_original.drop(columns=columns_to_exclude)
y = train_original['Premium Amount']
y = pd.to_numeric(y)
y_transformed = np.log1p(y)  # log(1 + y) to avoid log(0) for y = 0
y_transformed = y_transformed.astype(str)
X_test = test_original.drop(columns='id')

##############################################################################
#                            CAT BOOSTING REGRESSION
##############################################################################
# Define CatBoost Pool
train_pool = Pool(
    data=X,
    label=y_transformed,
    cat_features=categorical_cols
)

# Cross-validation
cv_results = cv(
    params={
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg':3,
        'loss_function': 'RMSE',
        'verbose': 1,
    },
    pool=train_pool,
    fold_count=5,
    shuffle=True,
    partition_random_seed=42,
    verbose=True
)

print("Cross-Validation Results (Last Iteration):")
print(cv_results.tail(1))

# Train the model on the entire dataset
model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=8, l2_leaf_reg=3, verbose=1)
model.fit(train_pool)

# Feature importance
feature_importances = model.get_feature_importance(train_pool)
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Sensitivity (Importance) Plot")
plt.gca().invert_yaxis()
plt.show()

##############################################################################
#                    PREDICTION FOR TEST DATASET
##############################################################################
X_pred = test_original.drop(columns='id')

# Predict Premium Amount for test data
y_pred_transformed_Cat = model.predict(X_pred)
y_pred_Cat = np.expm1(y_pred_transformed_Cat) # Inverse of log1p
# Convert to DataFrame and assign a column name
y_pred = pd.DataFrame(y_pred_Cat, columns=['Premium Amount'])
    
# Write predicted values in the sample submission format
# Combine columns into a new DataFrame
submit_df = pd.concat([test_original['id'], y_pred['Premium Amount']], axis=1)
submit_df['Premium Amount'] = submit_df['Premium Amount'].round(3)

# Write to CSV
submit_df.to_csv('Submission_Punit_Bandi.csv', index=False)

