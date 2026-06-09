import pandas as pd
import os

# Define the base path to the directory in Google Drive where the files are stored
base_path = '/kaggle/input/playground-series-s5e6'

# Load the training data
train_file_path = os.path.join(base_path, 'train.csv')
train_df = pd.read_csv(train_file_path)
display(train_df)

# Load the test data
test_file_path = os.path.join(base_path, 'test.csv')
test_df = pd.read_csv(test_file_path)
display(test_df)

# Load the sample submission data
sample_submission_file_path = os.path.join(base_path, 'sample_submission.csv')
sample_submission_df = pd.read_csv(sample_submission_file_path)
display(sample_submission_df)


# Understand the dataset dimensions, column names, and basic information for train_df
print("--- Training Data Information ---")
print("Shape:", train_df.shape)
print("\nColumns:", train_df.columns.tolist())
print("\nInfo:")
train_df.info()

print("\n--- Test Data Information ---")
print("Shape:", test_df.shape)
print("\nColumns:", test_df.columns.tolist())
print("\nInfo:")
test_df.info()

# Identify and categorize features as numerical or categorical
numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

print("\n--- Feature Categorization ---")
print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)


# Calculate the number of missing values for train_df
train_missing_count = train_df.isnull().sum()

# Calculate the percentage of missing values for train_df
train_missing_percentage = (train_missing_count / len(train_df)) * 100

# Print the missing values for train_df
print("--- Missing Values in Training Data ---")
print("Number of missing values per column:")
print(train_missing_count)
print("\nPercentage of missing values per column:")
print(train_missing_percentage)

# Calculate the number of missing values for test_df
test_missing_count = test_df.isnull().sum()

# Calculate the percentage of missing values for test_df
test_missing_percentage = (test_missing_count / len(test_df)) * 100

# Print the missing values for test_df
print("\n--- Missing Values in Test Data ---")
print("Number of missing values per column:")
print(test_missing_count)
print("\nPercentage of missing values per column:")
print(test_missing_percentage)


# Calculate the number of duplicate rows in train_df
train_duplicate_rows = train_df.duplicated().sum()
print(f"Number of duplicate rows in training data: {train_duplicate_rows}")

# Calculate the number of duplicate rows in test_df
test_duplicate_rows = test_df.duplicated().sum()
print(f"Number of duplicate rows in test data: {test_duplicate_rows}")


# Descriptive statistics for numerical features in train_df
print("--- Descriptive Statistics for Numerical Features (Training Data) ---")
display(train_df[numerical_features].describe())

# Unique values and counts for categorical features in train_df
print("\n--- Unique Values and Counts for Categorical Features (Training Data) ---")
for col in categorical_features:
    print(f"\nColumn: {col}")
    display(train_df[col].value_counts())

# Descriptive statistics for numerical features in test_df
print("\n--- Descriptive Statistics for Numerical Features (Test Data) ---")
# Exclude 'id' as it's just an identifier
display(test_df[[col for col in numerical_features if col != 'id']].describe())


# Unique values and counts for categorical features in test_df
print("\n--- Unique Values and Counts for Categorical Features (Test Data) ---")
# Test data does not have 'Fertilizer Name'
test_categorical_features = [col for col in categorical_features if col != 'Fertilizer Name']
for col in test_categorical_features:
    print(f"\nColumn: {col}")
    display(test_df[col].value_counts())


# 4. Target variables analysis

# Class Distribution of 'Fertilizer Name'
print("--- Class Distribution of 'Fertilizer Name' (Training Data) ---")
display(train_df['Fertilizer Name'].value_counts())
print("\nPercentage of each Fertilizer Name:")
display(train_df['Fertilizer Name'].value_counts(normalize=True) * 100)

# Class Relationships: Examine relationships between 'Fertilizer Name' and input features

# For numerical features, we can look at the mean of each numerical feature for each fertilizer type
print("\n--- Mean of Numerical Features by Fertilizer Name ---")
display(train_df.groupby('Fertilizer Name')[numerical_features].mean())

# For categorical features, we can look at the distribution of soil types and crop types for each fertilizer type
print("\n--- Distribution of Categorical Features by Fertilizer Name ---")
for col in ['Soil Type', 'Crop Type']:
    print(f"\nDistribution of {col} by Fertilizer Name:")
    display(pd.crosstab(train_df['Fertilizer Name'], train_df[col]))


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.countplot(data=train_df, x='Fertilizer Name', palette='viridis')
plt.title('Distribution of Fertilizer Types')
plt.xlabel('Fertilizer Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

numerical_features_to_plot = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

for feature in numerical_features_to_plot:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=train_df, x='Fertilizer Name', y=feature, palette='viridis')
    plt.title(f'Distribution of {feature} by Fertilizer Name')
    plt.xlabel('Fertilizer Name')
    plt.ylabel(feature)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# Create stacked bar plots for categorical features vs. Fertilizer Name

categorical_features_to_plot = ['Soil Type', 'Crop Type']

for col in categorical_features_to_plot:
    # Create a cross-tabulation
    cross_tab = pd.crosstab(train_df['Fertilizer Name'], train_df[col])

    # Normalize by index to get percentages
    cross_tab_normalized = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

    # Create stacked bar plot
    cross_tab_normalized.plot(kind='bar', stacked=True, figsize=(12, 7))
    plt.title(f'Distribution of {col} by Fertilizer Name')
    plt.xlabel('Fertilizer Name')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# Correlation Analysis for numerical features
print("--- Correlation Matrix for Numerical Features (Training Data) ---")
correlation_matrix = train_df[numerical_features].drop('id', axis=1).corr()
display(correlation_matrix)

# Optional: Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# Categorical-Numerical Relationships (using box plots as requested previously)
# We have already generated box plots for numerical features vs. Fertilizer Name
# Here we can focus on other categorical features like Soil Type and Crop Type

categorical_features_for_bivariate_plot = ['Soil Type', 'Crop Type']
numerical_features_for_bivariate_plot = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

for cat_feature in categorical_features_for_bivariate_plot:
    for num_feature in numerical_features_for_bivariate_plot:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=train_df, x=cat_feature, y=num_feature, palette='viridis')
        plt.title(f'Distribution of {num_feature} by {cat_feature}')
        plt.xlabel(cat_feature)
        plt.ylabel(num_feature)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


# Scatter Plot Analysis for pairs of numerical features
# Due to the large number of numerical features, we will select a few key pairs
# to avoid generating too many plots. We can also use a pair plot for a subset.

# Example pairs based on potential relationships or interesting correlations from the matrix
numerical_pairs_to_plot = [
    ('Temparature', 'Humidity'),
    ('Nitrogen', 'Phosphorous'),
    ('Potassium', 'Phosphorous'),
    ('Nitrogen', 'Potassium'),
    ('Temparature', 'Moisture')
]

for x_feature, y_feature in numerical_pairs_to_plot:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=train_df, x=x_feature, y=y_feature, alpha=0.5, s=5) # Reduced point size for large dataset
    plt.title(f'Scatter Plot of {x_feature} vs. {y_feature}')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.tight_layout()
    plt.show()

# Optional: Pair plot for a subset of numerical features (can be slow for many features)
# sns.pairplot(train_df[numerical_features_for_bivariate_plot])
# plt.suptitle('Pair Plot of Numerical Features', y=1.02)
# plt.show()


from sklearn.preprocessing import OneHotEncoder

# Identify categorical columns for encoding (excluding 'id' and the target 'Fertilizer Name')
categorical_cols = ['Soil Type', 'Crop Type']

# Initialize OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit the encoder on the training data's categorical columns and transform training data
train_encoded_features = encoder.fit_transform(train_df[categorical_cols])

# Transform the test data's categorical columns
test_encoded_features = encoder.transform(test_df[categorical_cols])

# Create DataFrames from the encoded features
train_encoded_df = pd.DataFrame(train_encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
test_encoded_df = pd.DataFrame(test_encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

# Drop original categorical columns and 'id' from training and test data
train_numerical_df = train_df.drop(columns=categorical_cols + ['id', 'Fertilizer Name'])
test_numerical_df = test_df.drop(columns=categorical_cols + ['id'])

# Concatenate numerical and encoded features
X_train_processed = pd.concat([train_numerical_df, train_encoded_df], axis=1)
X_test_processed = pd.concat([test_numerical_df, test_encoded_df], axis=1)

# Define the target variable
y_train = train_df['Fertilizer Name']

print("Processed Training Features (X_train_processed) head:")
display(X_train_processed.head())
print("\nTarget Variable (y_train) head:")
display(y_train.head())
print("\nProcessed Test Features (X_test_processed) head:")
display(X_test_processed.head())


from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_processed, y_train, test_size=0.2, random_state=42)

# Print the shapes of the resulting sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_val:", y_val.shape)


from sklearn.ensemble import RandomForestClassifier

# Instantiate the Random Forest Classifier
# Starting with 100 estimators and a random_state for reproducibility
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)

print("Random Forest model trained successfully.")


# Make predictions on the validation set
y_pred = rf_model.predict(X_val)

print("Predictions on the validation set have been made.")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate Accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Calculate Precision, Recall, and F1-score (using 'weighted' average)
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')

print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-score (weighted): {f1:.4f}")

# Generate Confusion Matrix
cm = confusion_matrix(y_val, y_pred)

# Display Confusion Matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Fertilizer Name')
plt.ylabel('True Fertilizer Name')
plt.show()


import numpy as np

# Predict probabilities for each class on the test data
test_probabilities = rf_model.predict_proba(X_test_processed)

# Get the class names from the trained model
class_names = rf_model.classes_

# Get the indices of the top 3 predicted classes for each test sample
# We use np.argsort and then slice the last 3 columns, and reverse to get in descending order of probability
top_3_indices = np.argsort(test_probabilities, axis=1)[:, -3:][:, ::-1]

# Get the corresponding top 3 class names for each test sample
top_3_fertilizers = []
for indices in top_3_indices:
    top_3_fertilizers.append([class_names[i] for i in indices])

# Create the submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id'], 'Fertilizer Name': [' '.join(fertilizers) for fertilizers in top_3_fertilizers]})

print("Submission DataFrame head:")
display(submission_df.head())


# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")

