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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df_train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_train.head()


df_test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_test.head()


df_train.info()


df_train.isnull().sum()


df_train.describe()


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution of Numerical Features')

sns.histplot(df_train['Time_spent_Alone'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Time_spent_Alone')

sns.histplot(df_train['Social_event_attendance'], kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Social_event_attendance')

sns.histplot(df_train['Going_outside'], kde=True, ax=axes[0, 2])
axes[0, 2].set_title('Going_outside')

sns.histplot(df_train['Friends_circle_size'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Friends_circle_size')

sns.histplot(df_train['Post_frequency'], kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Post_frequency')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(df_train.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()


categorical_cols = df_train.select_dtypes(include='object').columns

for col in categorical_cols:
    print(f"Column: {col}")
    print(f"Number of unique values: {df_train[col].nunique()}")
    print("Value counts:")
    print(df_train[col].value_counts())
    print("-" * 30)


categorical_cols = df_train.select_dtypes(include='object').columns

for col in categorical_cols:
    plt.figure(figsize=(8,3))
    sns.countplot(data=df_train, x=col)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()


categorical_cols = df_train.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('Personality')

for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_train, x=col, hue='Personality')
    plt.title(f'Relationship between {col} and Personality')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title='Personality')
    plt.show()


numerical_cols = df_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_train.select_dtypes(include='object').columns.tolist()

# Exclude 'id' from numerical columns for plotting
if 'id' in numerical_cols:
    numerical_cols.remove('id')

# Exclude 'Personality' from categorical columns for plotting against other categoricals
if 'Personality' in categorical_cols:
    categorical_cols_for_plotting = categorical_cols.copy()
    categorical_cols_for_plotting.remove('Personality')
else:
    categorical_cols_for_plotting = categorical_cols.copy()


# Plot numerical features against 'Personality'
for num_col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_train, x='Personality', y=num_col)
    plt.title(f'{num_col} vs Personality')
    plt.xlabel('Personality')
    plt.ylabel(num_col)
    plt.show()

# Plot numerical features against other categorical features
for num_col in numerical_cols:
    for cat_col in categorical_cols_for_plotting:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_train, x=cat_col, y=num_col)
        plt.title(f'{num_col} vs {cat_col}')
        plt.xlabel(cat_col)
        plt.ylabel(num_col)
        plt.show()


df_test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_test.head()


# Identify columns with missing values
missing_train = df_train.isnull().sum()
missing_train = missing_train[missing_train > 0]
print("Missing values in df_train:")
print(missing_train)

missing_test = df_test.isnull().sum()
missing_test = missing_test[missing_test > 0]
print("\nMissing values in df_test:")
print(missing_test)

numerical_cols_with_missing = missing_train[df_train[missing_train.index].dtypes != 'object'].index.tolist()
for col in numerical_cols_with_missing:
    median_val = df_train[col].median()
    df_train[col].fillna(median_val, inplace=True)
    df_test[col].fillna(median_val, inplace=True)

# Categorical columns: 'Stage_fear', 'Drained_after_socializing'
# Mode imputation is a common strategy for categorical data.
categorical_cols_with_missing = missing_train[df_train[missing_train.index].dtypes == 'object'].index.tolist()
for col in categorical_cols_with_missing:
    mode_val = df_train[col].mode()[0]
    df_train[col].fillna(mode_val, inplace=True)
    df_test[col].fillna(mode_val, inplace=True) # Impute test set with train set's mode

# Verify that there are no remaining missing values
print("\nMissing values in df_train after imputation:")
print(df_train.isnull().sum().sum())

print("\nMissing values in df_test after imputation:")
print(df_test.isnull().sum().sum())


numerical_cols = df_train.select_dtypes(include=np.number).columns.tolist()
if 'id' in numerical_cols:
    numerical_cols.remove('id')

for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df_train[col])
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)
    plt.show()


categorical_cols = df_train.select_dtypes(include='object').columns

for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_train, x=col)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()


categorical_cols = df_train.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('Personality')

for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_train, x=col, hue='Personality')
    plt.title(f'Relationship between {col} and Personality')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title='Personality')
    plt.show()


numerical_cols = df_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_train.select_dtypes(include='object').columns.tolist()

# Exclude 'id' from numerical columns for plotting
if 'id' in numerical_cols:
    numerical_cols.remove('id')

# Exclude 'Personality' from categorical columns for plotting against other categoricals
if 'Personality' in categorical_cols:
    categorical_cols_for_plotting = categorical_cols.copy()
    categorical_cols_for_plotting.remove('Personality')
else:
    categorical_cols_for_plotting = categorical_cols.copy()


# Plot numerical features against 'Personality'
for num_col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_train, x='Personality', y=num_col)
    plt.title(f'{num_col} vs Personality')
    plt.xlabel('Personality')
    plt.ylabel(num_col)
    plt.show()

# Plot numerical features against other categorical features
for num_col in numerical_cols:
    for cat_col in categorical_cols_for_plotting:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_train, x=cat_col, y=num_col)
        plt.title(f'{num_col} vs {cat_col}')
        plt.xlabel(cat_col)
        plt.ylabel(num_col)
        plt.show()


# Identify columns with missing values
missing_train = df_train.isnull().sum()
missing_train = missing_train[missing_train > 0]
print("Missing values in df_train:")
print(missing_train)

missing_test = df_test.isnull().sum()
missing_test = missing_test[missing_test > 0]
print("\nMissing values in df_test:")
print(missing_test)

# Imputation Strategy:
# Numerical columns: 'Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency'
# Based on the histograms, most numerical features appear somewhat skewed or have a concentration of values.
# Median imputation is a robust strategy for skewed data or data with outliers.
numerical_cols_with_missing = missing_train[df_train[missing_train.index].dtypes != 'object'].index.tolist()
for col in numerical_cols_with_missing:
    median_val = df_train[col].median()
    df_train[col].fillna(median_val, inplace=True)
    df_test[col].fillna(median_val, inplace=True) # Impute test set with train set's median

# Categorical columns: 'Stage_fear', 'Drained_after_socializing'
# Mode imputation is a common strategy for categorical data.
categorical_cols_with_missing = missing_train[df_train[missing_train.index].dtypes == 'object'].index.tolist()
for col in categorical_cols_with_missing:
    mode_val = df_train[col].mode()[0]
    df_train[col].fillna(mode_val, inplace=True)
    df_test[col].fillna(mode_val, inplace=True) # Impute test set with train set's mode

# Verify that there are no remaining missing values
print("\nMissing values in df_train after imputation:")
print(df_train.isnull().sum().sum())

print("\nMissing values in df_test after imputation:")
print(df_test.isnull().sum().sum())


numerical_cols = df_train.select_dtypes(include=np.number).columns.tolist()
if 'id' in numerical_cols:
    numerical_cols.remove('id')

for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df_train[col])
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)
    plt.show()


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Separate features and target variable from the training data
X_train = df_train.drop(['id', 'Personality'], axis=1)
y_train = df_train['Personality']
X_test = df_test.drop('id', axis=1)

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include='object').columns
numerical_cols = X_train.select_dtypes(include=np.number).columns

# Create transformers for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Create a preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the training data
X_train_processed = preprocessing_pipeline.fit_transform(X_train)

# Transform the testing data
X_test_processed = preprocessing_pipeline.transform(X_test)

# Convert processed data back to DataFrames for easier handling (optional but good for inspection)
# Get feature names after one-hot encoding
feature_names = preprocessing_pipeline.named_steps['preprocessor'].get_feature_names_out()

X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

display(X_train_processed_df.head())
display(X_test_processed_df.head())


from sklearn.ensemble import RandomForestClassifier

# Instantiate the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train_processed_df, y_train)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_train_pred = model.predict(X_train_processed_df)

accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred, pos_label='Extrovert')
recall = recall_score(y_train, y_train_pred, pos_label='Extrovert')
f1 = f1_score(y_train, y_train_pred, pos_label='Extrovert')

print(f"Training Accuracy: {accuracy:.4f}")
print(f"Training Precision: {precision:.4f}")
print(f"Training Recall: {recall:.4f}")
print(f"Training F1-score: {f1:.4f}")


test_predictions = model.predict(X_test_processed_df)


# Create the submission DataFrame
submission_df = pd.DataFrame({'id': df_test['id'], 'Personality': test_predictions})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

display(submission_df.head())


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Separate features and target variable from the training data
X_train = df_train.drop(['id', 'Personality'], axis=1)
y_train = df_train['Personality']

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include='object').columns
numerical_cols = X_train.select_dtypes(include=np.number).columns

# Create transformers for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Create a preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the training data
X_train_processed = preprocessing_pipeline.fit_transform(X_train)

# Convert processed data back to DataFrames for easier handling (optional but good for inspection)
# Get feature names after one-hot encoding
feature_names = preprocessing_pipeline.named_steps['preprocessor'].get_feature_names_out()

X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)


# Instantiate the model
model = RandomForestClassifier(random_state=42)

# Define the scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, pos_label='Extrovert'),
    'recall': make_scorer(recall_score, pos_label='Extrovert'),
    'f1': make_scorer(f1_score, pos_label='Extrovert')
}

# Perform cross-validation
cv_results = cross_val_score(model, X_train_processed_df, y_train, cv=5, scoring='accuracy')

print(f"Cross-validation Accuracy: {cv_results.mean():.4f} (+/- {cv_results.std():.4f})")

# You can also perform cross-validation for other metrics similarly
cv_precision = cross_val_score(model, X_train_processed_df, y_train, cv=5, scoring=scoring['precision'])
cv_recall = cross_val_score(model, X_train_processed_df, y_train, cv=5, scoring=scoring['recall'])
cv_f1 = cross_val_score(model, X_train_processed_df, y_train, cv=5, scoring=scoring['f1'])

print(f"Cross-validation Precision: {cv_precision.mean():.4f} (+/- {cv_precision.std():.4f})")
print(f"Cross-validation Recall: {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")
print(f"Cross-validation F1-score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")


# Train the model on the entire training data (after cross-validation)
model.fit(X_train_processed_df, y_train)

# Separate features from the test data
X_test = df_test.drop('id', axis=1)

# Transform the test data using the same preprocessor fitted on the training data
X_test_processed = preprocessing_pipeline.transform(X_test)

# Convert processed test data back to DataFrame
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)


# Make predictions on the test set
test_predictions = model.predict(X_test_processed_df)

# Create the submission DataFrame
submission_df = pd.DataFrame({'id': df_test['id'], 'Personality': test_predictions})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

# Display the first few rows of the submission file
display(submission_df.head())




