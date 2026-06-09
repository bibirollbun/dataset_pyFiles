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


# Loading & naming datasets
train = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')

print("Training data shape:", train.shape)
print("Test data shape:", test.shape)
print("\nTraining columns:", train.columns.tolist())


import matplotlib.pyplot as plt
import seaborn as sns

# Target distribution
plt.figure(figsize=(8,4))
sns.countplot(x=train['Target'], palette='viridis')
plt.title('Target Variable Distribution', weight='bold')
plt.show()

# Numerical features analysis
num_cols = ['Admission grade', 'Previous qualification (grade)', 
            'Curricular units 1st sem (approved)', 'Age at enrollment']
train[num_cols].hist(bins=20, figsize=(12,8), layout=(2,2))
plt.tight_layout()
plt.show()

# Categorical features analysis
cat_cols = ['Gender', 'Scholarship holder', 'Displaced']
fig, axes = plt.subplots(1, 3, figsize=(18,4))
for i, col in enumerate(cat_cols):
    sns.countplot(x=col, data=train, ax=axes[i], palette='pastel')
plt.tight_layout()
plt.show()


# Function to create a count plot for categorical variables
def count_plot(df: pd.DataFrame, variable: str):
    # Set plot size and create count plot
    plt.figure(figsize=(4, 2))
    sns.countplot(data=df, x=variable, color="grey")

    # Customize plot labels and title
    plt.title(f'Count of {variable}')
    plt.xlabel(variable)
    plt.ylabel('Count')
    plt.show()
    print("\n-----\n")

# List of categorical features to visualize
features = ['Target', 'Tuition fees up to date', 'Displaced', 'Debtor']

# Generate plots for each categorical feature
for feature in features:
    count_plot(df=train, variable=feature)


def create_features(df):
    try:
        
# Feature 1: Course Load Efficiency
# - Measures how efficiently a student converts enrolled courses into approved courses.
# - Formula: (Approved Courses + 1) / (Enrolled Courses + 1)
# - The "+1" ensures we avoid division by zero and smooths the ratio for students with low enrollment.
        
        df['academic_efficiency'] = (
            (df['Curricular units 1st sem (approved)'] + 1) / 
            (df['Curricular units 1st sem (enrolled)'] + 1)
        )
        
# Feature 2: Academic Preparation Gap
# - Captures the difference between a student's admission grade and their previous qualification grade.
# - Formula: Admission Grade - Previous Qualification Grade
# - This helps identify students who may be underprepared or overprepared for their current program.
      
        df['grade_diff'] = (
            df['Admission grade'] - 
            df['Previous qualification (grade)'])

# Both features are designed to provide additional context about student performance.
        
        # Handle potential calculation issues
        df.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace infinities with NaN
        df.fillna(0, inplace=True) # Fill NaN with 0
        
        print("Successfully created features")
        
    except KeyError as e:
        print(f"Missing critical column: {e}")
        raise
        
    return df

# Apply feature engineering with verification
print("\nApplying feature engineering to training data:")
train = create_features(train)
print("\nApplying feature engineering to test data:")
test = create_features(test)

# Verify new features exist
print("\nUpdated training columns:", train.columns.tolist())


numeric_features = ['Admission grade', 'Previous qualification (grade)', 
                    'academic_efficiency', 'grade_diff']
categorical_features = ['Gender', 'Scholarship holder', 'Displaced']

# Checking if features exist before proceeding
missing_features = [f for f in numeric_features + categorical_features 
                   if f not in train.columns]
if missing_features:
    raise ValueError(f"Missing critical features: {missing_features}")


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Define feature groups
numeric_features = ['Admission grade', 'Previous qualification (grade)', 
                    'academic_efficiency', 'grade_diff']
categorical_features = ['Gender', 'Scholarship holder', 'Displaced']

# Preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
        # 1- Impute missing values with the median
        # 2- Scale features to have zero mean and unit variance
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight='balanced'
    ))
])


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Prepare data
X = train[numeric_features + categorical_features]
y = train['Target'].map({'Dropout':0, 'Enrolled':1, 'Graduate':2})

# Training & Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
pipeline.fit(X_train, y_train)


preprocessor


val_preds = pipeline.predict(X_val)
print(classification_report(y_val, val_preds, target_names=['Dropout', 'Enrolled', 'Graduate']))


test_preds = pipeline.predict(test[numeric_features + categorical_features])

# Convert numerical predictions to string labels
label_mapping = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
submission_labels = [label_mapping[pred] for pred in test_preds]  # List comprehension


# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Target': submission_labels  # Use converted labels
})

# Verifying conversion
print("Prediction distribution:")
print(submission['Target'].value_counts())

# Saving final output
submission.to_csv('submission.csv', index=False)
print("\nSubmission sample:")
print(submission.head())

