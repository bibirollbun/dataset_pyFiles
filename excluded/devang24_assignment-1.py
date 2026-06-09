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

# Load the datasets from the correct path
train_df = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')  # Correct path to train.csv
test_df = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')    # Correct path to test.csv

# Display basic info and check for missing values
print(train_df.info())
print(train_df.head())
import numpy as np  # Ensure numpy is imported
import pandas as pd



# Check the column names to understand the structure of the data
print(train_df.columns)
print(test_df.columns)

# Check if the 'Age at enrollment' column exists and its data type
print(train_df['Age at enrollment'].dtype)
print(train_df['Age at enrollment'].head())



# Transform Target into binary (1 for 'Graduate', 0 for others)
train_df['Target'] = train_df['Target'].apply(lambda x: 1 if x == 'Graduate' else 0)

# Feature Engineering: Create Age_Category based on 'Age at enrollment'
train_df['Age_Category'] = pd.cut(train_df['Age at enrollment'], bins=[0, 18, 30, 45, np.inf], labels=['Teen', 'Young_Adult', 'Adult', 'Senior'])

# Same transformation for the test dataset
test_df['Age_Category'] = pd.cut(test_df['Age at enrollment'], bins=[0, 18, 30, 45, np.inf], labels=['Teen', 'Young_Adult', 'Adult', 'Senior'])



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define the numerical and categorical columns
numerical_cols = ['Previous qualification (grade)', 'Admission grade', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'Unemployment rate', 'Inflation rate', 'GDP']
categorical_cols = ['Marital status', 'Application mode', 'Course', 'Daytime/evening attendance', 'Previous qualification', 'Nacionality', 'Mother\'s qualification', 'Father\'s qualification', 'Mother\'s occupation', 'Father\'s occupation', 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (without evaluations)', 'Age_Category']

# Preprocessing pipelines for both numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both into a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)



from sklearn.ensemble import RandomForestClassifier

# Create the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])



from sklearn.model_selection import train_test_split

# Split the data into features (X) and target (y)
X_train = train_df.drop(['Target', 'id'], axis=1)
y_train = train_df['Target']

# Split the training data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)



from sklearn.metrics import accuracy_score

# Evaluate on the validation set
y_valid_pred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_valid_pred)
print("Validation Accuracy: ", accuracy)



# Predict on the test dataset
X_test = test_df.drop(['id'], axis=1)  # Ensure 'id' is not in features
predictions = model.predict(X_test)

# Create a submission dataframe
submission = pd.DataFrame({
    'id': test_df['id'],
    'Target': ['Graduate' if pred == 1 else 'Dropout' for pred in predictions]  # Map the binary prediction back to labels
})

# Save the submission as a CSV
submission.to_csv('submission.csv', index=False)





