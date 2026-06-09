# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

csv_files = []
for dirname, _, filenames in os.walk('/kaggle/input/stanford-rna-3d-folding'): 
    for filename in filenames:
        if filename.endswith('.csv'):
            csv_files.append(filename)

print(csv_files)



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

# Let's inspect the files provided by the user to understand their structure.
# Adjust the paths if needed for local or Kaggle environments.

# Attempt to read the test_sequences file to inspect its structure
try:
    test_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
except Exception as e:
    test_sequences = str(e)

# Attempt to read the sample_submission file to inspect its structure
try:
    sample_submission = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")
except Exception as e:
    sample_submission = str(e)

test_sequences, sample_submission


import pandas as pd

# Let's inspect the files provided by the user to understand their structure.
# Adjust the paths if needed for local or Kaggle environments.

# Attempt to read the test_sequences file to inspect its structure
try:
    test_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
except Exception as e:
    test_sequences = str(e)

# Attempt to read the sample_submission file to inspect its structure
try:
    sample_submission = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")
except Exception as e:
    sample_submission = str(e)

test_sequences, sample_submission


import pandas as pd

df_sample = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")
df_sample


df_test = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")  # or use `read_excel` for Excel files
df_test


print(df_sample.head())
print(df_test.head())


import pandas as pd

# Assuming df_test and df_sample are your two dataframes

# Extract base target ID from sample
df_sample['target_id'] = df_sample['ID'].str.extract(r'^(R\d+)', expand=False)

# Now join with test dataframe on 'target_id'
df_merged = df_sample.merge(df_test, on='target_id', how='left')


df_merged


# Sequence length
df_test['seq_length'] = df_test['sequence'].str.len()

# Number of residues per target in sample
residue_counts = df_sample['target_id'].value_counts().reset_index()
residue_counts.columns = ['target_id', 'residue_count']

# Merge back
df_test = df_test.merge(residue_counts, on='target_id', how='left')


# Example: Suppose df_sample already has encoded features
X = df_sample.drop(columns=["target_id"])  # all columns except target
X


y = df_sample["target_id"]  # Adjust this based on your actual target column name
y


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
)



# Check for non-numeric columns
non_numeric_cols = X.select_dtypes(include=['object']).columns
print(non_numeric_cols)


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for col in non_numeric_cols:
    X[col] = label_encoder.fit_transform(X[col])

# Verify that the encoding worked
print(X.head())


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# For example, using Logistic Regression again
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the model
classifier = LogisticRegression(max_iter=10000)
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))


# Assuming you have test data (X_test) and a trained model (classifier)
y_pred_test = classifier.predict(X_test)

# Check the length of predictions and ensure it matches the number of rows in the test set
print(len(y_pred_test))  # Should match the number of rows in the test data (2515)


print(df_test.columns)



import pandas as pd
import numpy as np

# Use this if you already have df_test loaded
# You'll need a list of all the test IDs that require predictions

# If you have a sample_submission.csv, read from there
sample_submission = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")

# Extract the 'ID' column from it
test_ids = sample_submission["ID"]

# Generate dummy predictions (shape must be [n_samples, 15])
# Replace this with your model predictions
y_pred = np.zeros((len(test_ids), 15))  # 5 atoms × (x, y, z) = 15 values per ID

# Create a DataFrame for submission
submission_df = pd.DataFrame(y_pred, columns=[
    'x_1', 'y_1', 'z_1',
    'x_2', 'y_2', 'z_2',
    'x_3', 'y_3', 'z_3',
    'x_4', 'y_4', 'z_4',
    'x_5', 'y_5', 'z_5'
])

# Add ID column
submission_df.insert(0, 'ID', test_ids)

# Save to CSV
submission_df.to_csv("sample_submission.csv", index=False)
print("✅ Submission file created successfully with shape:", submission_df.shape)



import pandas as pd

# Load your submission file
submission = pd.read_csv("sample_submission.csv")

# Show the first few rows
print(submission.head())



expected_columns = ['ID',
    'x_1', 'y_1', 'z_1',
    'x_2', 'y_2', 'z_2',
    'x_3', 'y_3', 'z_3',
    'x_4', 'y_4', 'z_4',
    'x_5', 'y_5', 'z_5'
]

# Compare to your submission file
missing = [col for col in expected_columns if col not in submission.columns]
extra = [col for col in submission.columns if col not in expected_columns]

print("Missing columns:", missing)
print("Extra columns:", extra)


# Check if there are any NaN values
print("Any NaN values?", submission.isnull().values.any())

# Show how many in each column
print(submission.isnull().sum())


print("Submission shape:", submission.shape)

