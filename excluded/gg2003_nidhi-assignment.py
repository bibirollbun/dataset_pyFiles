import os


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set the correct file path
data_path = '/kaggle/input/leash-BELKA/train.csv'

# Load a subset of the data for faster processing
df_train = pd.read_csv(data_path, nrows=100000)

test_path = '/kaggle/input/leash-BELKA/test.csv'
df_test = pd.read_csv(test_path)

print("Dataset loaded successfully!")

# Data Inspection
print("\nFirst 5 rows of the dataframe:")
print(df_train.head())

print("\nSummary information:")
df_train.info()

print("\nDescriptive statistics:")
print(df_train.describe())

# Target Variable Analysis
print("\nDistribution of the target variable `binds`:")
print(df_train['binds'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='binds', data=df_train)
plt.title('Distribution of the Target Variable (binds)')
plt.xlabel('Binds (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Feature Analysis
print("\nMissing values in each column:")
print(df_train.isnull().sum())

num_unique_molecules = df_train['molecule_smiles'].nunique()
print(f"\nNumber of unique molecules in the subset: {num_unique_molecules}")


# Calculate and print the percentage of each class to highlight imbalance
total_rows = len(df_train)
binds_counts = df_train['binds'].value_counts()
percentage_binds_1 = (binds_counts.get(1, 0) / total_rows) * 100
percentage_binds_0 = (binds_counts.get(0, 0) / total_rows) * 100

print(f"\nPercentage of 'binds=1' (positive class): {percentage_binds_1:.2f}%")
print(f"Percentage of 'binds=0' (negative class): {percentage_binds_0:.2f}%")



# Analyze the 'protein_name' feature
print("\nUnique protein names and their counts:")
protein_counts = df_train['protein_name'].value_counts()
print(protein_counts)

# Visualize the distribution of protein names
plt.figure(figsize=(10, 6))
sns.countplot(y='protein_name', data=df_train, order=protein_counts.index)
plt.title('Distribution of Protein Names')
plt.xlabel('Count')
plt.ylabel('Protein Name')
plt.tight_layout()
plt.show()



# Create new features for the length of SMILES strings
df_train['molecule_smiles_length'] = df_train['molecule_smiles'].apply(len)
df_train['buildingblock1_smiles_length'] = df_train['buildingblock1_smiles'].apply(len)

# Analyze the distribution of these new features
print("\nDescriptive statistics for molecule and building block SMILES lengths:")
print(df_train[['molecule_smiles_length', 'buildingblock1_smiles_length']].describe())

# Visualize the distribution of molecule SMILES length
plt.figure(figsize=(10, 5))
sns.histplot(df_train['molecule_smiles_length'], bins=50, kde=True)
plt.title('Distribution of Molecule SMILES Lengths')
plt.xlabel('Molecule SMILES Length')
plt.ylabel('Frequency')
plt.show()


# Compare molecule length for positive and negative binding events
plt.figure(figsize=(10, 6))
sns.boxplot(x='binds', y='molecule_smiles_length', data=df_train)
plt.title('Molecule SMILES Length vs. Binding Status')
plt.xlabel('Binds (0 = No, 1 = Yes)')
plt.ylabel('Molecule SMILES Length')
plt.show()



!pip install rdkit-pypi
!pip install pandarallel


# Task 2: Data Preprocessing

import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Initialize pandarallel
pandarallel.initialize(progress_bar=True, verbose=0)

# Take a small sample
df_train_subset = df_train.sample(n=30000, random_state=42)
df_test_subset = pd.read_csv('/kaggle/input/leash-BELKA/test.csv', nrows=10000)

# Fingerprint function
def smiles_to_fingerprint(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is not None:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    else:
        return np.zeros(1024, dtype=int)

# Generate fingerprints
train_fp_series = df_train_subset['molecule_smiles'].parallel_apply(smiles_to_fingerprint)
test_fp_series = df_test_subset['molecule_smiles'].parallel_apply(smiles_to_fingerprint)

# Convert to DataFrames, now with explicit STRING column names
fp_cols = [f'fp_{i}' for i in range(1024)] # Create string names like 'fp_0', 'fp_1', etc.
X_train_fp = pd.DataFrame(train_fp_series.to_list(), columns=fp_cols).reset_index(drop=True)
X_test_fp = pd.DataFrame(test_fp_series.to_list(), columns=fp_cols).reset_index(drop=True)

# Reset the index on the one-hot encoded data
X_train_protein = pd.get_dummies(df_train_subset['protein_name'], prefix='protein').reset_index(drop=True)
X_test_protein = pd.get_dummies(df_test_subset['protein_name'], prefix='protein').reset_index(drop=True)

# Concatenation 
X_train = pd.concat([X_train_protein, X_train_fp], axis=1)
X_test = pd.concat([X_test_protein, X_test_fp], axis=1)

# The target 'y' must also be reset to match
y = df_train_subset['binds'].reset_index(drop=True)

# Align columns and finish
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
df_test = df_test_subset

print("\n Data preprocessing complete!")
print(f"All {len(X_train.columns)} column names are now strings.")


# Task 3 & 4: Model, Train, and Evaluate

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 1. Split the data into a training set and a validation set
# We use the X_train and y data we created during preprocessing
# The model will train on the 'tr' set and we'll check it on the 'val' set
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42, stratify=y)

# 2. Initialize the Logistic Regression model
# class_weight='balanced' helps the model handle the highly imbalanced data
model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42, max_iter=1000)

# 3. Train the model on the training portion of the data
print("Training the model...")
model.fit(X_tr, y_tr)
print("Model training complete.")

# 4. Evaluate the model on the unseen validation set
print("\n Model Evaluation on Validation Set ")
# Predict probabilities for the positive class (binds=1)
y_pred_proba = model.predict_proba(X_val)[:, 1]

# To generate a classification report, we need class labels (0 or 1)
y_pred_class = model.predict(X_val)

# Print the performance metrics
roc_auc = roc_auc_score(y_val, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_class))



print("Making predictions on the official test data...")

# Use the trained model to predict the binding probabilities on the test set
# We use .predict_proba() because the competition metric (average precision) uses probabilities.
test_predictions = model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully.")

# Create the final submission file in the required format
submission_df = pd.DataFrame({'id': df_test['id'], 'binds': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully.")
print("First 5 rows of the submission file:")
print(submission_df.head())


