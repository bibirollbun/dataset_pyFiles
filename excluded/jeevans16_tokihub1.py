
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Path to the large train.csv file
file_path = "/kaggle/input/leash-BELKA/train.csv"

# --- Set up chunking to read the file ---
# We use a large chunk size to read the file in manageable pieces.
CHUNK_SIZE = 10_000_000

# Initialize a Series to store the counts for each 'binds' value
binds_counts = pd.Series(dtype=int)

print("Starting to count 'binds' values in the full dataset...")

# Read the large CSV file in chunks
with pd.read_csv(file_path, chunksize=CHUNK_SIZE) as reader:
    for i, chunk in enumerate(reader):
        # Calculate the value counts for the 'binds' column in the current chunk
        chunk_counts = chunk['binds'].value_counts()
        
        # Add the chunk counts to our running total
        binds_counts = binds_counts.add(chunk_counts, fill_value=0)
        
        print(f"Processed chunk {i+1}...")

print("\n--- Final Counts ---")
print("Total counts for each value in the 'binds' column:")
print(binds_counts.astype(int))




# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

import random

# Path to the large train.csv file
file_path = "/kaggle/input/leash-BELKA/train.csv"

# --- Set up parameters for creating a balanced subset ---
# We will collect an equal number of rows for each class.
rows_per_class = 20_000 
target_col = 'binds'

# Dictionaries to store collected samples
subset_samples = {
    0.0: [],
    1.0: []
}

print("Starting to build a balanced subset...")
# We will use a larger chunk size to get more samples per chunk.
CHUNK_SIZE = 1_000_000

# Read the large CSV file in chunks
with pd.read_csv(file_path, chunksize=CHUNK_SIZE) as reader:
    for i, chunk in enumerate(reader):
        print(f"Processing chunk {i+1}...")

        # Separate the chunk into its respective classes
        chunk_class_0 = chunk[chunk[target_col] == 0.0]
        chunk_class_1 = chunk[chunk[target_col] == 1.0]

        # Add rows to our subset lists, if we haven't reached the limit yet
        if len(subset_samples[0.0]) < rows_per_class:
            rows_to_add = min(len(chunk_class_0), rows_per_class - len(subset_samples[0.0]))
            subset_samples[0.0].extend(chunk_class_0.sample(rows_to_add, random_state=42).to_dict('records'))

        if len(subset_samples[1.0]) < rows_per_class:
            rows_to_add = min(len(chunk_class_1), rows_per_class - len(subset_samples[1.0]))
            subset_samples[1.0].extend(chunk_class_1.sample(rows_to_add, random_state=42).to_dict('records'))
        
        # Check if we have collected enough samples from both classes
        if (len(subset_samples[0.0]) >= rows_per_class and
            len(subset_samples[1.0]) >= rows_per_class):
            print("Enough samples collected from both classes. Stopping.")
            break

# Concatenate the collected samples into a single DataFrame
df_subset = pd.DataFrame(subset_samples[0.0] + subset_samples[1.0])

# Shuffle the final DataFrame to randomize the row order
df_subset = df_subset.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced subset created successfully!")
print(f"Final subset shape: {df_subset.shape}")
print("Displaying the distribution of 'binds' in the subset:")
print(df_subset['binds'].value_counts())
print("\nFirst 5 rows of the new subset:")
print(df_subset.head())



# EDA on the newly created subset
# Let's get the summary statistics and check data types of the subset
print("\n--- EDA on the Subset ---")

# Get a concise summary of the DataFrame, including data types and non-null counts
print("\nSubset Info:")
df_subset.info()

# Generate summary statistics for numerical features
print("\nSummary Statistics for Numerical Features:")
print(df_subset.describe())

# Generate summary statistics for object type features
print("\nSummary Statistics for Categorical Features:")
print(df_subset.describe(include=['object']))

# Assuming df_subset is your DataFrame from the previous step

# --- Visualize the class distribution of the 'binds' column ---
print("--- Plotting Class Distribution ---")
plt.figure(figsize=(8, 6))
sns.countplot(x='binds', data=df_subset, palette='viridis')
plt.title('Distribution of the Target Variable (binds)')
plt.xlabel('Binds Status (0: No, 1: Yes)')
plt.ylabel('Count')
plt.show()


# --- Visualize the distribution of the 'protein_name' feature ---
print("\n--- Plotting Protein Name Distribution ---")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Count of each unique protein
sns.countplot(x='protein_name', data=df_subset, ax=axes[0], palette='viridis')
axes[0].set_title('Count of Molecules per Protein')
axes[0].set_xlabel('Protein Name')
axes[0].set_ylabel('Number of Molecules')

# Binding rate for each protein
sns.barplot(x='protein_name', y='binds', data=df_subset, ax=axes[1], palette='viridis')
axes[1].set_title('Binding Rate per Protein')
axes[1].set_xlabel('Protein Name')
axes[1].set_ylabel('Average Binds Score')

plt.tight_layout()
plt.show()

# --- Analyze distribution of SMILES strings (top 5 most frequent) ---
# A count plot of all SMILES would be too large, so we'll look at the top 5
print("\n--- Plotting Top 5 Building Blocks ---")
top_5_bb1 = df_subset['buildingblock1_smiles'].value_counts().nlargest(5).index
top_5_bb2 = df_subset['buildingblock2_smiles'].value_counts().nlargest(5).index
top_5_bb3 = df_subset['buildingblock3_smiles'].value_counts().nlargest(5).index

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.countplot(x='buildingblock1_smiles', data=df_subset[df_subset['buildingblock1_smiles'].isin(top_5_bb1)], ax=axes[0], palette='viridis')
axes[0].set_title('Top 5 Most Frequent Building Block 1')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

sns.countplot(x='buildingblock2_smiles', data=df_subset[df_subset['buildingblock2_smiles'].isin(top_5_bb2)], ax=axes[1], palette='viridis')
axes[1].set_title('Top 5 Most Frequent Building Block 2')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

sns.countplot(x='buildingblock3_smiles', data=df_subset[df_subset['buildingblock3_smiles'].isin(top_5_bb3)], ax=axes[2], palette='viridis')
axes[2].set_title('Top 5 Most Frequent Building Block 3')
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()


# --- Check for Duplicates ---
print("--- Checking for Duplicates ---")
duplicate_molecules = df_subset.duplicated(subset=['molecule_smiles'], keep=False)
num_duplicates = duplicate_molecules.sum()

if num_duplicates > 0:
    print(f"Found {num_duplicates} rows with duplicate molecules.")
else:
    print("No duplicate molecules found.")

# --- Remove Duplicates ---
print("\n--- Removing Duplicates ---")
# Drop duplicates based on the 'molecule_smiles' column
# The 'keep' parameter set to 'first' will keep the first occurrence of each molecule.
df_no_duplicates = df_subset.drop_duplicates(subset=['molecule_smiles'], keep='first').copy()

print(f"Original shape: {df_subset.shape}")
print(f"Shape after removing duplicates: {df_no_duplicates.shape}")

# Let's verify the number of unique molecules now
print(f"Number of unique molecules in the cleaned data: {df_no_duplicates['molecule_smiles'].nunique()}")

# Update our working DataFrame for the next steps
df_cleaned = df_no_duplicates
print("\nDuplicates have been successfully removed.")



# Continue EDA on the cleaned subset
# Let's get the summary statistics and check data types of the subset
print("\n--- EDA on the cleaned set ---")

# Get a concise summary of the DataFrame, including data types and non-null counts
print("\nSubset Info:")
df_cleaned.info()

# Generate summary statistics for numerical features
print("\nSummary Statistics for Numerical Features:")
print(df_cleaned.describe())

# Generate summary statistics for object type features
print("\nSummary Statistics for Categorical Features:")
print(df_cleaned.describe(include=['object']))

# Assuming df_subset is your DataFrame from the previous step

# --- Visualize the class distribution of the 'binds' column ---
print("--- Plotting Class Distribution ---")
plt.figure(figsize=(8, 6))
sns.countplot(x='binds', data=df_cleaned, palette='viridis')
plt.title('Distribution of the Target Variable (binds)')
plt.xlabel('Binds Status (0: No, 1: Yes)')
plt.ylabel('Count')
plt.show()


# --- Visualize the distribution of the 'protein_name' feature ---
print("\n--- Plotting Protein Name Distribution ---")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Count of each unique protein
sns.countplot(x='protein_name', data=df_cleaned, ax=axes[0], palette='viridis')
axes[0].set_title('Count of Molecules per Protein')
axes[0].set_xlabel('Protein Name')
axes[0].set_ylabel('Number of Molecules')

# Binding rate for each protein
sns.barplot(x='protein_name', y='binds', data=df_cleaned, ax=axes[1], palette='viridis')
axes[1].set_title('Binding Rate per Protein')
axes[1].set_xlabel('Protein Name')
axes[1].set_ylabel('Average Binds Score')

plt.tight_layout()
plt.show()

# --- Analyze distribution of SMILES strings (top 5 most frequent) ---
# A count plot of all SMILES would be too large, so we'll look at the top 5
print("\n--- Plotting Top 5 Building Blocks ---")
top_5_bb1 = df_cleaned['buildingblock1_smiles'].value_counts().nlargest(5).index
top_5_bb2 = df_cleaned['buildingblock2_smiles'].value_counts().nlargest(5).index
top_5_bb3 = df_cleaned['buildingblock3_smiles'].value_counts().nlargest(5).index

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.countplot(x='buildingblock1_smiles', data=df_cleaned[df_cleaned['buildingblock1_smiles'].isin(top_5_bb1)], ax=axes[0], palette='viridis')
axes[0].set_title('Top 5 Most Frequent Building Block 1')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

sns.countplot(x='buildingblock2_smiles', data=df_cleaned[df_cleaned['buildingblock2_smiles'].isin(top_5_bb2)], ax=axes[1], palette='viridis')
axes[1].set_title('Top 5 Most Frequent Building Block 2')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

sns.countplot(x='buildingblock3_smiles', data=df_cleaned[df_cleaned['buildingblock3_smiles'].isin(top_5_bb3)], ax=axes[2], palette='viridis')
axes[2].set_title('Top 5 Most Frequent Building Block 3')
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()





# 'df_cleaned' is the DataFrame from the previous step
df_final = df_cleaned.copy()

# --- Preprocessing ---
print("--- Preprocessing ---")

# Step 1: Drop irrelevant columns that will not be used for modeling
# 'id' and 'molecule_smiles' are identifiers
df_final = df_final.drop(columns=['id', 'molecule_smiles'])

# Step 2: One-hot encode the categorical SMILES and protein columns
# pd.get_dummies() creates new binary columns for each unique value
# drop_first=True prevents multicollinearity by dropping one of the new columns
categorical_cols_to_encode = ['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'protein_name']
df_final = pd.get_dummies(df_final, columns=categorical_cols_to_encode, drop_first=True, dtype=int)

print("\nDataFrame after One-Hot Encoding:")
print(df_final.head())

print("\n" + "="*50 + "\n")

print("Final DataFrame Info:")
df_final.info()



# Import necessary libraries for modeling, evaluation, and plotting
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# --- Step 1: Define Features and Target ---
X = df_final.drop(columns=['binds'])
y = df_final['binds']

# --- Step 2: Split Data into Training and Validation Sets ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Step 3: Train the Logistic Regression Model ---
print("Training the Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model training complete.")

# --- Step 4: Make Predictions on Both Datasets ---
# Predictions for Training Set
y_pred_train = model.predict(X_train)
y_proba_train = model.predict_proba(X_train)[:, 1]

# Predictions for Validation Set
y_pred_val = model.predict(X_val)
y_proba_val = model.predict_proba(X_val)[:, 1]

# --- Step 5: Detailed Evaluation ---

# Evaluate on the TRAINING set first
print("\n" + "="*50)
print("--- Training Set Performance (Seen Data) ---")
print("="*50)
print(f"Training Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Training Average Precision: {average_precision_score(y_train, y_proba_train):.4f}")
print("\nTraining Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print("\nTraining Classification Report:\n", classification_report(y_train, y_pred_train))

# Evaluate on the VALIDATION set
print("\n" + "="*50)
print("--- Validation Set Performance (Unseen Data) ---")
print("="*50)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
print(f"Validation Average Precision: {average_precision_score(y_val, y_proba_val):.4f}")
print("\nValidation Confusion Matrix:\n", confusion_matrix(y_val, y_pred_val))
print("\nValidation Classification Report:\n", classification_report(y_val, y_pred_val))

# --- Step 6: Visualize Training vs. Validation Scores ---
metrics = ['Accuracy', 'Average Precision']
train_scores = [accuracy_score(y_train, y_pred_train), average_precision_score(y_train, y_proba_train)]
val_scores = [accuracy_score(y_val, y_pred_val), average_precision_score(y_val, y_proba_val)]

x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, train_scores, width, label='Training')
rects2 = ax.bar(x + width/2, val_scores, width, label='Validation')

# Add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Training vs. Validation Scores by Metric')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1)

# Function to attach a text label above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()




# Load the official test data
test_df_full = pd.read_csv("/kaggle/input/leash-BELKA/test.csv")

# --- Take a small random sample of 500 for demonstration ---
test_sample = test_df_full.sample(n=500, random_state=42).reset_index(drop=True)

print(f"Using a test sample of shape: {test_sample.shape}")

# --- Prepare the sample for prediction ---

# Keep a copy of the original IDs for the final report
test_ids = test_sample['id']

# STEP 1: Apply the same preprocessing transformations as on the training data
test_processed = test_sample.drop(columns=['id', 'molecule_smiles'])
test_processed = pd.get_dummies(test_processed, columns=categorical_cols_to_encode, drop_first=True, dtype=int)

# STEP 2: CRITICAL - Align columns to match the training data's structure
# This fixes any mismatches from the random sample by adding missing columns and filling with 0.
test_aligned = test_processed.reindex(columns=X.columns, fill_value=0)

# --- Make Predictions ---
# Use the trained model to predict the probability of binding (the positive class)
predictions = model.predict_proba(test_aligned)[:, 1]

# --- Create Final Results DataFrame ---
# Combine the IDs with their corresponding predictions
results = pd.DataFrame({
    'id': test_ids,
    'binds_probability': predictions
})

# Show the first 20 predictions from our sample
print("\nSample of predictions on the test set:")
results.head(20)


