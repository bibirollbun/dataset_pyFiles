import os
import pandas as pd

# Get the path to the input directory
# Kaggle automatically provides this when your notebook is run in a competition context
INPUT_DIR = '/kaggle/input'

# List all directories in the input path
all_dirs = os.listdir(INPUT_DIR)

# Assuming your competition data is in one of these directories, let's pick the first one.
# You might need to adjust this if your data is structured differently.
competition_dir = os.path.join(INPUT_DIR, all_dirs[0])

# Now you can list the files in the competition directory
all_files = os.listdir(competition_dir)

print("Files in competition directory:", all_files)

# --- Example: Reading a CSV file ---
# Let's say you know there's a CSV file named 'train.csv'
try:
    train_path = os.path.join(competition_dir, 'TRAIN_CATEGORICAL_METADATA_new.xlsx')
    df_train = pd.read_excel(train_path)
    print("Shape of train data:", df_train.shape)
except FileNotFoundError:
    print("Error: 'TRAIN_CATEGORICAL_METADATA_new.xlsx' not found. Check the file name and path.")
except Exception as e:
    print("An error occurred while reading the file:", e)


# --- Example:  Accessing other files ---
# You can adapt the above pattern to read other files as needed
# For instance, if you have a 'test.csv', replace 'train.csv'


# **Important Notes:**

# 1.  **Directory Structure:** Kaggle competitions have a specific directory structure.  Usually, the data is within a subdirectory of `/kaggle/input`.  The code above helps you find that subdirectory.  If you know the exact directory name, you can hardcode it, but it's safer to be flexible.
# 2.  **File Names:** You'll need to know the names of the files you want to read (e.g., `train.csv`, `test.csv`).  You can get these names by looking at the "Data" tab of the Kaggle competition.
# 3.  **File Types:** Use the appropriate pandas function to read the file type (e.g., `pd.read_csv()`, `pd.read_excel()`, `pd.read_parquet()`).
# 4.  **Error Handling:** It's good practice to use `try...except` blocks to handle potential errors, like files not being found.
# 5.  **Adapt:** Modify the code to fit the specific file names and structure of your Kaggle competition's dataset.


import numpy as np
import pandas as pd


# Standard library imports
import warnings

# Data manipulation and analysis libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

# Ignore warnings
warnings.simplefilter(action='ignore', category=Warning)


import os
import pandas as pd

# Define the base input directory for Kaggle
INPUT_DIR = '/kaggle/input'

# List all directories in the input directory
all_dirs = os.listdir(INPUT_DIR)

# Assume the relevant data is in the first subdirectory (adjust if needed)
if all_dirs:  # Check if there are any directories
    competition_dir = os.path.join(INPUT_DIR, all_dirs[0])

    # Construct the file paths
    categorical_path = os.path.join(competition_dir, 'TRAIN_NEW', 'TRAIN_CATEGORICAL_METADATA_new.xlsx')
    connectome_path = os.path.join(competition_dir, 'TRAIN_NEW', 'TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv')
    quantitative_path = os.path.join(competition_dir, 'TRAIN_NEW', 'TRAIN_QUANTITATIVE_METADATA_new.xlsx')
    solutions_path = os.path.join(competition_dir, 'TRAIN_NEW', 'TRAINING_SOLUTIONS.xlsx')

    try:
        df_categorical = pd.read_excel(categorical_path)
        print(f"Shape of df_categorical: {df_categorical.shape}")

        df_connectome = pd.read_csv(connectome_path, index_col=0)
        print(f"Shape of df_connectome: {df_connectome.shape}")

        df_quantitative = pd.read_excel(quantitative_path)
        print(f"Shape of df_quantitative: {df_quantitative.shape}")

        df_solutions = pd.read_excel(solutions_path)
        print(f"Shape of df_solutions: {df_solutions.shape}")

    except FileNotFoundError:
        print("One or more files not found. Please check the directory structure and filenames.")

    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print("Error: No input directories found in /kaggle/input. Please ensure the data is attached correctly.")


# Examine each DataFrame
print("df_categorical:")
display(df_categorical.info())
display(df_categorical.describe(include='all'))
for col in df_categorical.columns:
    print(f"\nColumn: {col}")
    print(f"Unique values: {df_categorical[col].unique()}")
    print(f"Missing values: {df_categorical[col].isnull().sum()}")

print("\ndf_connectome:")
display(df_connectome.info())
display(df_connectome.describe())
print(f"Missing values: {df_connectome.isnull().sum().sum()}")

print("\ndf_quantitative:")
display(df_quantitative.info())
display(df_quantitative.describe(include='all'))
for col in df_quantitative.columns:
    print(f"\nColumn: {col}")
    print(f"Unique values: {df_quantitative[col].unique()}")
    print(f"Missing values: {df_quantitative[col].isnull().sum()}")

print("\ndf_solutions:")
display(df_solutions.info())
display(df_solutions.describe(include='all'))
for col in df_solutions.columns:
    print(f"\nColumn: {col}")
    print(f"Unique values: {df_solutions[col].unique()}")
    print(f"Missing values: {df_solutions[col].isnull().sum()}")

# Analyze the solution variable
print("\nSolution Variable Analysis:")
print(f"Unique ADHD_Outcome values: {df_solutions['ADHD_Outcome'].unique()}")
print(f"Prediction task: Classification (ADHD_Outcome is likely a binary variable)")


# Identify potential key columns for merging
print("\nPotential Key Columns for Merging:")
print("The 'participant_id' column appears to be the common key across all DataFrames.")
print("However, verify if the same 'participant_id' exists in all four datasets.")

# Check for consistency in participant_ids across dataframes.
print(f"Number of unique participant_id in df_categorical:{len(df_categorical['participant_id'].unique())}")
print(f"Number of unique participant_id in df_connectome:{len(df_connectome.index.unique())}")
print(f"Number of unique participant_id in df_quantitative:{len(df_quantitative['participant_id'].unique())}")
print(f"Number of unique participant_id in df_solutions:{len(df_solutions['participant_id'].unique())}")

#Further analysis and documentation can be added if needed.



# Impute missing values
for col in df_categorical.columns:
    if df_categorical[col].dtype == 'object' or df_categorical[col].dtype.name == 'category':
        df_categorical[col] = df_categorical[col].fillna(df_categorical[col].mode()[0])
    else:
        df_categorical[col] = df_categorical[col].fillna(df_categorical[col].median())

for col in df_quantitative.columns:
    if df_quantitative[col].dtype == 'object' or df_quantitative[col].dtype.name == 'category':
        df_quantitative[col] = df_quantitative[col].fillna(df_quantitative[col].mode()[0])
    else:
        df_quantitative[col] = df_quantitative[col].fillna(df_quantitative[col].median())

# Convert data types if necessary
if not pd.api.types.is_numeric_dtype(df_categorical['Basic_Demos_Enroll_Year']):
    df_categorical['Basic_Demos_Enroll_Year'] = pd.to_numeric(df_categorical['Basic_Demos_Enroll_Year'], errors='coerce')
    df_categorical['Basic_Demos_Enroll_Year'] = df_categorical['Basic_Demos_Enroll_Year'].fillna(df_categorical['Basic_Demos_Enroll_Year'].median()).astype(int)

# Skip connectome matrix flattening
df_connectome_flattened = df_connectome.reset_index() # Just reset the index to align with other DataFrames

# Key verification and adjustment
# Ensure participant_id is consistent across all DataFrames
df_categorical['participant_id'] = df_categorical['participant_id'].astype(str)
df_quantitative['participant_id'] = df_quantitative['participant_id'].astype(str)
df_solutions['participant_id'] = df_solutions['participant_id'].astype(str)
df_connectome_flattened['participant_id'] = df_connectome_flattened['participant_id'].astype(str)

print(f"Shape of df_categorical: {df_categorical.shape}")
print(f"Shape of df_connectome_flattened: {df_connectome_flattened.shape}")
print(f"Shape of df_quantitative: {df_quantitative.shape}")
print(f"Shape of df_solutions: {df_solutions.shape}")


# Merge the dataframes
df_temp1 = pd.merge(df_categorical, df_quantitative, on='participant_id', how='inner')
df_temp2 = pd.merge(df_temp1, df_connectome_flattened, on='participant_id', how='inner')
df_merged = pd.merge(df_temp2, df_solutions, on='participant_id', how='inner')

# Print the shape of the merged dataframe
print(f"Shape of df_merged: {df_merged.shape}")

# Display the first few rows of the merged dataframe
display(df_merged.head())

# Document any discrepancies
initial_rows = len(df_categorical)
rows_after_merge1 = len(df_temp1)
rows_after_merge2 = len(df_temp2)
rows_after_merge3 = len(df_merged)

print("\nRows lost during merging:")
print(f"Merge 1 (categorical & quantitative): {initial_rows - rows_after_merge1}")
print(f"Merge 2 (temp1 & connectome): {rows_after_merge1 - rows_after_merge2}")
print(f"Merge 3 (temp2 & solutions): {rows_after_merge2 - rows_after_merge3}")

# this section combining the four separate datasets (df_categorical, df_connectome_flattened, df_quantitative, and df_solutions) into a single dataset called df_merged.
# This is a crucial step because it allows us to analyze the relationships between different types of data for each participant.


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, confusion_matrix  # Import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
try:
    df_categorical = pd.read_excel("/content/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
    df_connectome = pd.read_csv("/content/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv", index_col=0)
    df_quantitative = pd.read_excel("/content/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
    df_solutions = pd.read_excel("/content/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
except FileNotFoundError:
    print("One or more files not found.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# Merge the dataframes
df_merged = pd.concat([df_categorical.set_index('participant_id'),
                       df_connectome,
                       df_quantitative.set_index('participant_id')], axis=1, join='inner')

# Prepare data for modeling
X = df_merged
y = df_solutions.set_index('participant_id')['ADHD_Outcome'].loc[X.index]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1}")

# Create the submission DataFrame
submission = pd.DataFrame({'participant_id': df_test_merged.index, 'ADHD_Outcome': y_pred_test})

# Save the submission file
submission.to_csv('submission.csv', index=False)  # index=False to avoid writing the index to the CSV

print("submission.csv created successfully!")


import seaborn as sns
import matplotlib.pyplot as plt

# Select quantitative features
quantitative_features = df_quantitative.select_dtypes(include=['number']).columns

# Calculate correlation matrix
corr_matrix = df_merged[quantitative_features].corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Quantitative Features')
plt.show()

# Visually identifies relationships between the quantitative features in the dataset.
#By looking at the heatmap, you can easily see which features are strongly positively correlated
 #(red), strongly negatively correlated (blue), or have little to no correlation (lighter colors)

