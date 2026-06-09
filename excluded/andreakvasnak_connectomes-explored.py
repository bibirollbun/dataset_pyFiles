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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Create Dataframes of the Train Datasets of Data Dictionary
train_connectome=pd.read_csv(f"/kaggle/input/widsdatathon2025/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv")
train_quant=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx")
train_categorical=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx")
train_solution=pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx")


# Create 1 dataframe that joins train_solution and train_connectome on participant_id
train_outcome_connect = pd.merge(train_solution, train_connectome, on='participant_id', how='inner')


# Create 4 sub-dataframes from train_outcome_connect

# DataFrame for females with ADHD
df_female_adhd = train_outcome_connect[(train_outcome_connect['Sex_F'] == True) & (train_outcome_connect['ADHD_Outcome'] == True)]

# DataFrame for females without ADHD
df_female_no_adhd = train_outcome_connect[(train_outcome_connect['Sex_F'] == True) & (train_outcome_connect['ADHD_Outcome'] == False)]

# DataFrame for males with ADHD
df_male_adhd = train_outcome_connect[(train_outcome_connect['Sex_F'] == False) & (train_outcome_connect['ADHD_Outcome'] == True)]

# DataFrame for males without ADHD
df_male_no_adhd = train_outcome_connect[(train_outcome_connect['Sex_F'] == False) & (train_outcome_connect['ADHD_Outcome'] == False)]



# drop 'participant_id', 'Sex_F', and 'ADHD_Outcome' to create a new dataframe to visualize
df_female_adhd2 = df_female_adhd.drop(columns=['participant_id','Sex_F','ADHD_Outcome'])

# Create a subject_id that is numeric versus the participant_id which was random characters and numeric
if "subject_id" in df_female_adhd2.columns:
    subject_ids = df_female_adhd2["subject_id"].tolist()
else:
    subject_ids = df_female_adhd2.index.tolist()

# Identify columns that represent connectome features
# Here that connectome features have names containing an underscore 
connectome_features = [col for col in df_female_adhd2.columns if "_" in col]

# Parse feature names to extract unique row and column labels
row_labels = set()
col_labels = set()
for feature in connectome_features:
    # Split each feature into its row and column parts.
    row_part, col_part = feature.split("_")
    row_labels.add(row_part)
    col_labels.add(col_part)

# Sort the labels so that the matrix rows and columns are in a consistent order
row_labels = sorted(list(row_labels))
col_labels = sorted(list(col_labels))

# Create mapping dictionaries from label to matrix index
row_to_idx = {label: idx for idx, label in enumerate(row_labels)}
col_to_idx = {label: idx for idx, label in enumerate(col_labels)}

# Create a dictionary to store each subject's connectome matrix
connectome_matrices = {}

# Loop over each subject 
for i, row in df_female_adhd2.iterrows():
    # Use the subject_id if available, otherwise use the row index
    subj_id = row["subject_id"] if "subject_id" in df_female_adhd2.columns else i
    
    # Initialize an empty matrix with dimensions based on the unique row and column labels
    matrix = np.zeros((len(row_labels), len(col_labels)))
    
    # Populate the matrix with values from the connectome features
    for feature in connectome_features:
        row_label, col_label = feature.split("_")
        value = row[feature]
        # Use the mappings to get the correct indices in the matrix
        i_idx = row_to_idx[row_label]
        j_idx = col_to_idx[col_label]
        matrix[i_idx, j_idx] = value
    
    # Save the constructed matrix for the subject
    connectome_matrices[subj_id] = matrix

# Retrieve the connectome matrix for the first subject
first_subj = next(iter(connectome_matrices))
first_matrix = connectome_matrices[first_subj]
print("Subject:", first_subj)
print("Row labels:", row_labels)
print("Column labels:", col_labels)
print("Connectome Matrix:")
print(first_matrix)

# Create a heatmap to visualize the connectome matrix for the first subject
plt.figure(figsize=(10, 8))
#sns.heatmap(first_matrix, xticklabels=col_labels, yticklabels=row_labels, cmap="viridis", annot=True, fmt=".2f")
sns.heatmap(first_matrix, cmap="viridis", annot=True, fmt=".2f")
plt.title(f"Connectome Matrix for Subject {first_subj}")
#plt.xlabel("Column Labels")
#plt.ylabel("Row Labels")
#plt.tight_layout()
plt.show()


# Extract the First Column for Subject 1
first_matrix_col1 = first_matrix[:, 0]  
 

# Create an x-axis index for plotting
x = np.arange(first_matrix_col1.shape[0])

# Create a color array: red for negative values, blue for positive values
colors = np.where(first_matrix_col1 < 0, 'red', 'blue')

# Plot the data for subject 1 Column 1
#plt.plot(first_matrix_no_col1, color=colors, marker='o')
plt.scatter(x, first_matrix_col1, color=colors, marker='o')
plt.title("Graph of Column for Subject 1 Female ADHD")
plt.ylim(-0.2, 0.2)  # Set the y-axis limits to -0.2 and 0.2
plt.show()


# drop 'participant_id', 'Sex_F', and 'ADHD_Outcome' to create a new dataframe to visualize
df_female_no_adhd2 = df_female_no_adhd.drop(columns=['participant_id','Sex_F','ADHD_Outcome'])

# Create a subject_id that is numeric versus the participant_id which was random characters and numeric
if "subject_id" in df_female_no_adhd2.columns:
    subject_ids = df_female_no_adhd2["subject_id"].tolist()
else:
    subject_ids = df_female_no_adhd2.index.tolist()

# Identify columns that represent connectome features
# Here that connectome features have names containing an underscore 
connectome_features_no = [col for col in df_female_no_adhd2.columns if "_" in col]

# Parse feature names to extract unique row and column labels
row_labels = set()
col_labels = set()
for feature in connectome_features_no:
    # Split each feature into its row and column parts.
    row_part, col_part = feature.split("_")
    row_labels.add(row_part)
    col_labels.add(col_part)

# Sort the labels so that the matrix rows and columns are in a consistent order
row_labels = sorted(list(row_labels))
col_labels = sorted(list(col_labels))

# Create mapping dictionaries from label to matrix index
row_to_idx = {label: idx for idx, label in enumerate(row_labels)}
col_to_idx = {label: idx for idx, label in enumerate(col_labels)}

# Create a dictionary to store each subject's connectome matrix
connectome_matrices_no = {}

# Loop over each subject 
for i, row in df_female_no_adhd2.iterrows():
    # Use the subject_id if available, otherwise use the row index
    subj_id = row["subject_id"] if "subject_id" in df_female_no_adhd2.columns else i
    
    # Initialize an empty matrix with dimensions based on the unique row and column labels
    matrix_no = np.zeros((len(row_labels), len(col_labels)))
    
    # Populate the matrix with values from the connectome features
    for feature in connectome_features_no:
        row_label, col_label = feature.split("_")
        value = row[feature]
        # Use the mappings to get the correct indices in the matrix
        i_idx = row_to_idx[row_label]
        j_idx = col_to_idx[col_label]
        matrix_no[i_idx, j_idx] = value
    
    # Save the constructed matrix for the subject
    connectome_matrices_no[subj_id] = matrix_no

# Retrieve the connectome matrix for the first subject
first_subj_no = next(iter(connectome_matrices_no))
first_matrix_no = connectome_matrices_no[first_subj_no]
print("Subject:", first_subj_no)
print("Row labels:", row_labels)
print("Column labels:", col_labels)
print("Connectome Matrix:")
print(first_matrix_no)


# Extract the First Column for Subject 1
first_matrix_no_col1 = first_matrix_no[:, 0]  

# Create an x-axis index for plotting
x = np.arange(first_matrix_no_col1.shape[0])

# Create a color array: red for negative values, blue for positive values
colors = np.where(first_matrix_no_col1 < 0, 'red', 'blue')

# Plot the data for subject 1 Column 1
#plt.plot(first_matrix_no_col1, color=colors, marker='o')
plt.scatter(x, first_matrix_no_col1, color=colors, marker='o')
plt.title("Graph of Column for Subject 1 Female No ADHD")
plt.ylim(-0.2, 0.2)  # Set the y-axis limits to -0.2 and 0.2
plt.show()



# drop 'participant_id', 'Sex_F', and 'ADHD_Outcome' to create a new dataframe to visualize
df_male_adhd2 = df_male_adhd.drop(columns=['participant_id','Sex_F','ADHD_Outcome'])

# Create a subject_id that is numeric versus the participant_id which was random characters and numeric
if "subject_id" in df_male_adhd2.columns:
    subject_ids = df_male_adhd2["subject_id"].tolist()
else:
    subject_ids = df_male_adhd2.index.tolist()

# Identify columns that represent connectome features
# Here that connectome features have names containing an underscore 
connectome_features_m = [col for col in df_male_adhd2.columns if "_" in col]

# Parse feature names to extract unique row and column labels
row_labels = set()
col_labels = set()
for feature in connectome_features_m:
    # Split each feature into its row and column parts.
    row_part, col_part = feature.split("_")
    row_labels.add(row_part)
    col_labels.add(col_part)

# Sort the labels so that the matrix rows and columns are in a consistent order
row_labels = sorted(list(row_labels))
col_labels = sorted(list(col_labels))

# Create mapping dictionaries from label to matrix index
row_to_idx = {label: idx for idx, label in enumerate(row_labels)}
col_to_idx = {label: idx for idx, label in enumerate(col_labels)}

# Create a dictionary to store each subject's connectome matrix
connectome_matrices_m = {}

# Loop over each subject 
for i, row in df_male_adhd2.iterrows():
    # Use the subject_id if available, otherwise use the row index
    subj_id = row["subject_id"] if "subject_id" in df_male_adhd2.columns else i
    
    # Initialize an empty matrix with dimensions based on the unique row and column labels
    matrix_m = np.zeros((len(row_labels), len(col_labels)))
    
    # Populate the matrix with values from the connectome features
    for feature in connectome_features_m:
        row_label, col_label = feature.split("_")
        value = row[feature]
        # Use the mappings to get the correct indices in the matrix
        i_idx = row_to_idx[row_label]
        j_idx = col_to_idx[col_label]
        matrix_m[i_idx, j_idx] = value
    
    # Save the constructed matrix for the subject
    connectome_matrices_m[subj_id] = matrix_m

# Retrieve the connectome matrix for the first subject
first_subj_m = next(iter(connectome_matrices_m))
first_matrix_m = connectome_matrices_m[first_subj_m]
print("Subject:", first_subj_m)
print("Row labels:", row_labels)
print("Column labels:", col_labels)
print("Connectome Matrix:")
print(first_matrix_m)


# Extract the First Column for Subject 1
first_matrix_m_col1 = first_matrix_m[:, 0]  

# Create an x-axis index for plotting
x = np.arange(first_matrix_m_col1.shape[0])

# Create a color array: red for negative values, blue for positive values
colors = np.where(first_matrix_m_col1 < 0, 'red', 'blue')

# Plot the data for subject 1 Column 1
plt.scatter(x, first_matrix_m_col1, color=colors, marker='o')
plt.title("Graph of Column for Subject 1 Male ADHD")
plt.ylim(-0.2, 0.2)  # Set the y-axis limits to -0.2 and 0.2
plt.show()



# drop 'participant_id', 'Sex_F', and 'ADHD_Outcome' to create a new dataframe to visualize
df_male_no_adhd2 = df_male_no_adhd.drop(columns=['participant_id','Sex_F','ADHD_Outcome'])

# Create a subject_id that is numeric versus the participant_id which was random characters and numeric
if "subject_id" in df_male_no_adhd2.columns:
    subject_ids = df_male_no_adhd2["subject_id"].tolist()
else:
    subject_ids = df_male_no_adhd2.index.tolist()

# Identify columns that represent connectome features
# Here that connectome features have names containing an underscore 
connectome_features_m_no = [col for col in df_male_no_adhd2.columns if "_" in col]

# Parse feature names to extract unique row and column labels
row_labels = set()
col_labels = set()
for feature in connectome_features_m_no:
    # Split each feature into its row and column parts.
    row_part, col_part = feature.split("_")
    row_labels.add(row_part)
    col_labels.add(col_part)

# Sort the labels so that the matrix rows and columns are in a consistent order
row_labels = sorted(list(row_labels))
col_labels = sorted(list(col_labels))

# Create mapping dictionaries from label to matrix index
row_to_idx = {label: idx for idx, label in enumerate(row_labels)}
col_to_idx = {label: idx for idx, label in enumerate(col_labels)}

# Create a dictionary to store each subject's connectome matrix
connectome_matrices_m_no = {}

# Loop over each subject 
for i, row in df_male_no_adhd2.iterrows():
    # Use the subject_id if available, otherwise use the row index
    subj_id = row["subject_id"] if "subject_id" in df_male_no_adhd2.columns else i
    
    # Initialize an empty matrix with dimensions based on the unique row and column labels
    matrix_m = np.zeros((len(row_labels), len(col_labels)))
    
    # Populate the matrix with values from the connectome features
    for feature in connectome_features_m_no:
        row_label, col_label = feature.split("_")
        value = row[feature]
        # Use the mappings to get the correct indices in the matrix
        i_idx = row_to_idx[row_label]
        j_idx = col_to_idx[col_label]
        matrix_m[i_idx, j_idx] = value
    
    # Save the constructed matrix for the subject
    connectome_matrices_m_no[subj_id] = matrix_m

# Retrieve the connectome matrix for the first subject
first_subj_m_no = next(iter(connectome_matrices_m_no))
first_matrix_m_no = connectome_matrices_m_no[first_subj_m_no]
print("Subject:", first_subj_m_no)
print("Row labels:", row_labels)
print("Column labels:", col_labels)
print("Connectome Matrix:")
print(first_matrix_m_no)


# Extract the First Column for Subject 1
first_matrix_m_no_col1 = first_matrix_m_no[:, 0]  

# Create an x-axis index for plotting
x = np.arange(first_matrix_m_no_col1.shape[0])

# Create a color array: red for negative values, blue for positive values
colors = np.where(first_matrix_m_no_col1 < 0, 'red', 'blue')

# Plot the data for subject 1 Column 1
plt.scatter(x, first_matrix_m_no_col1, color=colors, marker='o')
plt.title("Graph of Column for Subject 1 Male No ADHD")
plt.ylim(-0.2, 0.2)  # Set the y-axis limits to -0.2 and 0.2
plt.show()

