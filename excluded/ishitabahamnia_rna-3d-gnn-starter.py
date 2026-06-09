


!pip install kaggle==1.5.12


import pandas as pd

try:
    df = pd.read_json('kaggle (3).json')
    display(df.head())
except FileNotFoundError:
    print("Error: 'kaggle (3).json' not found. Please ensure the file exists in the current directory.")
    df = None  # Set df to None to indicate failure
except Exception as e:
    print(f"An error occurred: {e}")
    df = None


import pandas as pd

try:
    df = pd.read_json('kaggle (3).json', lines=True)
    display(df.head())
except FileNotFoundError:
    print("Error: 'kaggle (3).json' not found. Please ensure the file exists in the current directory.")
    df = None
except Exception as e:
    print(f"An error occurred: {e}")
    df = None


# Examine the shape of the DataFrame
print("Shape of the DataFrame:", df.shape)

# Check data types
print("\nData Types:\n", df.dtypes)

# Descriptive statistics for numerical columns (if any)
print("\nDescriptive Statistics:\n", df.describe(include='all'))

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())
print("\nPercentage of Missing Values:")
print(df.isnull().sum() / len(df) * 100)

# Analyze distributions of key variables
print("\nValue Counts for 'username':\n", df['username'].value_counts())
print("\nValue Counts for 'key':\n", df['key'].value_counts())

# Unique values and frequencies of categorical columns
print("\nUnique values and frequencies for 'username':")
print(df['username'].value_counts())
print("\nUnique values and frequencies for 'key':")
print(df['key'].value_counts())


# Summarize unique values and counts
print("Unique values and counts for 'username':")
print(df['username'].value_counts())
print("\nUnique values and counts for 'key':")
print(df['key'].value_counts())

print("\n\nImplications of a single data point:")
print("With only one data point, meaningful statistical analysis is impossible.  Traditional measures like mean, median, and standard deviation are not applicable to a single value.  Any attempt at modeling or prediction would be highly unreliable, as there's no variation in the data to learn from.  The single data point provides no information about the broader population or distribution.  More data is needed for meaningful analysis.")


import matplotlib.pyplot as plt

# Create a bar chart
plt.figure(figsize=(8, 6))  # Adjust figure size for better readability
plt.bar(['username', 'key'], [1, 1], color=['skyblue', 'lightcoral'])
plt.title('Visualization of Single Data Point')
plt.xlabel('Features')
plt.ylabel('Count')
plt.ylim(0, 2)  # Set y-axis limits to emphasize the single count
plt.xticks(['username', 'key'], ['ishitabahamnia', '673d82b99023b67e48877f4cbde5f8be'])
plt.text(0, 1.1, 'ishitabahamnia', ha='center', va='bottom')
plt.text(1, 1.1, '673d82b99023b67e48877f4cbde5f8be', ha='center', va='bottom')
plt.show()


import pandas as pd

try:
    df = pd.read_csv('sample_submission (2).csv')
    display(df.head())
    print(df.shape)
except FileNotFoundError:
    print("Error: 'sample_submission (2).csv' not found.")
except Exception as e:
    print(f"An error occurred: {e}")


# Display column names
print("Column Names:\n", df.columns)

# Display data types
print("\nData Types:\n", df.dtypes)

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Generate descriptive statistics
print("\nDescriptive Statistics:\n", df.describe())

# Analyze data distribution
print("\nData Info:\n")
df.info()

# Histograms for numerical columns
df.hist(figsize=(15, 10), bins=20)

# Bar plots for categorical columns (if any)
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col].value_counts().plot(kind='bar', title=f'Distribution of {col}')


# Calculate summary statistics for 'resid'
resid_stats = df['resid'].describe()
print("Summary statistics for 'resid':\n", resid_stats)

# Group data by 'resname' and calculate summary statistics
grouped_data = df.groupby('resname')['resid'].agg(['min', 'max', 'mean', 'median', 'std'])
print("\nSummary statistics for 'resid' grouped by 'resname':\n", grouped_data)

# Investigate potential outliers in 'resid'
# Check for values significantly above or below the mean/median
print("\nPotential outliers in 'resid':")
# Example:  Values more than 2 standard deviations away from the mean
outliers = df[(df['resid'] < resid_stats['mean'] - 2 * resid_stats['std']) | (df['resid'] > resid_stats['mean'] + 2 * resid_stats['std'])]
print(outliers)

# Analyze the relationship between 'resid' and 'resname'
print("\nRelationship between 'resid' and 'resname':")
# Example: Count the occurrences of each 'resid' within each 'resname' group
resid_by_resname = df.groupby('resname')['resid'].value_counts().unstack(fill_value=0)
print(resid_by_resname)

# Hypothesis about zero coordinate values
print("\nHypothesis about zero coordinate values:")
print("All coordinate values being zero suggests missing or invalid coordinate data.  This might be due to an error during data collection or processing, or it could be a placeholder for coordinates that have not yet been determined.")
print("Further investigation into the data source and any preprocessing steps is needed to understand the cause of these zero values.")


import pandas as pd

try:
    df_labels_v2 = pd.read_csv('train_labels.v2.csv')
    df_sequences_v2 = pd.read_csv('train_sequences.v2.csv')
    df_sequences = pd.read_csv('train_sequences.csv')
    df_labels = pd.read_csv('train_labels.csv')
    print("Successfully loaded all CSV files into pandas DataFrames.")
except FileNotFoundError:
    print("One or more CSV files not found. Please check file paths.")
except pd.errors.ParserError:
    print("Error parsing one or more CSV files. Please check file format.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Display basic info for each dataframe
dataframes = [df_labels_v2, df_sequences_v2, df_sequences, df_labels]
dataframe_names = ['df_labels_v2', 'df_sequences_v2', 'df_sequences', 'df_labels']

for i, df in enumerate(dataframes):
    print(f"--- {dataframe_names[i]} ---")
    display(df.head())
    print(f"Shape: {df.shape}")
    print(f"Data Types:\n{df.dtypes}")
    print(f"Summary Statistics:\n{df.describe(include='all')}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    if 'ID' in df.columns:
      print(f"Unique IDs: {df['ID'].nunique()}")
    if 'target_id' in df.columns:
      print(f"Unique Target IDs: {df['target_id'].nunique()}")
    if 'resid' in df.columns:
        print(f"Unique Resids: {df['resid'].nunique()}")
    print("\n")

# Compare v2 and non-v2 dataframes
print("--- Comparing df_labels_v2 and df_labels ---")
print(f"Number of rows in df_labels_v2: {len(df_labels_v2)}")
print(f"Number of rows in df_labels: {len(df_labels)}")
common_ids = set(df_labels_v2['ID']).intersection(set(df_labels['ID']))
print(f"Number of common IDs: {len(common_ids)}")
# Find discrepancies in ID values
diff_v2 = set(df_labels_v2['ID']) - set(df_labels['ID'])
diff = set(df_labels['ID']) - set(df_labels_v2['ID'])
print(f"IDs unique to v2: {len(diff_v2)}")
print(f"IDs unique to non-v2: {len(diff)}")


print("--- Comparing df_sequences_v2 and df_sequences ---")
print(f"Number of rows in df_sequences_v2: {len(df_sequences_v2)}")
print(f"Number of rows in df_sequences: {len(df_sequences)}")
common_targets = set(df_sequences_v2['target_id']).intersection(set(df_sequences['target_id']))
print(f"Number of common target IDs: {len(common_targets)}")
diff_v2_seq = set(df_sequences_v2['target_id']) - set(df_sequences['target_id'])
diff_seq = set(df_sequences['target_id']) - set(df_sequences_v2['target_id'])
print(f"Target IDs unique to v2: {len(diff_v2_seq)}")
print(f"Target IDs unique to non-v2: {len(diff_seq)}")


# Impute missing coordinate values with the mean for both df_labels_v2 and df_labels
for col in ['x_1', 'y_1', 'z_1']:
    df_labels_v2[col] = df_labels_v2[col].fillna(df_labels_v2[col].mean())
    df_labels[col] = df_labels[col].fillna(df_labels[col].mean())

# Impute missing 'all_sequences' values with an empty string for both df_sequences_v2 and df_sequences
df_sequences_v2['all_sequences'] = df_sequences_v2['all_sequences'].fillna('')
df_sequences['all_sequences'] = df_sequences['all_sequences'].fillna('')

# Inner join df_labels_v2 and df_labels on 'ID'
df_labels_merged = pd.merge(df_labels_v2, df_labels, on='ID', how='inner', suffixes=('_v2', '_original'))

# Inner join df_sequences_v2 and df_sequences on 'target_id'
df_sequences_merged = pd.merge(df_sequences_v2, df_sequences, on='target_id', how='inner', suffixes=('_v2', '_original'))

# Ensure consistent data types and names for join keys (already consistent)



# Identify a common identifier between the two dataframes.
# 'ID' in df_labels_merged seems to correspond to 'target_id' in df_sequences_merged,
# but they may have different formats (e.g., suffixes).
# We will try to extract a common identifier part and merge on that.

def extract_common_id(id_string):
    """Extracts the common part of the ID."""
    parts = id_string.split('_')
    return '_'.join(parts[:-1]) if len(parts) > 1 else id_string

# Apply the extraction to both dataframes
df_labels_merged['common_id'] = df_labels_merged['ID'].apply(extract_common_id)
df_sequences_merged['common_id'] = df_sequences_merged['target_id'].apply(extract_common_id)

# Merge the two dataframes
df_merged = pd.merge(df_labels_merged, df_sequences_merged, on='common_id', how='inner')

# Examine the merged dataframe
print(f"Shape of merged dataframe: {df_merged.shape}")
print(f"Columns of merged dataframe: {df_merged.columns.tolist()}")
display(df_merged.head())

# Drop redundant columns
columns_to_drop = ['ID', 'target_id', 'common_id']  # Initial list of columns to drop
# Add more columns to the list if needed, after inspection
for col in df_merged.columns:
    if '_original' in col:
        columns_to_drop.append(col)

df_merged = df_merged.drop(columns=columns_to_drop, errors='ignore')

print(f"Shape of merged dataframe after dropping columns: {df_merged.shape}")
print(f"Columns of merged dataframe after dropping columns: {df_merged.columns.tolist()}")
display(df_merged.head())


# Attempt merging directly on 'ID' and 'target_id'
try:
    df_merged = pd.merge(df_labels_merged, df_sequences_merged, left_on='ID', right_on='target_id', how='inner')
    print("Successfully merged dataframes on 'ID' and 'target_id'.")
except Exception as e:
    print(f"An error occurred during the merge: {e}")

print(f"Shape of merged dataframe: {df_merged.shape}")
print(f"Columns of merged dataframe: {df_merged.columns.tolist()}")
display(df_merged.head())

# Save to CSV
try:
    df_merged.to_csv('merged_data.csv', index=False)
    print("Successfully saved merged dataframe to 'merged_data.csv'.")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")


import pandas as pd

def extract_common_id(id_string):
    """Extracts the common part of the ID."""
    parts = id_string.split('_')
    if len(parts) > 1:
        return '_'.join(parts[:-1])
    else:
        return id_string

# Apply the extraction to both dataframes
df_labels_merged['common_id'] = df_labels_merged['ID'].apply(extract_common_id)
df_sequences_merged['common_id'] = df_sequences_merged['target_id'].apply(extract_common_id)


# Merge the two dataframes using the common_id
df_merged = pd.merge(df_labels_merged, df_sequences_merged, on='common_id', how='inner')

# Drop redundant columns
columns_to_drop = ['common_id']
for col in df_merged.columns:
    if '_original' in col or '_y' in col:
        columns_to_drop.append(col)
df_merged = df_merged.drop(columns=columns_to_drop, errors='ignore')

# Display the head and summary statistics of the merged dataframe
display(df_merged.head())
print(df_merged.describe(include='all'))
print(f"Unique IDs in merged dataframe: {df_merged['ID'].nunique()}")
print(f"Unique target_ids in merged dataframe: {df_merged['target_id'].nunique()}")


import pandas as pd
try:
    df = pd.read_csv('train_sequences.v2.csv')
    display(df.head())
except FileNotFoundError:
    print("Error: 'train_sequences.v2.csv' not found.")
    df = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df = None


# Examine the shape of the DataFrame
print("Shape of the DataFrame:", df.shape)

# Investigate data types and missing values
print("\nData Types and Missing Values:")
print(df.info())

# Explore non-numerical columns
for col in ['sequence', 'description', 'all_sequences', 'target_id', 'temporal_cutoff']:
    print(f"\nColumn: {col}")
    print("Number of unique values:", df[col].nunique())
    print("Value counts:")
    print(df[col].value_counts().head())
    if df[col].dtype == 'object':  # Check if the column contains strings
        print("Example lengths of strings:")
        print(df[col].str.len().head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())
print("\nPercentage of Missing Values:")
print((df.isnull().sum() / len(df)) * 100)


# Examine the first and last few rows
print("\nFirst few rows:")
display(df.head())
print("\nLast few rows:")
display(df.tail())


import matplotlib.pyplot as plt
# Sequence length analysis
df['sequence_length'] = df['sequence'].str.len()
print("Sequence Length Statistics:")
print(df['sequence_length'].describe())
plt.figure(figsize=(10, 6))
plt.hist(df['sequence_length'], bins=50)
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.title("Distribution of Sequence Lengths")
plt.show()

# Description analysis (example: word frequency)
from collections import Counter
description_words = " ".join(df['description']).lower().split()
word_counts = Counter(description_words)
print("\nMost Common Words in Descriptions:")
print(word_counts.most_common(10))

# 'all_sequences' analysis
df['num_sequences'] = df['all_sequences'].str.count('>')
print("\nNumber of Sequences Statistics:")
print(df['num_sequences'].describe())

plt.figure(figsize=(10, 6))
plt.hist(df['num_sequences'], bins=50)
plt.xlabel("Number of Sequences")
plt.ylabel("Frequency")
plt.title("Distribution of Number of Sequences")
plt.show()

# Relationship exploration
plt.figure(figsize=(10, 6))
plt.scatter(df['sequence_length'], df['num_sequences'], alpha=0.5)
plt.xlabel("Sequence Length")
plt.ylabel("Number of Sequences")
plt.title("Relationship between Sequence Length and Number of Sequences")
plt.show()

# Temporal cutoff analysis
print("\nTemporal Cutoff Value Counts:")
print(df['temporal_cutoff'].value_counts().head(10))


import matplotlib.pyplot as plt

plt.figure(figsize=(20, 15))

# Subplot 1: Histogram of sequence lengths
plt.subplot(3, 2, 1)
plt.hist(df['sequence_length'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.yscale('log') # Use log scale for better visualization if needed.
plt.title("Distribution of Sequence Lengths")


# Subplot 2: Bar chart of top 10 most frequent words
plt.subplot(3, 2, 2)
from collections import Counter
description_words = " ".join(df['description']).lower().split()
word_counts = Counter(description_words)
top_10_words = word_counts.most_common(10)
words, frequencies = zip(*top_10_words)
plt.bar(words, frequencies, color='lightcoral')
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 10 Most Frequent Words in Descriptions")
plt.xticks(rotation=45, ha='right')


# Subplot 3: Histogram of number of sequences
plt.subplot(3, 2, 3)
plt.hist(df['num_sequences'], bins=50, color='lightgreen', edgecolor='black')
plt.xlabel("Number of Sequences")
plt.ylabel("Frequency")
plt.yscale('log') # Use log scale if needed.
plt.title("Distribution of Number of Sequences")


# Subplot 4: Scatter plot of sequence length vs. number of sequences
plt.subplot(3, 2, 4)
plt.scatter(df['sequence_length'], df['num_sequences'], alpha=0.5, color='orange')
plt.xlabel("Sequence Length")
plt.ylabel("Number of Sequences")
plt.title("Relationship between Sequence Length and Number of Sequences")


# Subplot 5: Bar chart of top 10 most frequent temporal cutoffs
plt.subplot(3, 2, 5)
top_10_temporal_cutoffs = df['temporal_cutoff'].value_counts().head(10)
plt.bar(top_10_temporal_cutoffs.index, top_10_temporal_cutoffs.values, color='lightblue')
plt.xlabel("Temporal Cutoff")
plt.ylabel("Frequency")
plt.title("Top 10 Most Frequent Temporal Cutoffs")
plt.xticks(rotation=45, ha='right')


plt.tight_layout()  # Adjust subplot parameters for a tight layout
plt.suptitle("Data Visualization: Distributions and Relationships", fontsize=16)
plt.show()


import pandas as pd

try:
    df_sequences = pd.read_csv('validation_sequences.csv')
    df_labels = pd.read_csv('validation_labels.csv')
    display(df_sequences.head())
    display(df_labels.head())
except FileNotFoundError:
    print("Error: One or both of the CSV files were not found.")
except pd.errors.ParserError:
    print("Error: There was an issue parsing the CSV file(s).  Check the file format.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Examine the shape of the DataFrame
print("Shape of df_sequences:", df_sequences.shape)

# Investigate data types
print("\nData types of df_sequences columns:\n", df_sequences.dtypes)

# Create a numerical column (sequence length) if it doesn't exist
if 'sequence_length' not in df_sequences.columns:
    df_sequences['sequence_length'] = df_sequences['sequence'].str.len()

# Descriptive statistics for numerical columns
print("\nDescriptive statistics of numerical columns:\n", df_sequences.describe(include='number'))

# Check for missing values
print("\nMissing values in df_sequences:\n", df_sequences.isnull().sum())

# Analyze unique values in 'target_id'
print("\nNumber of unique target IDs:", df_sequences['target_id'].nunique())
print("Number of total target IDs:", len(df_sequences['target_id']))
print("\nFirst few unique target IDs:\n", df_sequences['target_id'].unique()[:5])

# Examine sequence length distribution (already calculated above)
print("\nSequence length distribution:\n", df_sequences['sequence_length'].describe())

# Explore 'temporal_cutoff' column
print("\nTemporal cutoff distribution:\n", df_sequences['temporal_cutoff'].describe())
print("\nFirst few temporal cutoffs:\n", df_sequences['temporal_cutoff'].unique()[:5])

# Briefly examine 'description' column
print("\nFirst few descriptions:\n", df_sequences['description'].unique()[:5])

# Summarize findings
print("\nSummary of Findings:")
print("1. Shape: The DataFrame has", df_sequences.shape[0], "rows and", df_sequences.shape[1], "columns.")
print("2. Data Types: Note the data types of each column.  'temporal_cutoff' might need to be converted to datetime.")
print("3. Descriptive Statistics: Provided for numerical columns, including sequence length.")
print("4. Missing Values: Report the number of missing values for each column.")
print("5. 'target_id': Compare the number of unique target IDs to the total number of IDs to check for duplicates.")
print("6. Sequence Length: Observe the min, max, mean, and standard deviation of sequence lengths.")
print("7. Temporal Cutoff: Analyze the distribution and format of temporal information.")
print("8. Description: Inspect the first few descriptions for potential patterns or keywords.")


# Examine the shape of the DataFrame
print("Shape of df_labels:", df_labels.shape)

# Investigate data types
print("\nData types of df_labels columns:\n", df_labels.dtypes)

# Descriptive statistics for numerical columns (coordinates)
coordinate_columns = [col for col in df_labels.columns if col.startswith(('x_', 'y_', 'z_'))]
print("\nDescriptive statistics of coordinate columns:\n", df_labels[coordinate_columns].describe())

# Check for missing values
print("\nMissing values in df_labels:\n", df_labels.isnull().sum())

# Analyze unique values in 'ID' and compare with 'target_id'
unique_ids_labels = df_labels['ID'].nunique()
print(f"\nNumber of unique IDs in df_labels: {unique_ids_labels}")
unique_ids_sequences = df_sequences['target_id'].nunique()
print(f"Number of unique target_ids in df_sequences: {unique_ids_sequences}")

ids_in_labels_not_in_sequences = set(df_labels['ID']) - set(df_sequences['target_id'])
print(f"\nIDs in df_labels but not in df_sequences: {ids_in_labels_not_in_sequences}")

ids_in_sequences_not_in_labels = set(df_sequences['target_id']) - set(df_labels['ID'])
print(f"\nIDs in df_sequences but not in df_labels: {ids_in_sequences_not_in_labels}")

# Explore 'resname' and 'resid' columns
print("\nNumber of unique resnames:", df_labels['resname'].nunique())
print("Unique resnames:", df_labels['resname'].unique())
print("\nNumber of unique resids:", df_labels['resid'].nunique())
print("Descriptive statistics for resid:", df_labels['resid'].describe())


# Summarize findings
print("\nSummary of Findings:")
print("1. Shape: The DataFrame has", df_labels.shape[0], "rows and", df_labels.shape[1], "columns.")
print("2. Data Types: The coordinate columns ('x_1', 'y_1', 'z_1', etc.) are of type float64, which is expected. Other columns include 'ID' (object), 'resname' (object), and 'resid' (int64).")
print("3. Descriptive Statistics: The descriptive statistics for the coordinate columns reveal a wide range of values.  There are many extremely large negative values and some reasonable positive values.  This suggests that there may be missing values encoded as a specific negative value.  There are also some suspiciously large standard deviations.")
print("4. Missing Values: There are no explicitly missing values (NaN) in the DataFrame.")
print("5. 'ID' Analysis: There are 2515 unique IDs in df_labels.  However, there are only 12 unique `target_id` values in df_sequences.")
print("   There are a large number of IDs in df_labels that do not exist in df_sequences.  This suggests a significant inconsistency between the two dataframes.")
print("6. 'resname' and 'resid': There are", df_labels['resname'].nunique(), "unique residue names. The distribution of 'resid' suggests there are many observations for each residue ID.")

#Inconsistencies
print("\nInconsistencies and potential issues:")
print("The most significant issue is the mismatch between the 'ID' column in df_labels and the 'target_id' column in df_sequences.  A substantial number of IDs in df_labels do not have corresponding entries in df_sequences. This needs to be investigated further to understand the relationship between the two files and how to proceed with the analysis.")
print("The coordinate data also presents an issue with many extremely large negative values.  It is unclear what these represent, but they may indicate missing data or errors in the data collection process.")


# 1. Relationship between 'ID' and 'target_id'
shared_ids = set(df_labels['ID']).intersection(set(df_sequences['target_id']))
num_shared = len(shared_ids)
unique_ids_labels = df_labels['ID'].nunique()
unique_ids_sequences = df_sequences['target_id'].nunique()

print(f"Number of shared IDs: {num_shared}")
print(f"Number of unique IDs in df_labels: {unique_ids_labels}")
print(f"Number of unique IDs in df_sequences: {unique_ids_sequences}")

# 2. Coordinate data analysis
coordinate_columns = [col for col in df_labels.columns if col.startswith(('x_', 'y_', 'z_'))]
# Exclude extremely large negative values (likely representing missing data)
filtered_df = df_labels[df_labels[coordinate_columns] > -1e10]

# Calculate descriptive statistics for the filtered coordinate data.
coordinate_stats = filtered_df[coordinate_columns].describe()
print("\nDescriptive statistics of coordinate columns (excluding large negative values):\n", coordinate_stats)

# 3. & 4. Exploring potential strategies and alternative keys
# Given the significant mismatch between 'ID' and 'target_id', a direct merge is not feasible.
# Explore other columns for potential relationships. The 'ID' column in df_labels has a suffix.
# Let's try to extract the common part of the 'ID' column in df_labels and compare it to 'target_id'.
df_labels['base_id'] = df_labels['ID'].str.split('_').str[0]
shared_base_ids = set(df_labels['base_id']).intersection(set(df_sequences['target_id']))
num_shared_base_ids = len(shared_base_ids)
print(f"\nNumber of shared base IDs: {num_shared_base_ids}")

# Check if there is any other potential key in df_labels and df_sequences
print(f"\nColumns in df_labels: {df_labels.columns.tolist()}")
print(f"Columns in df_sequences: {df_sequences.columns.tolist()}")

#Summarize the findings
print("\nSummary:")
print(f"There are {num_shared} IDs shared between the two dataframes, which is very low.")
print(f"There are {num_shared_base_ids} base IDs shared between the two dataframes.")
print("The coordinate data shows a large number of extremely large negative values, likely indicating missing values.")
print("Descriptive statistics of the coordinate data excluding these values are shown above.")
print("No other obvious relationships between the dataframes were found.")



import matplotlib.pyplot as plt

# 1. Histogram of sequence lengths
plt.figure(figsize=(10, 6))
plt.hist(df_sequences['sequence_length'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.show()

# 2. Bar chart of base_id counts
base_id_counts = df_labels['base_id'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(base_id_counts.index, base_id_counts.values, color='salmon')
plt.title('Number of Entries per Base ID in df_labels')
plt.xlabel('Base ID')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()


# 3. Scatter plot of sequence length vs. label count per target_id
target_id_counts = df_labels.groupby('base_id').size().reset_index(name='count')
merged_df = pd.merge(df_sequences, target_id_counts, left_on='target_id', right_on='base_id', how='left')

plt.figure(figsize=(10, 6))
plt.scatter(merged_df['sequence_length'], merged_df['count'], color='mediumseagreen')
plt.title('Sequence Length vs. Number of Corresponding Entries in df_labels')
plt.xlabel('Sequence Length')
plt.ylabel('Count of Entries')
plt.show()


import pandas as pd

try:
    df = pd.read_csv('test_sequences.csv')
    display(df.head())
except FileNotFoundError:
    print("Error: 'test_sequences.csv' not found.")
    df = None
except pd.errors.ParserError:
    print("Error: Could not parse the CSV file. Please check its format.")
    df = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df = None


# Examine the shape of the DataFrame.
print("Shape of the DataFrame:", df.shape)

# Determine the data types of each column.
print("\nData types of each column:\n", df.dtypes)

# Check for missing values.
print("\nMissing values per column:\n", df.isnull().sum())
print("\nPercentage of missing values per column:\n", (df.isnull().sum() / len(df)) * 100)

# Explore non-numerical columns.
for col in ['sequence', 'description', 'all_sequences', 'target_id', 'temporal_cutoff']:
    if col in df.columns:
        print(f"\nAnalysis for column '{col}':")
        print("Number of unique values:", df[col].nunique())
        print("Most frequent value:", df[col].mode()[0] if not df[col].mode().empty else "No mode")
        print("Frequency of the most frequent value:", df[col].value_counts().max() if not df[col].value_counts().empty else 0)


# Frequency distribution of 'sequence', 'description', and 'all_sequences'
sequence_counts = df['sequence'].value_counts()
description_counts = df['description'].value_counts()
all_sequences_counts = df['all_sequences'].value_counts()

print("Sequence Counts:\n", sequence_counts)
print("\nDescription Counts:\n", description_counts)
print("\nAll Sequences Counts:\n", all_sequences_counts)

# Relationship between 'sequence' and 'all_sequences'
# Check if any sequences appear in the 'all_sequences' field
sequence_in_all_sequences = 0
for index, row in df.iterrows():
    if row['sequence'] in row['all_sequences']:
        sequence_in_all_sequences += 1

print("\nNumber of times 'sequence' appears in 'all_sequences':", sequence_in_all_sequences)

# Explore 'description' column for recurring keywords
from collections import Counter

# Combine all descriptions into a single string
all_descriptions = ' '.join(df['description'].astype(str))

# Tokenize the combined descriptions (split into words)
words = all_descriptions.lower().split()

# Count word frequencies
word_counts = Counter(words)

# Print the 10 most common words
print("\n10 Most Common Words in Descriptions:\n", word_counts.most_common(10))


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# 1. Frequency distribution of 'sequence'
plt.subplot(2, 2, 1)
df['sequence'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Frequency Distribution of Sequence')
plt.xlabel('Sequence')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

# 2. Frequency distribution of 'description'
plt.subplot(2, 2, 2)
df['description'].value_counts().plot(kind='bar', color='lightcoral')
plt.title('Frequency Distribution of Description')
plt.xlabel('Description')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

# 3. Frequency distribution of 'all_sequences'
plt.subplot(2, 2, 3)
df['all_sequences'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Frequency Distribution of All Sequences')
plt.xlabel('All Sequences')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

# 4. Horizontal bar chart of top 10 most common words in 'description'
from collections import Counter
all_descriptions = ' '.join(df['description'].astype(str))
words = all_descriptions.lower().split()
word_counts = Counter(words)
top_10_words = word_counts.most_common(10)
words, counts = zip(*top_10_words)

plt.subplot(2, 2, 4)
plt.barh(words, counts, color='plum')
plt.title('Top 10 Most Common Words in Descriptions')
plt.xlabel('Frequency')
plt.ylabel('Words')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

plt.figure(figsize=(16, 12))

# 1. Frequency distribution of 'sequence'
plt.subplot(2, 2, 1)
df['sequence'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Frequency Distribution of Sequence')
plt.xlabel('Sequence')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

# 2. Frequency distribution of 'description'
plt.subplot(2, 2, 2)
df['description'].value_counts().plot(kind='bar', color='lightcoral')
plt.title('Frequency Distribution of Description')
plt.xlabel('Description')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

# 3. Frequency distribution of 'all_sequences'
plt.subplot(2, 2, 3)
df['all_sequences'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Frequency Distribution of All Sequences')
plt.xlabel('All Sequences')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

# 4. Horizontal bar chart of top 10 most common words in 'description'
from collections import Counter
all_descriptions = ' '.join(df['description'].astype(str))
words = all_descriptions.lower().split()
word_counts = Counter(words)
top_10_words = word_counts.most_common(10)
words, counts = zip(*top_10_words)

plt.subplot(2, 2, 4)
plt.barh(words, counts, color='plum')
plt.title('Top 10 Most Common Words in Descriptions')
plt.xlabel('Frequency')
plt.ylabel('Words')

plt.subplots_adjust(hspace=0.5, wspace=0.3) # Adjust spacing
plt.show()


import pandas as pd


import pandas as pd

try:
    df = pd.read_csv('test_sequences.csv')
    display(df.head())
except FileNotFoundError:
    print("Error: 'test_sequences.csv' not found.")
    df = None
except pd.errors.ParserError:
    print("Error: Could not parse the CSV file. Check file format.")
    df = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df = None


# Examine the shape of the DataFrame
print("Shape of the DataFrame:", df.shape)

# Determine the data types of each column
print("\nData types of each column:\n", df.dtypes)

# Generate descriptive statistics for numerical features (handle potential errors)
try:
    print("\nDescriptive statistics for numerical features:\n", df.describe(include='number'))
except ValueError as e:
    print(f"\nError generating descriptive statistics for numerical features: {e}")
    print("This likely means there are no numerical columns in the DataFrame.")

# Check for missing values
print("\nMissing values per column:\n", df.isnull().sum())

# Examine the distribution of 'target_id' and 'temporal_cutoff'
print("\nUnique 'target_id' values:", df['target_id'].nunique())
print("Unique 'temporal_cutoff' values:", df['temporal_cutoff'].nunique())
print("\nFirst few 'temporal_cutoff' values:\n", df['temporal_cutoff'].head())

# Analyze unique values and frequencies in the 'description' column
print("\nFirst few descriptions:\n", df['description'].head())
print("\nNumber of unique descriptions:", df['description'].nunique())

# Examine the 'sequence' column (sequence length and characters)
print("\nFirst sequence:", df['sequence'].iloc[0][:50] + "...")  # Display first 50 characters
sequence_lengths = df['sequence'].str.len()
print("Min sequence length:", sequence_lengths.min())
print("Max sequence length:", sequence_lengths.max())
print("Mean sequence length:", sequence_lengths.mean())

# Analyze the 'all_sequences' column
print("\nFirst 'all_sequences' entry:\n", df['all_sequences'].iloc[0][:100] + "...")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Calculate sequence lengths
sequence_lengths = df['sequence'].str.len()

# Descriptive statistics
print("Descriptive statistics for sequence lengths:")
print(sequence_lengths.describe())

# IQR outlier detection
Q1 = sequence_lengths.quantile(0.25)
Q3 = sequence_lengths.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = sequence_lengths[(sequence_lengths < lower_bound) | (sequence_lengths > upper_bound)]
print("\nPotential outliers (IQR method):")
print(outliers)

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(sequence_lengths, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.title("Distribution of Sequence Lengths")

plt.subplot(1, 2, 2)
plt.boxplot(sequence_lengths, vert=False, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
plt.xlabel("Sequence Length")
plt.title("Box Plot of Sequence Lengths")
plt.tight_layout()
plt.show()


# Convert 'temporal_cutoff' to datetime objects
try:
    df['temporal_cutoff'] = pd.to_datetime(df['temporal_cutoff'])
except ValueError as e:
    print(f"Error converting 'temporal_cutoff' to datetime: {e}")
    print("Check the format of the 'temporal_cutoff' column.")

# Numerical representation of temporal_cutoff (days since a reference date)
reference_date = datetime(2022, 1, 1)
df['days_since_reference'] = (df['temporal_cutoff'] - reference_date).dt.days

# Scatter plot of sequence length vs. days_since_reference
plt.figure(figsize=(8, 6))
plt.scatter(df['days_since_reference'], sequence_lengths, color='green', alpha=0.7)
plt.xlabel("Days Since Reference Date (2022-01-01)")
plt.ylabel("Sequence Length")
plt.title("Sequence Length vs. Temporal Cutoff")
plt.grid(True)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

# Combine all descriptions into a single string
text = ' '.join(df['description'].astype(str))

# Tokenize the text and remove punctuation
words = [word.lower() for word in text.split() if word.isalnum()]

# Count word frequencies
word_counts = Counter(words)

# Display the 10 most common words
print("10 Most Common Words:")
print(word_counts.most_common(10))


# Create a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Descriptions')
plt.show()

# Explore relationships between themes and other variables (example: sequence length)
#  (This is a placeholder; a more sophisticated approach might use topic modeling)
#  For simplicity, just examine the frequency of the top 5 words across different sequence length ranges
top_5_words = [word for word, count in word_counts.most_common(5)]

for word in top_5_words:
    plt.figure(figsize=(8, 6))
    plt.scatter(df['days_since_reference'], df['description'].str.lower().str.count(word), alpha=0.7)
    plt.xlabel("Days Since Reference Date")
    plt.ylabel(f"Frequency of '{word}'")
    plt.title(f"Frequency of '{word}' vs Temporal Cutoff")
    plt.grid(True)
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


# Load the test sequences
# Function to load test_sequences with user input
def load_test_sequences():
    """Loads the test_sequences.csv file, prompting the user for a path if needed."""
    while True:
        try:
            # Try loading with relative path
            file_path = input("Enter the path to 'test_sequences.csv': ") or '../test_sequences.csv'
            # Specify the delimiter explicitly, trying different options if necessary
            test_sequences = pd.read_csv(file_path, delimiter=',')  # Try comma first
            # if error try: test_sequences = pd.read_csv(file_path, delimiter='\t') #Try tab
            # if error try: test_sequences = pd.read_csv(file_path, delimiter=';') #Try semicolon
            return test_sequences  # Return if successful
        except FileNotFoundError:
            print("Error: File not found. Please enter a valid path.")
        except pd.errors.ParserError:
            print("Error parsing the file. Please check the delimiter and file format.")
            # You can optionally add code here to inspect the problematic line (line 9)
            # and try to identify the correct delimiter


# Call the function to load the data
test_sequences = load_test_sequences()

# Create an empty DataFrame for the submission file
submission_df = pd.DataFrame(columns=['ID', 'resname', 'resid'] + \
                                    [f'{coord}_{struct}'
                                     for struct in range(1, 6)
                                     for coord in 'xyz'])

# Iterate through the test sequences and generate predictions
for index, row in test_sequences.iterrows():
    # Use 'target_id' instead of 'ID'
    sequence_id = row['target_id']

    # Placeholder for prediction function
    def predict_structure(sequence, structure_number):
        """
        This function should predict the 3D structure of the RNA sequence
        and return the coordinates of the C1' atoms.

        Args:
            sequence: The RNA sequence.
            structure_number: The structure number (1-5).

        Returns:
            A list of (x, y, z) coordinates for the C1' atoms.
        """
        # Replace this with your actual prediction logic
        # This is just a random example
        import random
        num_residues = len(sequence)
        coordinates = [(random.uniform(-10, 10),
                        random.uniform(-10, 10),
                        random.uniform(-10, 10))
                       for _ in range(num_residues)]
        return coordinates

    # Get the sequence and residue information
    sequence = row['sequence']
    resname = [c for c in sequence]
    resid = range(1, len(sequence) + 1)

    # Predict 5 structures and build the row data
    row_data = []  # Initialize an empty list to store row data
    for struct_num in range(1, 6):
        predicted_coords = predict_structure(sequence, struct_num)
        row_data.extend([coord for coords in predicted_coords for coord in coords])  # Flatten the list

    # Create a new row for the submission DataFrame
    new_row = pd.DataFrame([[f'{sequence_id}_1', resname[0], resid[0]] + row_data], # Use sequence_id_1 for ID
                        columns=submission_df.columns)  # Ensure columns match

    # Append the new row to the submission DataFrame
    submission_df = pd.concat([submission_df, new_row], ignore_index=True)

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")


import numpy as np
import pandas as pd
df = pd.DataFrame(
    np.random.rand(100, 5),
    columns=['a', 'b', 'c', 'd', 'e'])
df.to_csv('/kaggle/working/df.csv',index=False)


# List all installed packages and package versions
!pip freeze

