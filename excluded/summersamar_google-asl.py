import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


train_csv_path = "/kaggle/input/asl-signs/train.csv"
train_df = pd.read_csv(train_csv_path)

print('Shape of train.csv ',train_df.shape)

print('Data types and not null count')
train_df.info()

print("\nMissing Values in Each Column:")
print(train_df.isnull().sum())

print("\nNumber of Unique Signs (Classes):", train_df['sign'].nunique())

train_df.head()


from ydata_profiling import ProfileReport

profile = ProfileReport(train_df, title="ASL Sign Language Profile Report", explorative=True)
profile.to_notebook_iframe()



sign_classes = train_df['sign'].value_counts()

#Barplot for TOP 20 sign counts
plt.figure(figsize=(15,5))
top_20 = sign_classes.head(20)
sns.barplot(x=top_20.index, y=top_20.values)
plt.title('Top 20 most frequent signs')
plt.xlabel('Sign class')
plt.ylabel('Frequency')
plt.show()

#Barplot for Bottom 20 sign counts
plt.figure(figsize=(15,5))
bottom_20 = sign_classes.tail(20)
sns.barplot(x=bottom_20.index, y=bottom_20.values)
plt.title('Bottom 20 least frequent signs')
plt.xlabel('Sign class')
plt.ylabel('Frequency')
plt.show()

print('Minimum samples per sign: ',sign_classes.min())
print('Maximum samples per sign: ',sign_classes.max())
print('Median samples per sign: ',sign_classes.median())
print('Mean samples per sign: ',sign_classes.mean())
print('\nTop 20 signs: \n',top_20)
print('Bottom 20 signs: \n',bottom_20)


participant_counts = train_df['participant_id'].value_counts()

#Barplot for number of sequences per participant
plt.figure(figsize=(15,5))
sns.barplot(x=participant_counts.index, y=participant_counts.values)
plt.title('Number of sequences per participant')
plt.xlabel('Participant ID')
plt.ylabel('Number of sequences')
plt.xticks(rotation=90)
plt.show()

# Number of unique signs per participant
participant_signs = train_df.groupby('participant_id')['sign'].nunique()
plt.figure(figsize=(15,5))
sns.barplot(x=participant_signs.index, y=participant_signs.values)
plt.title('Number of unique signs per participant')
plt.xlabel('Participant_ID')
plt.ylabel('Number of unique signs')
plt.show()

print('Number of unique participant_ids:', train_df['participant_id'].nunique())
print('Min sequences per participant:', participant_counts.min())
print('Max sequences per participant:', participant_counts.max())
print('Mean sequences per participant:', participant_counts.mean())
print('Median sequences per participant:', participant_counts.median())


landmark_directory = '/kaggle/input/asl-signs/train_landmark_files'
directory_list = os.listdir(landmark_directory)
print('List of directories inside train_landmark_files: \n',directory_list)

participant_dirs = []
for directory in directory_list:
    directory_path = os.path.join(landmark_directory, directory)

    if os.path.isdir(directory_path):
        participant_dirs.append(directory)

print(f'Number of participant folders: {len(participant_dirs)}')
print('Participant IDs: ',participant_dirs)


# import os
# import pandas as pd
# import random

# # Set the target sample size:  
# # We want to load a maximum of 4000 files in total
# FILE_LIMIT = 210

# # Set the per participant limit:  
# # From each participant, we want at most 200 files
# PER_PARTICIPANT_LIMIT = 10

# # Initialize a counter to track how many files have been loaded in total so far
# files_loaded = 0

# # Set the output file path where we will save the final combined dataset
# output_file_path = '/kaggle/working/sample_landmarks_data.parquet'

# # Initialize an empty DataFrame to hold all combined participant data
# landmarks_df = pd.DataFrame()

# # Start looping through each participant folder
# for participant in participant_dirs:
    
#     # Construct the full path to the current participant's folder
#     participant_path = os.path.join(landmark_directory, participant)
    
#     # List all files in the current participant's folder
#     file_list = os.listdir(participant_path)

#     # Initialize an empty list to store only `.parquet` file names
#     parquet_files = []
    
#     # Loop through each file in the folder and filter `.parquet` files
#     for file in file_list:
#         if file.endswith('.parquet'):
#             parquet_files.append(file)

#     # Determine how many files we can sample from this participant
#     if len(parquet_files) >= PER_PARTICIPANT_LIMIT:
#         number_of_files_to_sample = PER_PARTICIPANT_LIMIT
#     else:
#         number_of_files_to_sample = len(parquet_files)

#     # If there are files to sample, randomly select them using random.sample
#     if number_of_files_to_sample > 0:
#         sample_files = random.sample(parquet_files, number_of_files_to_sample)
#     else:
#         sample_files = []  # In case participant folder is empty

#     # Initialize a DataFrame to hold this participant’s sampled files
#     participant_df = pd.DataFrame()

#     # Loop over each sampled `.parquet` file for this participant
#     for parquet_file in sample_files:
        
#         # Construct the full path to the current `.parquet` file
#         file_path = os.path.join(participant_path, parquet_file)
        
#         # Read the `.parquet` file into a DataFrame
#         df = pd.read_parquet(file_path)

#         # Add a column indicating the participant ID
#         df['participant_id'] = participant

#         # Add a column indicating the file name
#         df['file_name'] = parquet_file

#         # Concatenate this file’s data to the participant's DataFrame
#         participant_df = pd.concat([participant_df, df], ignore_index=True)

#         # Increment the **global file counter** since we loaded one more file
#         files_loaded = files_loaded + 1

#         # Check if we've reached the global file limit (FILE_LIMIT)
#         if files_loaded >= FILE_LIMIT:
#             break  # Stop loading more files

#     # After processing this participant, append their data to the main DataFrame
#     landmarks_df = pd.concat([landmarks_df, participant_df], ignore_index=True)

#     # Delete the participant DataFrame to free memory
#     del participant_df

#     # Check again if we've reached the global file limit
#     if files_loaded >= FILE_LIMIT:
#         print(f'Reached file limit of {FILE_LIMIT}. Stopping...')
#         break

# # After processing, save the combined DataFrame to a `.parquet` file on disk
# landmarks_df.to_parquet(output_file_path, index=False)

# print(f'Shape of combined sampled landmark data: {landmarks_df.shape}')
# landmarks_df.head()


landmarks_df = pd.read_parquet('/kaggle/input/sample-landmarks-dataset/sample_landmarks_data.parquet')


print('Shape of landmarks_df: ',landmarks_df.shape)
print('---------------------------------------------')
print(landmarks_df.info())
print('---------------------------------------------')
print('Missing values in landmarks_df: ',landmarks_df.isna().sum())
print('---------------------------------------------')
landmarks_df.head()


landmarks_df[['frame','landmark_index','x','y','z']].describe()


# Histogram to see distribution of 'x' coordinate
plt.figure(figsize=(20,6))

plt.subplot(1, 3, 1)
sns.histplot(landmarks_df['x'], bins=50, kde=True, color='skyblue')
plt.title("Distribution of 'x'")
plt.xlabel('x')

plt.subplot(1, 3, 2)
sns.histplot(landmarks_df['y'], bins=50, kde=True, color='lightgreen')
plt.title("Distribution of 'y'")
plt.xlabel('y')

plt.subplot(1, 3, 3)
sns.histplot(landmarks_df['z'], bins=50, kde=True, color='salmon')
plt.title("Distribution of 'z'")
plt.xlabel('z')

plt.tight_layout()
plt.show()

# Boxplot to see outliers
plt.figure(figsize=(25, 8))

plt.subplot(1, 3, 1)
sns.boxplot(x=landmarks_df['x'], color='lightblue')
plt.title("Boxplot of x")

plt.subplot(1, 3, 2)
sns.boxplot(x=landmarks_df['y'], color='lightgreen')
plt.title("Boxplot of y")

plt.subplot(1, 3, 3)
sns.boxplot(x=landmarks_df['z'], color='salmon')
plt.title("Boxplot of z")

plt.tight_layout()
plt.show()


print("Skewness of x:", landmarks_df['x'].skew())
print("Skewness of y:", landmarks_df['y'].skew())
print("Skewness of z:", landmarks_df['z'].skew())


Q1 = landmarks_df['x'].quantile(0.25)  # 25th percentile
Q3 = landmarks_df['x'].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1
print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")

outliers_lower = landmarks_df[landmarks_df['x'] < lower_bound]
outliers_upper = landmarks_df[landmarks_df['x'] > upper_bound]

outliers = pd.concat([outliers_lower, outliers_upper])
print(f"Outliers detected: {outliers.shape[0]} rows")
outliers.head()



plt.figure(figsize=(10,4))
sns.histplot(landmarks_df['frame'], bins=100, kde=False, color='teal')
plt.title('Distribution of Frame Numbers')
plt.xlabel('Frame Number')
plt.ylabel('Frequency')
plt.show()


print("Value counts for 'type' column: ",landmarks_df['type'].value_counts())
plt.figure(figsize=(10, 5))
sns.countplot(data=landmarks_df, x='type', palette='Set1')
plt.title('Distribution of Landmark Types')
plt.xlabel('Type of Landmark')
plt.ylabel('Count')
plt.show()


missing_by_type = landmarks_df.groupby('type')[['x', 'y', 'z']].apply(lambda group: group.isna().sum())
print(missing_by_type,end='\n')

total_values_by_type = landmarks_df.groupby('type')[['x', 'y', 'z']].count()
print('Total values by type:\n',total_values_by_type)


missing_percent = (missing_by_type / (missing_by_type + total_values_by_type)) * 100
print("Missing percentage:\n", missing_percent)


missing_by_type.plot(kind='bar', figsize=(10,6), colormap='Set2')
plt.title("Missing Values per Coordinate by Type")
plt.xlabel("Landmark Type")
plt.ylabel("Number of Missing Values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


top_landmarks = landmarks_df['landmark_index'].value_counts().head(20)
plt.figure(figsize=(10,5))
sns.barplot(x=top_landmarks.index, y=top_landmarks.values, palette='viridis')
plt.title('Top 20 Most Frequent Landmark Indexes')
plt.xlabel('Landmark Index')
plt.ylabel('Count')
plt.show()

bottom_landmarks = landmarks_df['landmark_index'].value_counts().tail(20)
plt.figure(figsize=(10,5))
sns.barplot(x=bottom_landmarks.index, y=bottom_landmarks.values, palette='viridis')
plt.title('Bottom 20 Least Frequent Landmark Indexes')
plt.xlabel('Landmark Index')
plt.ylabel('Count')
plt.show()






sample_df = landmarks_df[['x', 'y', 'z']].dropna().sample(10000, random_state=42)
sns.pairplot(sample_df, plot_kws={'alpha':0.4, 's':10})
plt.suptitle("Scatter plots among x, y, z", y=1.02)
plt.show()


numerical_cols = ['x','y','z']
plt.figure(figsize=(8,4))
sns.heatmap(landmarks_df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix (for x, y, z)')
plt.show


plt.figure(figsize=(12, 6))
sns.boxplot(x='type', y='x', data=landmarks_df, palette='Set2')
plt.title("Distribution of 'x' coordinate by Landmark Type")
plt.xlabel('Landmark Type')
plt.ylabel('X Coordinate')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='type', y='y', data=landmarks_df, palette='Set2')
plt.title("Distribution of 'y' coordinate by Landmark Type")
plt.xlabel('Landmark Type')
plt.ylabel('Y Coordinate')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='type', y='z', data=landmarks_df, palette='Set2')
plt.title("Distribution of 'z' coordinate by Landmark Type")
plt.xlabel('Landmark Type')
plt.ylabel('Z Coordinate')
plt.show()


iqr_stats = landmarks_df.groupby('type')[['x', 'y', 'z']].quantile([0.25, 0.75]).unstack()
iqr_stats.columns = ['Q1_x', 'Q3_x', 'Q1_y', 'Q3_y', 'Q1_z', 'Q3_z']

iqr_stats['IQR_x'] = iqr_stats['Q3_x'] - iqr_stats['Q1_x']
iqr_stats['IQR_y'] = iqr_stats['Q3_y'] - iqr_stats['Q1_y']
iqr_stats['IQR_z'] = iqr_stats['Q3_z'] - iqr_stats['Q1_z']

print(iqr_stats[['IQR_x', 'IQR_y', 'IQR_z']])

print('Upper and Lower bounds for each coordinates: ')
iqr_stats['lower_x'] = iqr_stats['Q1_x'] - 1.5 * iqr_stats['IQR_x']
iqr_stats['upper_x'] = iqr_stats['Q3_x'] + 1.5 * iqr_stats['IQR_x']

iqr_stats['lower_y'] = iqr_stats['Q1_y'] - 1.5 * iqr_stats['IQR_y']
iqr_stats['upper_y'] = iqr_stats['Q3_y'] + 1.5 * iqr_stats['IQR_y']

iqr_stats['lower_z'] = iqr_stats['Q1_z'] - 1.5 * iqr_stats['IQR_z']
iqr_stats['upper_z'] = iqr_stats['Q3_z'] + 1.5 * iqr_stats['IQR_z']

print('Upper bound for "X" coordinate')
print(iqr_stats['upper_x'],'\n')
print('Lower bound for "X" coordinate')
print(iqr_stats['lower_x'])

print('---------------------------------')
print('Upper bound for "Y" coordinate')
print(iqr_stats['upper_y'],'\n')
print('Lower bound for "Y" coordinate')
print(iqr_stats['lower_y'])

print('---------------------------------')
print('Upper bound for "Z" coordinate')
print(iqr_stats['upper_z'],'\n')
print('Lower bound for "Z" coordinate')
print(iqr_stats['lower_y'])


landmarks_with_bounds = landmarks_df.merge(iqr_stats, on='type')
landmarks_with_bounds.sample(10)


landmarks_with_bounds['x_outlier'] = (
    (landmarks_with_bounds['x'] < landmarks_with_bounds['lower_x']) |
    (landmarks_with_bounds['x'] > landmarks_with_bounds['upper_x'])
)

landmarks_with_bounds['y_outlier'] = (
    (landmarks_with_bounds['y'] < landmarks_with_bounds['lower_y']) |
    (landmarks_with_bounds['y'] > landmarks_with_bounds['upper_y'])
)

landmarks_with_bounds['z_outlier'] = (
    (landmarks_with_bounds['z'] < landmarks_with_bounds['lower_z']) |
    (landmarks_with_bounds['z'] > landmarks_with_bounds['upper_z'])
)
outlier_summary = landmarks_with_bounds.groupby('type')[['x_outlier', 'y_outlier', 'z_outlier']].sum()
print(outlier_summary)


outlier_summary.plot(kind='bar', figsize=(10, 6))
plt.title('Number of Outliers per Type and Coordinate Axis')
plt.ylabel('Outlier Count')
plt.xlabel('Landmark Type')
plt.legend(title='Coordinate')
plt.tight_layout()
plt.show()

