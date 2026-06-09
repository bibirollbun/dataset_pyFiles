# import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


folder_path = "/kaggle/input/widsdatathon2025/"


def load_data(file_path, file_type):
  '''Load data from a file.

    Args:
        file_path (str): Path to the file.
        file_type (str): Type of the file ('csv' or 'excel').

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        ValueError: If the file type is not supported.'''   
  if file_type == 'csv':
    data = pd.read_csv(file_path)
    return data
  elif file_type == 'excel':
    data = pd.read_excel(file_path)
    return data
  else:
    raise ValueError("Unsupported file type. Use 'csv' or 'excel'.")


def handle_datatypes(train_df, test_df):
   '''Handle mismatching data types between train and test data.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.

    Returns:
        pd.DataFrame: Test data with corrected data types.'''

   unmatching_dtypes = {col: (train_df.dtypes[col], test_df.dtypes[col]) 
                    for col in train_df.columns 
                        if col in test_df.columns and train_df.dtypes[col] != test_df.dtypes[col]}
   print("Unmatching data types:")
   display(unmatching_dtypes)
   for col, (train_dtype, test_dtype) in unmatching_dtypes.items():
     if train_dtype == np.int64:
       # replace NaN with 0 before converting
       test_df[col] = test_df[col].fillna(0).astype('int64')
   return test_df


def one_hot_encode_categorical_data(df, columns):
  '''One-hot encode categorical columns in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to be one-hot encoded.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns.'''
      
  encoded_df = pd.get_dummies(df, columns=columns, prefix=columns, prefix_sep='-', drop_first=True)
  encoded_df = encoded_df.map(lambda x: 1 if x == True else (0 if x == False else x))
  return encoded_df


def merge_data(fmri_data, quantitative_data, categorical_data):
  '''Merge fMRI, quantitative, and categorical data.

    Args:
        fmri_data (pd.DataFrame): fMRI data.
        quantitative_data (pd.DataFrame): Quantitative data.
        categorical_data (pd.DataFrame): Categorical data.

    Returns:
        pd.DataFrame'''
        
  merged_data = pd.merge(fmri_data, quantitative_data, on='participant_id', how='left')
  merged_data = pd.merge(merged_data, categorical_data, on='participant_id',how='left')
  return merged_data



# print all the files in sub-directories at folder_path
import os
for root, dirs, files in os.walk(folder_path):
    for file in files:
        print(os.path.join(root, file))


train_fmri_data = load_data(f"{folder_path}/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv", "csv")
train_categorical_data = load_data(f"{folder_path}/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx", "excel")
train_quantitative_data = load_data(f"{folder_path}/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx", "excel")


from google.colab import sheets
sheet = sheets.InteractiveSheet(df=train_categorical_data)


test_fmri_data = load_data(f"{folder_path}/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv", "csv")
test_categorical_data = load_data(f"{folder_path}/TEST/TEST_CATEGORICAL.xlsx", "excel")
test_quantitative_data = load_data(f"{folder_path}/TEST/TEST_QUANTITATIVE_METADATA.xlsx", "excel")


validation_data = load_data(f"{folder_path}/TRAIN/TRAINING_SOLUTIONS.xlsx", "excel")
sample_submission = load_data(f"{folder_path}/SAMPLE_SUBMISSION.xlsx", "excel")
data_dictionary = load_data(f"{folder_path}/Data Dictionary.xlsx", "excel")



print(display(validation_data.head()))
print(display(sample_submission.head()))
print(display(data_dictionary.head()))


!pip install ydata-profiling


!pip install great_expectations


from ydata_profiling import ProfileReport

def create_profile_report(df, title):
    profile = ProfileReport(df, title=title)
    return profile



train_categorical_data_profile = create_profile_report(train_categorical_data, "Train Categorical Data Profiling Report")


validation_data_profile = create_profile_report(validation_data, "Validation Data Profiling Report")
validation_data_profile


# find the data where there ADHD = 1 and Sex_F = 1 from validation_data

display(validation_data[(validation_data['ADHD_Outcome'] == 1) & (validation_data['Sex_F'] == 1)]) # 250 rows
print(f'Percentage of female with ADHD in the validation set is {250/1200*100:.2f}%')


# let's see the single variable distribution
# plot a histplot for train_quantitative_data['SDQ_SDQ_Emotional_Problems']
sns.histplot(train_quantitative_data['SDQ_SDQ_Emotional_Problems'], kde=True, color = 'orange', bins=50)
plt.xlabel('SDQ_SDQ_Emotional_Problems')
plt.ylabel('Frequency')
plt.title('Distribution of SDQ_SDQ_Emotional_Problems')
plt.show()


# use ydata profiling to analyze quantitative data
train_quantitative_data_profile = create_profile_report(train_quantitative_data, "Train Quantitative Data Profiling Report")
train_quantitative_data_profile


train_quantitative_data.shape


validation_data.shape


# merge quantitative data and validata data
train_data = pd.merge(train_quantitative_data, validation_data, on='participant_id')
train_data.shape


# merge the data with categorical data
train_data = pd.merge(train_data, train_categorical_data, on='participant_id')
# create a ydata profile report
train_data_profile = create_profile_report(train_data, "Train Data Profiling Report")


train_data_profile


adhd_percentages = train_data.groupby('Barratt_Barratt_P1_Edu')['ADHD_Outcome'].mean() * 100
adhd_percentages


train_categorical_data.head()


test_categorical_data.head()


one_hot_encoding_columns = ['Basic_Demos_Enroll_Year', 'Basic_Demos_Study_Site',
       'PreInt_Demos_Fam_Child_Ethnicity', 'PreInt_Demos_Fam_Child_Race',
       'MRI_Track_Scan_Location',
       'Barratt_Barratt_P1_Occ',
       'Barratt_Barratt_P2_Occ']


print(f"First few rows of train categorical data before encoding is \n {train_categorical_data.info()}")
print(f"First few rows of test categorical data before encoding is \n {display(test_categorical_data.info())}")


train_categorical_encoded_data = one_hot_encode_categorical_data(train_categorical_data, one_hot_encoding_columns)
#test_df = handle_datatypes(train_categorical_data, test_categorical_data)
test_categorical_encoded_data = one_hot_encode_categorical_data(test_df, one_hot_encoding_columns)


print(f"Shape of training categorical data before encoding{train_categorical_data.shape} and after is {train_categorical_encoded_data.shape}")
print(f"Shape of test categorical data before encoding{test_categorical_data.shape} and after is {test_categorical_encoded_data.shape}")


missing_in_train = set(test_categorical_encoded_data.columns) - set(train_categorical_data_encoded.columns)
missing_in_test = set(train_categorical_data_encoded.columns) - set(test_categorical_encoded_data.columns)

missing_columns = {
    "missing_in_train": list(missing_in_train),
    "missing_in_test": list(missing_in_test)
}

missing_columns


# drop missing in train columns from test data set
test_categorical_encoded_data.drop(columns=missing_in_train, inplace=True)
test_categorical_encoded_data.shape


# add missing columns in test
for col in missing_in_test:
  test_categorical_encoded_data[col] = 0
print(test_categorical_encoded_data.shape)


print(train_categorical_data_encoded.shape)


missing_in_train = set(test_categorical_encoded_data.columns) - set(train_categorical_data_encoded.columns)
missing_in_test = set(train_categorical_data_encoded.columns) - set(test_categorical_encoded_data.columns)

missing_columns = {
    "missing_in_train": list(missing_in_train),
    "missing_in_test": list(missing_in_test)
}

missing_columns


print(train_quantitative_data.isna().sum())
print(test_quantitative_data.isna().sum())


train_quantitative_data.fillna({'MRI_Track_Age_at_Scan': train_quantitative_data['MRI_Track_Age_at_Scan'].mode()[0]}, inplace=True)
print(train_quantitative_data.isna().sum())


test_data_columns_to_impute = ['SDQ_SDQ_Conduct_Problems', 'SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Generating_Impact', 'SDQ_SDQ_Internalizing', 'SDQ_SDQ_Peer_Problems','APQ_P_APQ_P_CP', 'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_INV', 'APQ_P_APQ_P_OPD', 'APQ_P_APQ_P_PM', 'APQ_P_APQ_P_PP', 'SDQ_SDQ_Difficulties_Total', 'SDQ_SDQ_Externalizing', 'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Prosocial']
for col in test_data_columns_to_impute:
  if test_quantitative_data[col].dtype in [np.float64, np.int64]:
    test_quantitative_data.fillna({col: train_quantitative_data[col].mean()}, inplace=True)
  else:
    print(f"{col} is not a numerical column")


train_merged_data = merge_data(train_fmri_data, train_quantitative_data, train_categorical_encoded_data)
test_merged_data = merge_data(test_fmri_data, test_quantitative_data, test_categorical_encoded_data)

print(train_merged_data.shape)
print(test_merged_data.shape)


# check if the column datatypes
handle_datatypes(train_merged_data, test_merged_data)


# handle datatypes
handle_datatypes(train_merged_data, test_merged_data)


# test if there are any more mismatch datatypes
unmatching_dtypes = {col: (train_merged_data.dtypes[col], test_merged_data.dtypes[col]) for col in train_merged_data.columns if col in test_merged_data.columns and train_merged_data.dtypes[col] != test_merged_data.dtypes[col]}

display(unmatching_dtypes)


# handle the order of the columns in train and test
# check the column order and print the columns don't match the order
unmatching_column_order = {col: (train_merged_data.columns.get_loc(col), test_merged_data.columns.get_loc(col)) for col in train_merged_data.columns if col in test_merged_data.columns and train_merged_data.columns.get_loc(col)!= train_merged_data.columns.get_loc(col)}
display(unmatching_column_order)

