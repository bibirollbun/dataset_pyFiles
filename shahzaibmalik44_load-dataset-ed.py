import os
import pandas as pd


def load_and_concatenate_csvs(folder_path):
    # Initialize an empty list to hold DataFrames
    ddf_list = []
    
    # Loop through the files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):  
            file_path = os.path.join(folder_path, file_name)
            print(f"Loading file: {file_path}")
            ddf = pd.read_csv(file_path)  
            ddf_list.append(ddf)  
    
    # Concatenate all DataFrames into one DataFrame
    concatenated_ddf = pd.concat(ddf_list, axis=0, ignore_index=True)
    return concatenated_ddf

# Usage example:
train_folder = "/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data"
concatenated_train_df = load_and_concatenate_csvs(train_folder)

test_folder = "/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data"
concatenated_test_df = load_and_concatenate_csvs(test_folder)



def display_dataset_info(dataset, name):  
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    print("-----------------------------------------------------------------")  
    print(f"{BLUE}{name} DataFrame Shape: Rows = {dataset.shape[0]}, Columns = {dataset.shape[1]}{RESET}")  
    
    # Numerical and categorical columns information  
    num_cols = dataset.select_dtypes(include='number')  
    cat_cols = dataset.select_dtypes(exclude='number')  
    print(f"{BLUE}{name} DataFrame has {len(num_cols.columns)} numeric columns and {len(cat_cols.columns)} categorical columns.{RESET}\n")  
    
    # Missing values information  
    total_missing = dataset.isnull().sum().sum()  
    if total_missing > 0:  
        missing_perc = (total_missing / (dataset.shape[0] * dataset.shape[1])) * 100  
        print(f"{YELLOW}There are a total of {total_missing} missing values in the {name} DataFrame ({missing_perc:.2f}% of all values).{RESET}")  
        print("Missing values per column:")
        # Corrected the printing of missing values per column
        missing_values_per_col = dataset.isnull().sum().sort_values(ascending=False)
        missing_values_per_col_percentage = (missing_values_per_col / dataset.shape[0]) * 100
        missing_df = pd.DataFrame({'Missing Values': missing_values_per_col, 'Percentage': missing_values_per_col_percentage})
        print(f"{RED}{missing_df.head(10)}{RESET}")
    else:  
        print(f"There are no missing values in the {name} DataFrame.")  
    
    # Duplicate rows information  
    total_duplicates = dataset.duplicated().sum()  
    if total_duplicates > 0:  
        print(f"\n{GREEN}There are {total_duplicates} duplicate rows in the {name} DataFrame.{RESET}")  
    else:  
        print(f"There are no duplicate rows in the {name} DataFrame.")  
    
    # Check for column data types  
    print("\nColumn data types:")  
    print(dataset.dtypes.value_counts())  
    
    print(f"{BLUE}-----------------------------------------------------------------{RESET}")  # Dark color for the final line



display_dataset_info(concatenated_train_df, "Training") 
display_dataset_info(concatenated_test_df, "Testing") 


# Function to drop duplicate rows
def drop_duplicates(dataset):
    before_rows = dataset.shape[0]
    dataset_cleaned = dataset.drop_duplicates()
    after_rows = dataset_cleaned.shape[0]
    print(f"Removed {before_rows - after_rows} duplicate rows.")
    return dataset_cleaned


# Drop duplicates
train_cleaned_dataset = drop_duplicates(concatenated_train_df)
test_cleaned_dataset = drop_duplicates(concatenated_test_df)


# List of columns to drop
columns_to_drop = [
    'visit_occurrence_id', 'procedure', 'procedure_datetime_hourly', 'measurement_datetime',
    'observation_concept_id', 'observation_concept_name', 'drug_datetime_hourly', 
    'drug_concept_id', 'route_concept_id', 'Glasgow coma scale', 'Left pupil Diameter Auto',
    'Right pupil Diameter Auto', 'Left pupil Pupillary response', 'Right pupil Pupillary response',
    'visit_start_date', 'birth_datetime', 'device_datetime_hourly', 'device', 
    'age_in_months', 'gender'
]

train_cleaned_dataset = train_cleaned_dataset.drop(columns=columns_to_drop)
test_cleaned_dataset = test_cleaned_dataset.drop(columns=columns_to_drop)

