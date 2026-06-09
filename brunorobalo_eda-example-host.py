# Load Libraries
import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os




# Define file paths
train_folder = "/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data"
test_folder = "/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data"

# Function to load data from a folder into a dictionary
def load_data_from_folder(folder_path):
    """
    Load data files from a given folder into a dictionary.

    Args:
        folder_path (str): Path to the folder containing the CSV files.
    
    Returns:
        dict: A dictionary where keys are table names (derived from filenames) and values are Dask DataFrames.
    """
    file_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            table_name = file_name.replace("_train.csv", "").replace("_test.csv", "").replace(".csv", "")
            file_path = os.path.join(folder_path, file_name)
            file_dict[table_name] = dd.read_csv(file_path)  # Assign Dask DataFrame directly
    return file_dict





# Load training and testing data
tables_train = load_data_from_folder(train_folder)
tables_test = load_data_from_folder(test_folder)



# Basic Data Inspection

print("\n--- Training Data ---")
for table_name, df in tables_train.items():
    rows, cols = df.shape
    print(f"{table_name}: {rows.compute()} rows and {cols} columns")  # Compute only rows, cols is a constant

print("\n--- Testing Data ---")
for table_name, df in tables_test.items():
    rows, cols = df.shape
    print(f"{table_name}: {rows.compute()} rows and {cols} columns")  # Compute only rows



# Example: Analyze one table in detail (e.g., devices)
train_devices = tables_train["measurement_meds"].compute()
test_devices = tables_test["measurement_meds"].compute()

# Display the first few rows of the training and testing data
print("\nTraining Data - Measurement_meds:")
print(train_devices.head())

print("\nTesting Data - Measurement_meds:")
print(test_devices.head())

# Check for missing values
print("\nMissing Values - Training Data (Measurement_meds):")
print(train_devices.isnull().sum())

print("\nMissing Values - Testing Data (Measurement_meds):")
print(test_devices.isnull().sum())

# Handling Missing Values
train_devices = train_devices.fillna(method='ffill').fillna(method='bfill')
test_devices = test_devices.fillna(method='ffill').fillna(method='bfill')



# Function to visualize numeric columns in all tables
def visualize_numeric_columns(tables, dataset_name):
    """
    Visualize numeric columns in all tables.

    Args:
        tables (dict): Dictionary containing table names and DataFrames.
        dataset_name (str): Name of the dataset (e.g., 'Training' or 'Testing').
    """
    for table_name, df in tables.items():
        print(f"\nVisualizing numeric columns in {table_name} ({dataset_name}):")
        
        # Compute the Dask DataFrame to get a Pandas DataFrame
        df = df.compute()
        
        # Identify numeric columns
        numeric_columns = df.select_dtypes(include='number').columns
        
        if numeric_columns.any():
            print(f"Numeric columns in {table_name}: {list(numeric_columns)}")
            
            # Visualize each numeric column
            for column in numeric_columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[column], kde=True)
                plt.title(f'Distribution of {column} in {table_name} ({dataset_name})')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.show()
        else:
            print(f"No numeric columns found in {table_name}.")




# Visualize numeric columns for all training tables
visualize_numeric_columns(tables_train, "Training")

# Visualize numeric columns for all testing tables
#visualize_numeric_columns(tables_test, "Testing")



# Analyze categorical columns
def analyze_categorical_columns(tables, dataset_name):
    for table_name, df in tables.items():
        print(f"\nAnalyzing categorical columns in {table_name} ({dataset_name}):")
        df = df.compute()
        categorical_columns = df.select_dtypes(include='object').columns
        
        if len(categorical_columns) > 0:
            print(f"Categorical columns in {table_name}: {list(categorical_columns)}")
            
            # Plot distribution for each categorical column
            for column in categorical_columns:
                plt.figure(figsize=(10, 6))
                df[column].value_counts().plot(kind='bar')
                plt.title(f'Distribution of {column} in {table_name} ({dataset_name})')
                plt.xlabel(column)
                plt.ylabel('Count')
                plt.show()
        else:
            print(f"No categorical columns found in {table_name}.")




# Analyze categorical columns for training and testing tables
analyze_categorical_columns(tables_train, "Training")
#analyze_categorical_columns(tables_test, "Testing")



# Function to compute and plot correlation heatmap
def correlation_analysis(tables, dataset_name):
    for table_name, df in tables.items():
        print(f"\nCorrelation analysis for {table_name} ({dataset_name}):")
        df = df.compute()
        numeric_columns = df.select_dtypes(include='number').columns
        
        if len(numeric_columns) > 1:
            corr_matrix = df[numeric_columns].corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title(f'Correlation Heatmap for {table_name} ({dataset_name})')
            plt.show()
        else:
            print(f"Not enough numeric columns in {table_name} for correlation analysis.")




# Perform correlation analysis for training and testing tables
correlation_analysis(tables_train, "Training")
#correlation_analysis(tables_test, "Testing")



# Detect outliers using boxplots
def detect_outliers(tables, dataset_name):
    for table_name, df in tables.items():
        print(f"\nOutlier detection for {table_name} ({dataset_name}):")
        df = df.compute()
        numeric_columns = df.select_dtypes(include='number').columns
        
        for column in numeric_columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[column])
            plt.title(f'Outlier Detection for {column} in {table_name} ({dataset_name})')
            plt.xlabel(column)
            plt.show()

# Detect outliers for training and testing tables
detect_outliers(tables_train, "Training")
#detect_outliers(tables_test, "Testing")


# Visualize missing data
def visualize_missing_data(tables, dataset_name):
    for table_name, df in tables.items():
        print(f"\nMissing data visualization for {table_name} ({dataset_name}):")
        df = df.compute()
        missing_data = df.isnull().sum() / len(df) * 100
        print(f"Missing data percentage in {table_name}:\n{missing_data}")
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title(f'Missing Data Heatmap for {table_name} ({dataset_name})')
        plt.show()

# Visualize missing data for training and testing tables
visualize_missing_data(tables_train, "Training")
#visualize_missing_data(tables_test, "Testing")



# Function to analyze time series for a specific patient in a specific table
def analyze_patient_time_series_single_table(table, table_name, patient_id, time_column, variables_of_interest=None):
    """
    Analyze and visualize how specific variables vary over time for a given patient in a specific table.

    Args:
        table (dask.dataframe): The table to analyze.
        table_name (str): Name of the table being analyzed.
        patient_id (int or str): The ID of the patient to analyze.
        time_column (str): Name of the time-related column in the data.
        variables_of_interest (list): List of variable names to visualize. If None, analyzes all numeric columns.
    """
    print(f"\nAnalyzing time series for patient {patient_id} in table {table_name}...")

    # Compute the DataFrame and filter by patient ID
    df = table.compute()

    if 'person_id' not in df.columns:
        print(f"Skipping {table_name}: 'person_id' column not found.")
        return

    patient_data = df[df['person_id'] == patient_id]

    if time_column not in patient_data.columns:
        print(f"Skipping {table_name}: '{time_column}' column not found.")
        return

    if patient_data.empty:
        print(f"No data found for patient {patient_id} in table {table_name}.")
        return

    # Convert time column to datetime
    patient_data[time_column] = pd.to_datetime(patient_data[time_column])
    patient_data = patient_data.sort_values(by=time_column)

    # If no variables are specified, visualize all numeric columns
    if variables_of_interest is None:
        variables_of_interest = patient_data.select_dtypes(include='number').columns

    for variable in variables_of_interest:
        if variable not in patient_data.columns:
            print(f"Skipping variable '{variable}' as it is not in table {table_name}.")
            continue

        plt.figure(figsize=(12, 6))
        plt.plot(patient_data[time_column], patient_data[variable], marker='o', label=variable)
        plt.title(f'{variable} Over Time for Patient {patient_id} ({table_name})')
        plt.xlabel('Time')
        plt.ylabel(variable)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.show()




# Select the table, patient ID, and variables of interest
table_name = "measurement_meds"
selected_table = tables_train[table_name]  # Replace with the appropriate table from tables_train or tables_test
example_patient_id = 1523648500

"""
measurement_datetime                                    0.000000
Systolic blood pressure                                62.470264
Diastolic blood pressure                               62.492467
Body temperature                                       93.085292
Respiratory rate                                       14.060964
Heart rate                                             12.189552
Measurement of oxygen saturation at periphery           0.034891
Oxygen/Gas total [Pure volume fraction] Inhaled gas    98.337933

"""
variables_to_analyze = ['Respiratory rate', 'Heart rate', 'Measurement of oxygen saturation at periphery',
                        'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure']  # Replace with actual variable names
time_column = 'measurement_datetime'  # Replace with the actual time column name

# Analyze time series for the selected table
analyze_patient_time_series_single_table(selected_table, table_name, example_patient_id, time_column, variables_to_analyze)


