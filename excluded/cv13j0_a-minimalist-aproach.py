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


# Importing the necessary libraries

import pandas as pd  # Powerful data manipulation and analysis library; offers DataFrame and Series data structures
import numpy as np  # Fundamental package for numerical computing in Python; provides N-dimensional array objects and routines
import random  # Python’s built-in module for generating pseudo-random numbers and selecting random elements
import warnings  # Provides a way to control the display of warning messages (e.g., filter out deprecation warnings)

from sklearn.preprocessing import LabelEncoder  # Utility from scikit-learn to convert categorical labels into numeric form (e.g., “red” → 0, “blue” → 1)
from IPython.display import display  # for nicer display in notebooks


def configure_notebook(seed=548, float_precision=3, max_columns=15, max_rows=25):
    """
    Configure notebook settings:
      - Disables warnings for cleaner output.
      - Sets pandas display options for better table formatting.
      - Returns a seed value for reproducibility.
    
    Parameters:
      seed (int): Random seed (default 548).
      float_precision (int): Number of decimal places for floats (default 3).
      max_columns (int): Maximum number of columns to display (default 15).
      max_rows (int): Maximum number of rows to display (default 25).

    Returns:
      int: The provided seed.
    """
    # Disable all warnings
    warnings.filterwarnings("ignore")
    
    # Set pandas display options for nicer output
    pd.options.display.float_format = f"{{:,.{float_precision}f}}".format
    pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.max_rows", max_rows)

    # Set seeds for reproducibility in numpy and the standard random module
    np.random.seed(seed)
    random.seed(seed)
    
    return seed

# Apply configuration and set random seeds for reproducibility
seed = configure_notebook()


def load_csv_to_dataframe(file_path, ignore_fields=[], delimiter = ','):
    """
    Load a CSV file into a pandas DataFrame, optionally ignoring specified fields.

    Parameters:
    file_path (str): The file path of the CSV file to be loaded.
    ignore_fields (list): A list of field names to be ignored when loading the CSV.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the CSV file, excluding the ignored fields.
    """
    # Read the CSV file from the given file path using pandas
    df = pd.read_csv(file_path, delimiter = delimiter)
    
    # Drop the fields that need to be ignored, if they exist in the DataFrame
    df = df.drop(columns=ignore_fields, errors='ignore')
    
    # Return the resulting DataFrame
    return df


trn_file_path = "/kaggle/input/playground-series-s5e10/train.csv"  # Replace with your CSV file path
trn_df = load_csv_to_dataframe(trn_file_path, ignore_fields=['id'])

test_file_path = "/kaggle/input/playground-series-s5e10/test.csv"  # Replace with your CSV file path
tst_df = load_csv_to_dataframe(test_file_path, ignore_fields=['id'])

sample_file_path = "/kaggle/input/playground-series-s5e10/sample_submission.csv"  # Replace with your CSV file path
sub_df = load_csv_to_dataframe(sample_file_path)


# orig_file_path = "/kaggle/input/bank-marketing-dataset-full/bank-full.csv"  # Replace with your CSV file path
# org_df = load_csv_to_dataframe(orig_file_path, ignore_fields=['User_ID'], delimiter = ";")
# org_df['y'] = org_df.y.map({'yes':1,'no':0})
# #org_df['id'] = (np.arange(len(org_df))+1e6).astype('int')
# #org_df = org_df.set_index('id')

# trn_df = pd.concat([trn_df, org_df], axis=0, ignore_index=True)


def eda_summary(df):
    # 1. Display the first few rows
    print("======== First 5 Rows ========")
    display(df.head().T)
    
    # 2. DataFrame information (data types, non-null counts, etc.)
    print("\n======== DataFrame Info ========")
    df.info()
    
    # 3. Descriptive statistics for numeric columns
    print("\n======== Descriptive Statistics (Numeric Columns) ========")
    display(df.describe())
    
    # 4. Descriptive statistics for categorical columns (if any)
    categorical_df = df.select_dtypes(include=['object', 'category'])
    print("\n======== Descriptive Statistics (Categorical Columns) ========")
    if not categorical_df.empty:
        display(categorical_df.describe())
    else:
        print("No categorical columns found.")
    
    # 5. Missing values summary
    print("\n======== Missing Values Summary ========")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_summary = pd.DataFrame({
        "Missing Count": missing,
        "Percentage": missing_percent
    })
    display(missing_summary)
    
    # 6. Count of duplicated rows
    print("\n======== Duplicated Rows ========")
    print(f"Total duplicated rows: {df.duplicated().sum()}")
    
    # 7. Count of each data type
    print("\n======== Data Types Count ========")
    display(df.dtypes.value_counts())
    
    # 8. Correlation matrix for numeric variables (if more than one exists)
    numeric_cols = df.select_dtypes(include=[np.number])
    if numeric_cols.shape[1] > 1:
        print("\n======== Correlation Matrix (Numeric Columns) ========")
        display(numeric_cols.corr())
    else:
        print("\n======== Correlation Matrix ========")
        print("Not enough numeric columns to compute correlation.")
    
    # 9. Value counts for categorical variables with low cardinality
    print("\n======== Value Counts for Categorical Columns (Low Cardinality) ========")
    if not categorical_df.empty:
        for col in categorical_df.columns:
            if df[col].nunique() <= 20:
                print(f"\nValue Counts for '{col}':")
                display(df[col].value_counts())
    else:
        print("No categorical columns found.")


eda_summary(trn_df)




