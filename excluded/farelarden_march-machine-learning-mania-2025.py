pip install pandas-profiling > /dev/null 2>&1


import pandas as pd
from pandas_profiling import ProfileReport

def analyze_dataset(df, rows_to_show=5):
    """
    Function to display the head of a dataset and generate an EDA report.

    Parameters:
    - file_path (str): Path to the CSV file.
    - rows_to_show (int): Number of rows to display from the dataset's head. Default is 5.

    Returns:
    - None: Displays dataset head and EDA report directly.
    """
    
    # Display the first few rows of the dataset
    display(df.head(rows_to_show))
    
    # Generate an EDA report
    profile = ProfileReport(df, title="Pandas Profiling Report")
    
    # Display the report directly (works in Jupyter Notebooks or similar environments)
    profile.to_widgets()


df_mTeams = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MTeams.csv")
analyze_dataset(df_mTeams)


df_wTeams = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WTeams.csv")
analyze_dataset(df_wTeams)


selected_rows = df_wTeams[df_wTeams['TeamName'].str.contains('St')]

print(selected_rows)


df_mSeasons = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv")
analyze_dataset(df_mSeasons)


df_wSeasons = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WSeasons.csv")
analyze_dataset(df_wSeasons)




