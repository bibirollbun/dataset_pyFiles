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


def load_csv_to_dataframe(file_path, ignore_fields=[]):
    """
    Load a CSV file into a pandas DataFrame, optionally ignoring specified fields.

    Parameters:
    file_path (str): The file path of the CSV file to be loaded.
    ignore_fields (list): A list of field names to be ignored when loading the CSV.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the CSV file, excluding the ignored fields.
    """
    # Read the CSV file from the given file path using pandas
    df = pd.read_csv(file_path)
    
    # Drop the fields that need to be ignored, if they exist in the DataFrame
    df = df.drop(columns=ignore_fields, errors='ignore')
    
    # Return the resulting DataFrame
    return df

# Example usage:
# df = load_csv_to_dataframe('data/sample.csv', ignore_fields=['column_to_ignore'])
# print(df.head())


# Load the competiion dataset
MRegularSeasonCompactResults = '/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv'
WRegularSeasonCompactResults = '/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv'
SampleSubmissionStage1 = '/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv' 

#original_input = '/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv'

MRegularSeasonCompactResults_df = load_csv_to_dataframe(MRegularSeasonCompactResults, ignore_fields=['id'])
WRegularSeasonCompactResults_df = load_csv_to_dataframe(WRegularSeasonCompactResults, ignore_fields=['id'])
SampleSubmissionStage1_df = load_csv_to_dataframe(SampleSubmissionStage1)


MRegularSeasonCompactResults_df.info()


MRegularSeasonCompactResults_df.tail()


WRegularSeasonCompactResults_df.info()


WRegularSeasonCompactResults_df.head()


SampleSubmissionStage1_df.info()


SampleSubmissionStage1_df.head()




