import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import zipfile

!pip install autogluon
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.metrics import make_scorer

from sklearn.preprocessing import LabelEncoder

# !pip install holidays
import holidays

import zipfile


def is_us_holiday(month, day, year=2018):
    """Check if a given date is a holiday in the United States."""
    us_holidays = holidays.US(years=year)
    return (month, day) in [(d.month, d.day) for d in us_holidays]

def process_flight_data_with_holidays(file_path, is_train=True):
    """
    Processes flight delay data by:
    - Converting Month, DayofMonth, and DayOfWeek into proper date components
    - Extracting deptime_hour and deptime_min from DepTime
    - Encoding UniqueCarrier and Origin into numerical values
    - Handling the dep_delayed_15min target column
    - Adding a column to indicate if the flight is on a US holiday
    """
    # Read the CSV from ZIP file
    with zipfile.ZipFile(file_path, 'r') as z:
        csv_filename = z.namelist()[0]  # Get the CSV file name inside the ZIP
        with z.open(csv_filename) as f:
            df = pd.read_csv(f)    

    # Convert categorical month and day to integers
    df['Month'] = df['Month'].str.replace("c-", "").astype(int)
    df['DayofMonth'] = df['DayofMonth'].str.replace("c-", "").astype(int)
    df['DayOfWeek'] = df['DayOfWeek'].str.replace("c-", "").astype(int)

    # Create proper date components
    df['DepTime'] = df['DepTime'].astype(int)
    
    # Extract hour and minutes from DepTime (e.g., 1934 -> 19 hours, 34 minutes)
    df['deptime_hour'] = df['DepTime'] // 100
    # df['deptime_min'] = df['DepTime'] % 100

    # # Encode UniqueCarrier and Origin into numeric values
    # for col in ['UniqueCarrier', 'Origin', 'Dest']:
    #     le = LabelEncoder()
    #     df[col] = le.fit_transform(df[col])

    # Handle the target column
    if is_train:
        df['dep_delayed_15min'] = df['dep_delayed_15min'].map({'Y': 1, 'N': 0}).astype(float)
    else:
        df['dep_delayed_15min'] = np.nan  # Ensure test data has the column but filled with NaN

    # Drop the original DepTime column (since we extracted hour & min)
    df.drop(columns=['DepTime'], inplace=True)

    # Add a holiday indicator column
    df['is_holiday'] = df.apply(lambda row: is_us_holiday(row['Month'], row['DayofMonth']), axis=1)

    return df

# Process the datasets
train_df = process_flight_data_with_holidays("/kaggle/input/flight-delays-fall-2018/flight_delays_train.csv.zip")
test_df = process_flight_data_with_holidays("/kaggle/input/flight-delays-fall-2018/flight_delays_test.csv.zip", False)
train_df.head(5)


# Define the predictor with ROC AUC as the evaluation metric
predictor = TabularPredictor(label='dep_delayed_15min', eval_metric="roc_auc").fit(
    train_df,
    time_limit=3600,
)


test_data = TabularDataset(test_df)
# test_pred = predictor.predict(test_data)
test_pred = predictor.predict_proba(test_data)
# test_pred = np.round(np.expm1(test_pred))

submission = pd.DataFrame()
submission['id'] = range(0, len(test_df))  # Creates an ID column starting from 1
submission['dep_delayed_15min'] = test_pred[1]
submission.to_csv('submission.csv',index=False)
print('Done producing submission.csv')


test_submission = pd.read_csv('submission.csv')
test_submission

