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


# Import data
train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


# Clean code
def cleanCode(df):
    # Imputing median values for numeric columns
    df['Time_spent_Alone'] = df['Time_spent_Alone'].fillna(df['Time_spent_Alone'].median())
    df['Social_event_attendance'] = df['Social_event_attendance'].fillna(df['Social_event_attendance'].median())
    df['Going_outside'] = df['Going_outside'].fillna(df['Going_outside'].median())
    df['Post_frequency'] = df['Post_frequency'].fillna(df['Post_frequency'].median())

    # Replace NaN data for Drained_after_socializing based on Going_outside value
    # Using 0, 1, or 2 hours a day outside as qualifying someone as probably being drained after socializing
    df['Drained_after_socializing'] = df.apply(
    lambda row: 'Yes' if pd.isna(row['Drained_after_socializing']) and row['Going_outside'] <= 4
    else ('No' if pd.isna(row['Drained_after_socializing']) and row['Going_outside'] > 4
          else row['Drained_after_socializing']),
    axis=1
    )




cleanCode(test_data)
cleanCode(train_data)


# Encoding 'Drained_after_socializing' as 1 (Yes) & 0 (No) for model building
train_data['Drained_after_socializing'].replace({'Yes': 1, 'No': 0}, inplace=True)
test_data['Drained_after_socializing'].replace({'Yes': 1, 'No': 0}, inplace=True)


# Model - Random Forest
from sklearn.ensemble import RandomForestClassifier

# Using the following columns for the model
y = train_data["Personality"]
features = ["Time_spent_Alone",
            "Social_event_attendance",
            "Going_outside",
            "Drained_after_socializing"]
X = train_data[features]
X_test = test_data[features]

# Make and fit the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=49)
model.fit(X,y)

# Produce predictions
predictions = model.predict(X_test)

# Produce dataframe for submission
ID = test_data.id
submission = pd.DataFrame({'id': ID, "Personality": predictions})


# Export submission
submission.to_csv('submission.csv', index=False)
print("yep")

