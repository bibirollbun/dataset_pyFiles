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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the devices dataset
devices = pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/devices_test.csv")

# Exploratory Data Analysis (EDA) on Devices Dataset
print("Devices Dataset Info:")
devices.info()

print("\nFirst Few Rows of Devices Dataset:")
print(devices.head())

print("\nSummary Statistics for Devices Dataset:")
print(devices.describe(include='all'))

# Check for missing values in devices dataset
missing_values = devices.isnull().sum()
print("\nMissing Values in Devices Dataset:\n", missing_values)

# Unique device types
unique_devices = devices['device'].unique()
print(f"\nUnique Devices: {unique_devices}")

# Distribution of device usage
plt.figure(figsize=(12, 6))
device_counts = devices['device'].value_counts()
device_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Device Usage')
plt.xlabel('Device')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Time-based analysis of device usage
devices['device_datetime_hourly'] = pd.to_datetime(devices['device_datetime_hourly'], errors='coerce')
device_usage_by_hour = devices.groupby(devices['device_datetime_hourly'].dt.hour).size()

plt.figure(figsize=(12, 6))
device_usage_by_hour.plot(kind='line', marker='o', color='orange')
plt.title('Device Usage by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Device Usages')
plt.grid()
plt.tight_layout()
plt.show()

# Insights on device usage by patient
usage_per_patient = devices.groupby('person_id').size()
print("\nDevice Usage Per Patient Summary Statistics:")
print(usage_per_patient.describe())

plt.figure(figsize=(12, 6))
usage_per_patient.hist(bins=30, color='purple', edgecolor='black')
plt.title('Histogram of Device Usage Per Patient')
plt.xlabel('Number of Device Usages')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()

# Most frequently used devices
most_used_devices = devices['device'].value_counts().head(5)
print("\nMost Frequently Used Devices:")
print(most_used_devices)

# Least frequently used devices
least_used_devices = devices['device'].value_counts().tail(5)
print("\nLeast Frequently Used Devices:")
print(least_used_devices)

# Devices with missing datetime information
missing_datetime = devices[devices['device_datetime_hourly'].isnull()]
print("\nRecords with Missing Device Datetime Information:")
print(missing_datetime)

# Boxplot of device usage by person_id
plt.figure(figsize=(12, 6))
sns.boxplot(data=usage_per_patient.reset_index(), x=0, color='lightblue')
plt.title('Boxplot of Device Usage Per Patient')
plt.xlabel('Number of Device Usages')
plt.tight_layout()
plt.show()





# Analyze trends in device usage over days
devices['device_date'] = devices['device_datetime_hourly'].dt.date
device_usage_by_day = devices.groupby('device_date').size()

plt.figure(figsize=(12, 6))
device_usage_by_day.plot(kind='line', marker='o', color='green')
plt.title('Device Usage Over Days')
plt.xlabel('Date')
plt.ylabel('Number of Device Usages')
plt.grid()
plt.tight_layout()
plt.show()

# Most common devices used per hour
device_hourly_trends = devices.groupby([devices['device_datetime_hourly'].dt.hour, 'device']).size().unstack(fill_value=0)
most_common_device_per_hour = device_hourly_trends.idxmax(axis=1)
print("\nMost Common Device Used Each Hour:")
print(most_common_device_per_hour)

# Visualize hourly trends of top 3 devices
top_3_devices = devices['device'].value_counts().head(3).index
top_3_device_data = devices[devices['device'].isin(top_3_devices)]
plt.figure(figsize=(12, 6))
sns.countplot(data=top_3_device_data, x=top_3_device_data['device_datetime_hourly'].dt.hour, hue='device')
plt.title('Usage of Top 3 Devices by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Usage Count')
plt.legend(title='Device')
plt.tight_layout()
plt.show()


