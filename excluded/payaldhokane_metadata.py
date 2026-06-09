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


import os

# List everything in the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(f"Directory: {dirname}")
    for filename in filenames:
        print(f"  - {filename}")


df = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')


import pandas as pd

# Load the competitions data
competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')

# Show the first 5 rows
competitions.head()


# Convert the 'EnabledDate' column to datetime
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'])

# Extract year
competitions['Year'] = competitions['EnabledDate'].dt.year

# Count number of competitions each year
yearly_counts = competitions['Year'].value_counts().sort_index()

# Plot the result
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
yearly_counts.plot(kind='bar', color='skyblue')
plt.title('Kaggle Competitions by Year')
plt.xlabel('Year')
plt.ylabel('Number of Competitions')
plt.grid(True)
plt.show()


competitions = pd.read_csv("/kaggle/input/meta-kaggle/Competitions.csv")
users = pd.read_csv("/kaggle/input/meta-kaggle/Users.csv")

competitions.head()
users.head()


print("Total Competitions:", competitions.shape[0])
print("Total Users:", users.shape[0])

