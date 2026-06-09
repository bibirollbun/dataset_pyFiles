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
from datetime import datetime
import statsmodels.api as sm

# Step 1: Load the data
file_path = '/kaggle/input/DontGetKicked/training.csv'  # Enter the path to your CSV file
data = pd.read_csv(file_path)

# Step 2: Data preprocessing
# Convert the purchase date column to datetime format
data['PurchDate'] = pd.to_datetime(data['PurchDate'], format='%m/%d/%Y')

# Determine the reference date (the earliest date in the data)
reference_date = data['PurchDate'].min()

# Convert the date to the number of days since the reference date
data['PurchDateNumeric'] = (data['PurchDate'] - reference_date).dt.days

# Step 3: Define variables for modeling
X = data['PurchDateNumeric']  # Independent variable: purchase date as a numeric value
y = data['IsBadBuy']          # Dependent variable: bad purchase (0 or 1)

# Add a constant column to the independent variables for regression
X = sm.add_constant(X)

# Step 4: Train the logistic regression model
model = sm.Logit(y, X).fit()

# Display the model results
print(model.summary())



import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Step 1: Load the data
# We assume the data is in the file 'data.csv'. Replace with your file path.
data = pd.read_csv('/kaggle/input/carvana-traning/training.csv')

# Step 2: Convert dates to time categories (month and year)
data['PurchDate'] = pd.to_datetime(data['PurchDate'], format='%m/%d/%Y')
data['MonthYear'] = data['PurchDate'].dt.to_period('M')

# Step 3: Create a contingency table
contingency_table = pd.crosstab(data['MonthYear'], data['IsBadBuy'])

# Step 4: Perform the Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Display the test results
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
if p < 0.05:
    print("There is a significant relationship between purchase date and good/bad purchase.")
else:
    print("No significant relationship was observed.")

# Step 5: Visualize the data with a stacked bar chart
contingency_table.plot(kind='bar', stacked=True, color=['green', 'red'])
plt.title('Number of Good and Bad Purchases per Month')
plt.xlabel('Month and Year')
plt.ylabel('Count')
plt.legend(['Good Purchase (0)', 'Bad Purchase (1)'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

