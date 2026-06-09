# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train, test, submission =pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv'),pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv'), pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

train.head()


aggregated_data = train.groupby('Genre').aggregate({
    'Listening_Time_minutes': ['sum', 'mean']
})
aggregated_data.columns = ['Total_Listening_Time', 'Average_Listening_Time']

aggregated_data



# Set the figure size
plt.figure(figsize=(12, 6))

# Plot Total Listening Time.  Access the correct column from the MultiIndex
plt.bar(aggregated_data.index, aggregated_data['Total_Listening_Time'], color='skyblue', label='Total Listening Time')

# Plot Average Listening Time on a secondary y-axis. Access the correct column
plt.twinx()
plt.plot(aggregated_data.index, aggregated_data['Average_Listening_Time'], color='orange', marker='o', label='Average Listening Time')

# Add labels and title
plt.title('Total and Average Listening Time by Genre')
plt.xlabel('Genre')
plt.ylabel('Total Listening Time (millions)')
plt.gca().set_ylabel('Average Listening Time (minutes)', color='orange')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add legends
plt.legend(loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()


aggregated_sentiment = train.groupby('Episode_Sentiment').aggregate({
    'Listening_Time_minutes': ['sum', 'mean']
})
aggregated_sentiment.columns = ['Total_Listening_Time', 'Average_Listening_Time']

aggregated_sentiment


aggregated_daily = train.groupby('Publication_Day').aggregate({
    'Listening_Time_minutes': ['sum', 'mean']
})
aggregated_daily.columns = ['Total_Listening_Time', 'Average_Listening_Time']

aggregated_daily


aggregated_guest_pop = train.groupby('Guest_Popularity_percentage').aggregate({
    'Listening_Time_minutes': ['sum', 'mean']
})
aggregated_guest_pop.columns = ['Total_Listening_Time', 'Average_Listening_Time']

aggregated_guest_pop


aggregated_host_pop = train.groupby('Host_Popularity_percentage').aggregate({
    'Listening_Time_minutes': ['sum', 'mean']
})
aggregated_host_pop.columns = ['Total_Listening_Time', 'Average_Listening_Time']

aggregated_host_pop


print(train.info())
print(train.describe())


plt.boxplot(train['Listening_Time_minutes'])


test.head()


from sklearn.impute import SimpleImputer

# ... (load train and test DataFrames)

# Create a list of columns to impute
columns_to_impute = ['Guest_Popularity_percentage', 'Host_Popularity_percentage']

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')

# Fit on the TRAINING data, selecting ONLY the columns to impute
imputer.fit(train[columns_to_impute])  # Fit on both columns

# Transform BOTH train and test data using the same columns
train[columns_to_impute] = imputer.transform(train[columns_to_impute])
test[columns_to_impute] = imputer.transform(test[columns_to_impute])




from sklearn import linear_model
x_guest_imputed = train['Guest_Popularity_percentage'].values.reshape(-1, 1)  # Reshape to 2D
x_host_imputed = train['Host_Popularity_percentage'].values.reshape(-1, 1) 
y = train['Listening_Time_minutes']
y_gest_imputed = test['Guest_Popularity_percentage'].values.reshape(-1, 1) 
y_host_imputed = test['Host_Popularity_percentage'].values.reshape(-1, 1) 

# Model 1: Guest Popularity vs. Listening Time
reg_guest = linear_model.LinearRegression()
reg_guest.fit(x_guest_imputed, y)
print("Guest Model Coefficients:", reg_guest.coef_)

# Model 2: Host Popularity vs. Listening Time
reg_host = linear_model.LinearRegression()
reg_host.fit(x_host_imputed, y)
print("Host Model Coefficients:", reg_host.coef_)


y_predicted_guest = reg_guest.predict(y_gest_imputed)
y_predicted_host = reg_host.predict(y_host_imputed) 


plt.boxplot(y_predicted_guest)


plt.boxplot(y_predicted_host)


import statsmodels.api as sm


# Let's consider the 'Episode_Length_minutes' column
data = train['Episode_Length_minutes']

# Create an ECDF object
ecdf = sm.distributions.ECDF(data)

# Get the x values for the plot
x = np.linspace(data.min(), data.max(), 100)

# Evaluate the ECDF at these x values
y = ecdf(x)

# Plot the ECDF
plt.figure(figsize=(8, 6))
plt.step(x, y)
plt.title('Empirical Cumulative Distribution Function (ECDF)')
plt.xlabel('Episode Length (minutes)')
plt.ylabel('Cumulative Probability')
plt.show()




