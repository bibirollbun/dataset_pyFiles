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


#required library
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("/kaggle/input/sf-crime/train.csv.zip")
df.head()


df.describe()


df.shape


# drop few column, keep only dates, category, x,y

df = df.drop(['PdDistrict', 'Address', 'Resolution', 'Descript', 'DayOfWeek'], axis = 1) 


df.head()


# to show bottom 5 rows from dataframe
df.tail()


# to check missing value in dataframe
df.isnull().sum()


# filter only dates from date column'
f = lambda x: (x["Dates"].split())[0] 


# apply function 'f' to split time from date
df["Dates"] = df.apply(f, axis=1)
df.head()


# split year, day, month from date column
f = lambda x: (x["Dates"].split('-'))[0] 
df["Dates"] = df.apply(f, axis=1)
df.head()


df.tail()


# select only year 2014 cases
df_2014 =  df[(df.Dates == '2014')]
df_2014.head()


# how rows for year 2024
df_2014.shape


# scaling the value of X and Y
scaler = MinMaxScaler()

scaler.fit(df_2014[['X']])

df_2014['X_scaled'] = scaler.transform(df_2014[['X']]) 

scaler.fit(df_2014[['Y']])
df_2014['Y_scaled'] = scaler.transform(df_2014[['Y']])


df_2014.head()


# applying kmeans cluatering for k=5
model = KMeans(n_clusters=5)


# predict cluster for x and y
y_predicted = model.fit_predict(df_2014[['X_scaled','Y_scaled']]) 
y_predicted


# assign predicted cluster number , and add in the dataframe
df_2014['cluster'] = y_predicted
df_2014


# plot cluster for cluster label 0
import matplotlib.pyplot as plt
 
#filter rows of original data
cluster_0 = df_2014[df_2014['cluster'] == 0]


cluster_0.head()


#plotting the results

plt.scatter(cluster_0.iloc[:, 4], cluster_0.iloc[:, 5])
plt.show()


#filter rows of original data frame 

cluster_1 = df_2014[df_2014['cluster'] == 1]
cluster_2 = df_2014[df_2014['cluster'] == 2]
cluster_3 = df_2014[df_2014['cluster'] == 3]
cluster_4 = df_2014[df_2014['cluster'] == 4]
 
#scatter plot for cluster 0, cluster 1 and cluster 2
plt.scatter(cluster_0.iloc[:, 4], cluster_0.iloc[:, 5] , color = 'green')
plt.scatter(cluster_1.iloc[:, 4], cluster_1.iloc[:, 5] , color = 'red')
plt.scatter(cluster_2.iloc[:, 4], cluster_2.iloc[:, 5] , color = 'blue')
plt.scatter(cluster_3.iloc[:, 4], cluster_3.iloc[:, 5] , color = 'yellow')
plt.scatter(cluster_4.iloc[:, 4], cluster_4.iloc[:, 5] , color = 'cyan')
plt.show()

