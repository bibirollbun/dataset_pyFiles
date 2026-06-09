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


import kagglehub

# Download latest version
path = kagglehub.dataset_download("kaggle/meta-kaggle")

print("Path to dataset files:", path)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import warnings
warnings.filterwarnings('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df=pd.read_csv(os.path.join("/kaggle/input/meta-kaggle/Datasets.csv"))


df.head()


df.sample(5)


df.tail()


df.shape


df.info()


df.describe()


df.columns


df.isnull().sum()


Total_rows=len(df)
missing_rows=df.isnull().sum()
columns=['Id', 'CreatorUserId', 'OwnerUserId', 'OwnerOrganizationId',
       'CurrentDatasetVersionId', 'CurrentDatasourceVersionId', 'ForumId',
       'Type', 'CreationDate', 'LastActivityDate', 'TotalViews',
       'TotalDownloads', 'TotalVotes', 'TotalKernels', 'Medal',
       'MedalAwardDate']
for i in range(0,len(missing_rows)):
    missing_percentage=(missing_rows[i])*100/(Total_rows)
    for j in range(0,len(columns)):
        if (i==j):
            print("missing_percentage of", columns[j] ,"is:" , missing_percentage)


plt.pie(df["Medal"].value_counts(),labels=df["Medal"].value_counts().index,autopct='%1.1f%%')


df.nunique()


df.duplicated().sum() # duplicate Data


df["LastActivityDate"]=pd.to_datetime(df["LastActivityDate"])
df["MedalAwardDate"]=pd.to_datetime(df["MedalAwardDate"])


df['CreationDate']=pd.to_datetime(df['CreationDate'])
df['Creation_year']=df["CreationDate"] .dt.year # creating new column as Creation_year
df['Creation_month']=df['CreationDate'].dt.month #creating new column as creation_month
df["LastActivity_year"]=df["LastActivityDate"] .dt.year


df.head() # now 19 columns are their


j=0;
for i in range(0,len(df)):
    if(df["Creation_year"][i]==df["LastActivity_year"][i]):
        j=j+1
print(j)
        


sns.countplot(x='Creation_year',data=df)
plt.show()
print(df['Creation_year'].value_counts())
print(df['Creation_year'].nunique()) # data of how many years
plt.pie(df["Creation_year"].value_counts(),labels=df["Creation_year"].value_counts().index,autopct='%1.1f%%')


sns.countplot(x='Creation_month',data=df)
plt.show()
print(df['Creation_month'].value_counts())
print(df['Creation_month'].nunique())
plt.pie(df["Creation_month"].value_counts(),labels=df["Creation_month"].value_counts().index,autopct='%1.1f%%')


TotalDownloads_sum=df.groupby("Creation_year").TotalDownloads.sum()
print(TotalDownloads_sum)
plt.plot(TotalDownloads_sum.index,TotalDownloads_sum)
plt.title("Creation_Year Vs TotalDownloads")
plt.show()


TotalViews_sum=df.groupby("Creation_year").TotalViews.sum()
print(TotalViews_sum)
plt.plot(TotalViews_sum.index,TotalViews_sum)
plt.title("Creation_Year Vs TotalViews")
plt.show()


TotalVotes_sum=df.groupby("Creation_year").TotalVotes.sum()
print(TotalVotes_sum)
plt.plot(TotalVotes_sum.index,TotalVotes_sum)
plt.title("Creation_Year Vs TotalVotes")
plt.show()


TotalKernels_sum=df.groupby("Creation_year").TotalKernels.sum()
print(TotalKernels_sum)
plt.plot(TotalKernels_sum.index,TotalKernels_sum)
plt.title("Creation_Year Vs TotalKernels")
plt.show()


df1=pd.DataFrame({
    "TotalDownloads_sum": TotalDownloads_sum,
    "TotalViews_sum": TotalViews_sum,
    "TotalVotes_sum":TotalVotes_sum,
    "TotalKernels_sum": TotalKernels_sum,
    "Creationyear": TotalDownloads_sum.index
})


df1.head(11)


sns.pairplot(df1,kind="scatter") # scatter plot of Downloads, Views ,Votes and Kernel in year


a=df['TotalDownloads'].max() # number of maximum  downloads for datset 
print(a)
b=df['TotalViews'].max() # max. Views
print(b)
c=df['TotalVotes'].max() # Maximum Votes
print(c)
d=df['TotalKernels'].max() # max. Kernel
print(d)



for i in range(len(df)):
    if (df['TotalDownloads'][i]==a):
        a1=df['Id'][i]
print(a1)


for i in range(len(df)):
    if (df['TotalViews'][i]==b):
        b1=df['Id'][i]
print(b1)        


for i in range(len(df)):
    if (df['TotalVotes'][i]==c):
        c1=df['Id'][i]
print(c1)


for i in range(len(df)):
    if (df['TotalKernels'][i]==d):
        d1=df['Id'][i]
print(d1)


df2=pd.read_csv('/kaggle/input/meta-kaggle/DatasetVersions.csv')
df2.head()


df2.shape


   for i in range(len(df2)):  #  for maximum Downloads
    if (df2['DatasetId'][i]==a1):
        print(df2['Title'][i])


for i in range(len(df2)):  #  for maximum TotalViews
    if (df2['DatasetId'][i]==b1):
        print(df2['Title'][i])


for i in range(len(df2)):  #  for maximum Totalvotes
    if (df2['DatasetId'][i]==c1):
        print(df2['Title'][i])


for i in range(len(df2)):  # for maximum TotalKernels
    if (df2['DatasetId'][i]==d1):
        print(df2['Title'][i])

