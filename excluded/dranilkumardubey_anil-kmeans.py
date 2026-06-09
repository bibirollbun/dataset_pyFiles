# Importing libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('/kaggle/input/k-means-clustering-for-heart-disease-analysis/heart_disease.csv')


df


df.head()


df.tail()


df.info()


plt.hist(df['age'])


df.duplicated()


df.duplicated().sum()


df.describe(include='all')


# Fill the zeros with Nan values
df.hist(figsize = (10,10), color="#076543")


df.isnull().sum()


# replace NaN values for the columns in accordance with their distribution
df['trestbps'].fillna(df['trestbps'].mean(), inplace = True)
df['chol'].fillna(df['chol'].mean(), inplace = True)
# df['fbs'].fillna(df['fbs'].median(), inplace = True)
# df['restecg'].fillna(df['restecg'].median(), inplace = True)
df['thalch'].fillna(df['thalch'].median(), inplace = True)
# df['exang'].fillna(df['exang'].median(), inplace = True)
df['oldpeak'].fillna(df['oldpeak'].median(), inplace = True)
# df['slope'].fillna(df['slope'].median(), inplace = True)
df['ca'].fillna(df['ca'].median(), inplace = True)
# df['thal'].fillna(df['thal'].median(), inplace = True)


df.isnull().sum()


# columns gives column names of features
df.columns


# shape gives number of rows and columns in a tuble
df.shape


#The dtypes property is used to find the dtypes in the DataFrame.

df.dtypes


# Plotting all data
df1 = df.loc[:,['id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs',
       'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
df1.plot()


# subplots
df1.plot(subplots = True)
plt.show()




