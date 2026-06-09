import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#additional config
pd.set_option('display.max_columns',None)


data = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
data.head()


data.shape


data.dtypes


features = data.columns
numerical_features = data.select_dtypes(exclude='object').columns
categorical_features = data.select_dtypes(include='object').columns
print(f'numerical features: {len(numerical_features)} \ncategorical features: {len(categorical_features)}')


data.isnull().sum()


data['id'].nunique()


na_features = []
for col in features:
    if data[col].isnull().any():
        na_features.append(col)
        print(f'{col:<22} has {np.round(data[col].isnull().mean() * 100,2)} % of NA')
print(f'\nNo of NA features: {len(na_features)}')
print(na_features)
        


data['Age'].nunique()


data['Age'].value_counts().sort_index().plot(kind='bar',figsize=(12,3))


data[data['Age'].isna()]['Gender'].value_counts()


data['Education Level'].value_counts().plot.bar()


#checking whether educational level has any relation
data[data['Occupation'].isna()]['Education Level'].value_counts().plot(kind='pie',autopct='%1.1f%%')


data['Occupation'].value_counts()


data[data['Annual Income'].isna()]['Occupation'].value_counts().plot(kind='pie',autopct="%1.1f%%")


data[data['Annual Income'].isna()]['Education Level'].value_counts().plot(kind='pie',autopct="%1.1f%%")


data['Marital Status'].value_counts().plot(kind='pie',autopct='%1.1f%%')


data[data['Marital Status'].isna()]['Number of Dependents'].value_counts().plot(kind='pie',autopct='%1.1f%%')


data.groupby('Number of Dependents')['Marital Status'].value_counts().plot.bar(figsize=(6,3))


data.columns


data['Smoking Status'].value_counts()


data.groupby('Smoking Status')['Health Score'].median().plot.bar(figsize=(6,3))


#Analysing health score by excercise habit
data['Exercise Frequency'].value_counts()
data[data['Health Score'].isna()]['Exercise Frequency'].value_counts()


data.groupby('Exercise Frequency')['Health Score'].mean()


data[data['Insurance Duration'].isna()]


#checking insurance duration for similar policy start data
data[data['Policy Start Date'].str.contains('2022-04-06')].head()


data[data['Premium Amount'] > data['Annual Income']].shape


for col in numerical_features:
    data[col].plot(kind='hist',bins=20,figsize=(6,3),edgecolor='black')
    plt.title(col)
    plt.show()


for col in categorical_features:
    if col != 'Policy Start Date':
        data[col].value_counts().plot(kind='bar',figsize=(6,3))
        plt.title(col)
        plt.show()


import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


for col in numerical_features:
    if data[col].skew() > 0.5 or data[col].skew() < -0.5:
        print(f'{col} {np.round(data[col].skew(),2)}')


sns.distplot(data['Annual Income'])


sns.boxplot(data['Annual Income'])


IQR = data['Annual Income'].quantile(0.75) - data['Annual Income'].quantile(0.25)
lower_bound = (data['Annual Income'].quantile(0.25)) - (IQR * 1.5)
upper_bound = (data['Annual Income'].quantile(0.75)) + (IQR * 1.5)
print(lower_bound,upper_bound)


data[data['Annual Income'] > 99583].shape


#for extreme outliers
lower_bound = (data['Annual Income'].quantile(0.25)) - (IQR * 3)
upper_bound = (data['Annual Income'].quantile(0.75)) + (IQR * 3)
print(lower_bound,upper_bound)


data[data['Annual Income'] > 99583].shape


sns.distplot(data['Previous Claims'])


sns.boxplot(data['Previous Claims'])


IQR = data['Previous Claims'].quantile(0.75) - data['Previous Claims'].quantile(0.25)
lower_bound = (data['Previous Claims'].quantile(0.25)) - (IQR * 1.5)
upper_bound = (data['Previous Claims'].quantile(0.75)) + (IQR * 1.5)
print(lower_bound,upper_bound)


data[data['Previous Claims'] > 5].shape


sns.distplot(data['Premium Amount'])


sns.boxplot(data['Premium Amount'])


IQR = data['Premium Amount'].quantile(0.75) - data['Premium Amount'].quantile(0.25)
lower_bound = (data['Premium Amount'].quantile(0.25)) - (IQR * 1.5)
upper_bound = (data['Premium Amount'].quantile(0.75)) + (IQR * 1.5)
print(lower_bound,upper_bound)


data[data['Premium Amount'] > 3001.5].shape


for col in categorical_features:
    if col != 'Policy Start Date':
        data[col].value_counts().plot(kind='pie',autopct="%1.1f%%",figsize=(3,3))
        plt.title(col)
        plt.show()

