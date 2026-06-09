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


df_sub = pd.read_csv("/kaggle/input/playground-series-s4e1/sample_submission.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s4e1/test.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s4e1/train.csv")


df_sub.head()



df_train.head()


df_test.head()


print("test:",df_test.shape)
print("train:",df_train.shape)


print("test nulls : " , df_test.isnull().sum())


print("train nulls : " , df_train.isnull().sum())


print("test dups:",df_test.duplicated().sum())
print("train dups:",df_train.duplicated().sum())


df_train.info()


df_test.info()


df_train.describe()


df_test.describe()


import matplotlib.pyplot as plt

gender_counts = df_train['Gender'].value_counts()

plt.bar(gender_counts.index, gender_counts.values)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution in Training Dataset')
plt.show()



print(gender_counts)


import matplotlib.pyplot as plt

gender_counts = df_train['Geography'].value_counts()

plt.bar(gender_counts.index, gender_counts.values)
plt.xlabel('country')
plt.ylabel('count')
plt.title('Geography Distribution in Training Dataset')
plt.show()






import matplotlib.pyplot as plt

gender_counts = df_train['Tenure'].value_counts()

plt.bar(gender_counts.index, gender_counts.values)
    
plt.title('Geography Distribution in Training Dataset')
plt.show()



import matplotlib.pyplot as plt

gender_counts = df_train['Exited'].value_counts()

plt.bar(gender_counts.index, gender_counts.values)
    
plt.title('Geography Distribution in Training Dataset')
plt.show()


import matplotlib.pyplot as plt

gender_counts = df_train['HasCrCard'].value_counts()

plt.bar(gender_counts.index, gender_counts.values)
    
plt.title('Geography Distribution in Training Dataset')
plt.show()


import matplotlib.pyplot as plt

# Select only numerical columns
numerical_cols = df_train.select_dtypes(include=['int64', 'float64']).columns

# Plot boxplots for each numerical column
plt.figure(figsize=(12, 6))
df_train[numerical_cols].boxplot()
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.grid(False)
plt.show()



numerical_cols





from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_train['Gender_encoded'] = le.fit_transform(df_train['Gender'])
df_test['Gender_encoded'] = le.fit_transform(df_test['Gender'])


import matplotlib.pyplot as plt

# Select only numerical columns
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Loop through each column and plot a boxplot
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    plt.boxplot(df_train[col].dropna())  # dropna to avoid NaN issues
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.grid(False)
    plt.show()



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_train['Gender_encoded'] = le.fit_transform(df_train['Gender'])
df_test['Gender_encoded'] = le.fit_transform(df_test['Gender'])


df_train.head()


lg = LabelEncoder()
df_train['Geography_encoded'] = le.fit_transform(df_train['Geography'])
df_test['Geography_encoded'] = le.fit_transform(df_test['Geography'])


df_train.head()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_train['EstimatedSalary'] = scaler.fit_transform(df_train[['EstimatedSalary']])
df_test['EstimatedSalary'] = scaler.transform(df_test[['EstimatedSalary']])



scaler2 = StandardScaler()

df_train['Balance'] = scaler2.fit_transform(df_train[['Balance']])
df_test['Balance'] = scaler2.transform(df_test[['Balance']])


df_train.head()


plt.hist(df_train['EstimatedSalary'])


plt.hist(df_train['Balance'])




