# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')


df.describe()


df['loan_intent'].value_counts()


df['person_home_ownership'].value_counts()


plt.figure(figsize=(15, 10))  
sns.boxplot(x='person_age', y='loan_amnt', data=df, whis=3)
plt.xlabel('Age in years') 
plt.ylabel('Loan Amount') 

plt.show()




ordered_categories = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
df['person_home_ownership'] = pd.Categorical(df['person_home_ownership'], categories=ordered_categories, ordered=True)
df_sorted = df[['person_home_ownership', 'loan_intent']].value_counts().sort_index(ascending=True)
print(df_sorted)



# считаем уникальные сочетания
df_sorted = df[['person_home_ownership', 'loan_intent']].value_counts().sort_index(ascending=True)
df_sorted = df_sorted.reset_index(name='count')
df_sorted.columns = ['person_home_ownership', 'loan_intent', 'count']
print(df_sorted.dtypes)
plt.figure(figsize=(12, 8))
sns.barplot(x='person_home_ownership', y='count', hue='loan_intent', data=df_sorted)
plt.show()



cols = [ 'person_income', 'loan_amnt','loan_grade','person_age']
sns.pairplot(df[cols])
plt.show()


loan_grade_order = ['A', 'B', 'C', 'D', 'E', 'F']
df['loan_grade'] = pd.Categorical(df['loan_grade'], categories=loan_grade_order, ordered=True)
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x='loan_amnt', hue='loan_grade', multiple="stack", kde=True)




