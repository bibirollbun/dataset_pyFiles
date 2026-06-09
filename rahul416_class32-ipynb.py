import pandas as pd
import numpy as np


df = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
df.head()


df.columns


df['Target'].value_counts()



import seaborn as sns



a = ['Graduate','Dropout','Enrolled']
df['Target'].replace(a,[0,1,2],inplace=True)


imp_col = ['Course', 'Daytime/evening attendance', 'Previous qualification', 
           'Previous qualification (grade)','Admission grade', 'Displaced', 'Educational special needs', 
           'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment', 
           'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)',
           'Unemployment rate', 'Inflation rate', 'GDP','Target' ]
x = df[imp_col]
y = df['Target']


x


import plotly.express as px
fig = px.scatter(df,x= 'Previous qualification (grade)', y = 'Curricular units 2nd sem (grade)', color='Target')
fig.show()


fig = px.scatter(df,x= 'Curricular units 1st sem (grade)', y = 'Curricular units 2nd sem (grade)', color='Target')
fig.show()


fig = px.scatter(df , x = 'Previous qualification (grade)', y = 'Admission grade',color = 'Target')
fig.show()


s = 1
n = 11
x = 1
for i in range(1,n+1):
    s += i/(x**i)

print(s)



a = 1
r = 9
s = 0
for i in range(0,11):
    s += a*(r**0)

print(s)


n = 10
def factorial(n):
    fact = 1
    for i in range(1,n+1):
        fact *= i
    return fact

s = 0
for i in range(n+1):
    s += 1/factorial(i)

print(s)




