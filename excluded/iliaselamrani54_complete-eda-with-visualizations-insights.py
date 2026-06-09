

import numpy as np 
import pandas as pd 

nn
import os  
for dirname, _, filenames in os.walk('/kaggle/input'):  
    for filename in filenames:  
        print(os.path.join(dirname, filename))  
import seaborn as sns   
import matplotlib.pyplot as plt 
 
 


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv') 


train.head(5)


train.describe()


train.isna().sum()


train['ethnicity'].hist()


train.groupby(["ethnicity", "diagnosed_diabetes"])['id'].count()


plt.figure(figsize=(12,6))
sns.countplot(
    data=train,
    x="ethnicity",
    hue="diagnosed_diabetes"
)
plt.title("Count of Diabetes vs No Diabetes per Ethnicity")
plt.show()



train.drop(['id','age','diagnosed_diabetes','cardiovascular_history','hypertension_history','family_history_diabetes'],axis=1).hist(bins=50,figsize=(12,12))
plt.show()


train.boxplot(column='physical_activity_minutes_per_week' , by='diagnosed_diabetes',showfliers=False)
plt.show()



train.boxplot(column='diet_score',  by='diagnosed_diabetes',showfliers=False)
plt.show()




train.boxplot(column ='age',by='diagnosed_diabetes')
plt.show()


sns.countplot(x='smoking_status', hue='diagnosed_diabetes', data=train)
plt.xticks(rotation=45)



sns.countplot(x='diagnosed_diabetes', data=train)
plt.show()



prop=train.groupby('family_history_diabetes')['diagnosed_diabetes'].mean()
prop.plot(kind='bar')
plt.ylabel('proportion disgnosed diabetes')
plt.xticks(rotation=0)
plt.show()


prob2 = train.groupby('cardiovascular_history')['diagnosed_diabetes'].mean()
prob2.plot(kind='bar')
plt.ylabel('proportion disgnosed diabetes')
plt.show


from pandas.plotting import scatter_matrix
corr_matrix=train.select_dtypes(include='number').corr()



pd.cut(train['age'],bins=[18,20,40,60])


plt.figure(figsize=(12,10))
numeric = train.select_dtypes(include="number").drop(['id','diagnosed_diabetes','cardiovascular_history','hypertension_history','family_history_diabetes'] , axis=1)
corr = numeric.corr()
sns.heatmap(corr, cmap="coolwarm", center=0,annot=True , fmt='.2f')
plt.show()



sample = train.sample(1100)
sns.scatterplot(data=sample, x='cholesterol_total', y='ldl_cholesterol')
plt.show()


train.columns 


plt.plot(sample['bmi'],sample['waist_to_hip_ratio'],'r.')
plt.xlabel('bmi')
plt.ylabel('waist to hip ratio')
plt.show() 

