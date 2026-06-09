import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


df_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
df_demo.head()


df_demo.shape


df_demo.describe()


df_demo.dtypes


df_demo.columns


df_demo.info()


df_demo[['adult_child' , 'sex' , 'handedness']] = df_demo[['adult_child' , 'sex' , 'handedness']].astype('object')
#df_demo[['shoulder_to_wrist_cm']] = df_demo[['shoulder_to_wrist_cm']].astype('float64')


df_demo.info()


df_demo.isna().sum()


df_demo.describe()


df_demo.head()


df_demo= pd.get_dummies(df_demo , columns = ['adult_child' , 'sex' , 'handedness'] , dtype=np.float64 , drop_first=True)


df_demo.head()


df_demo.rename(columns={'sex_1': 'sex_male' , 'adult_child_1': 'adult_child' , 'handedness_1': 'handedness_right'} , inplace = True)
df_demo


df_demo.info()


df_demo.mean(numeric_only=True)


df_demo.select_dtypes(include=['number']).median()


fig , axes = plt.subplots(3 , 3 , figsize = (12,12))
axes_flat = axes.flatten()

for i , col in enumerate(df_demo.columns[1:]):
    ax = axes_flat[i]
    ax.hist(df_demo[col])
    ax.set_title(col)


fig , axes = plt.subplots(2 , 3 , figsize = (12,12))
axes_flat = axes.flatten() 

for i , col in enumerate(df_demo.select_dtypes('float64')):
    ax = axes_flat[i]
    ax.boxplot(df_demo[col])
    ax.set_title(col)

