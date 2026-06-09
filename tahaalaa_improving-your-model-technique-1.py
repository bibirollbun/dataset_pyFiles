import numpy as np 
import pandas as pd


df=pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')


df.head()


#check if there null values of this Feature 
df['Number_of_Ads'].isnull().sum()


#there are one row , lets fill correct its data with mean 
df.fillna({'Number_of_Ads': df['Number_of_Ads'].mean()}, inplace=True)


df['Number_of_Ads'].describe()


df['Number_of_Ads'].isnull().sum()


#lets take a look 
df[df['Number_of_Ads'] > 3]


len(df[df['Number_of_Ads'] > 3 ])


mean_ads = df['Number_of_Ads'].mean()
df.loc[df['Number_of_Ads'] > 3, 'Number_of_Ads'] = mean_ads


#now check again 
len(df[df['Number_of_Ads'] > 3 ])


#its cleaned :) 




