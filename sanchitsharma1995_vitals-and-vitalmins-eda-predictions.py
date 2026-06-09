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


# importing data 
data=pd.read_csv('/kaggle/input/kaggle-community-olympiad-vitals-variables-predicting-patient-outcomes/train.csv')


#  run the data 
data


# get the information of data 
data.info()



# check the number of duplicates in data 
data.duplicated().sum()


#  check the sum of not null values in a data 
data.isna().sum()


#  getting top 5 rows 
data.head(5)


# checking univariate...

print("Mean age : ",data.age.mean())
print("Median age : ",data.age.median())
print("Age Standard Deviations : ",data.age.std())


data[(data.blood_pressure_systolic<90) |(data.blood_pressure_systolic>179)]


data[data.blood_pressure_diastolic==data.blood_pressure_diastolic.max()].nunique() 


data[data.blood_pressure_diastolic==data.blood_pressure_diastolic.min()]


data[data.blood_pressure_diastolic==data.blood_pressure_diastolic.max()]


#  gender column analysis 

data['gender'].value_counts()


#  finding the average age of men and femle : 
print("the Average Age of men is : ",data[data.gender=='Male'].age.mean())
print("the Average Age of female is : : ",data[data.gender=='Female'].age.mean())


 # Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')



sns.kdeplot(x=data.age, hue=data.gender, fill=True, palette='Set2')
plt.title("Age Distribution by Gender")
plt.xlabel("Age")
plt.ylabel("Density")
plt.show()


 ## Highest and Lowest Age

print('Highest Age : ',data.age.max())
print('Lowest Age : ',data.age.min())


# Rename Columns with high Bp and low BP

data.rename(columns={'blood_pressure_systolic':'High_BP','blood_pressure_diastolic':'Low_BP'},inplace=True)


# run the data
data


# Finding Critical patients rows  that bloop presuure came between 120 and 80 Low and High BP.

data[(data.High_BP>120) & (data.Low_BP<80)]


data[data.High_BP>120]  
# #  ther are  9940 patients which have High BP


data[data.Low_BP<80]   
#  ther are  9940 patients which have LOW BP


# now this we are checking those patients how's BP are laying in betweeen 80 and 102\

data[(data.High_BP<120) & (data.Low_BP>80)]
# so there are 6579 patients with normal BP


# checking the Patient's Normal Heart Rate...

data[(data.heart_rate>60) & (data.heart_rate<100)]
# there are 16755 patients with normal range of heart rate 


# distribution  of heart rate
sns.displot(data,x='heart_rate',hue='gender')



data


# checking maximum and minimum value of respiratory rate 
print(data['respiratory_rate'].min())
print(data['respiratory_rate'].max())


# checking average respiratory rate 
print(data['respiratory_rate'].mean())


#  checking the number patients who's respiratory rate is in between 12 and 18 

data[(data.respiratory_rate>12) & (data.respiratory_rate<18)]
# there are 8372 paitents hows respiratory rate is normal and laying in betweeen 12 and 18



data[data.respiratory_rate>20]   
#  ther are  14934 patients which have  high respiratory_rate


data[data.respiratory_rate<12]   
#  no have has low respiratory_rate


# distribution  of respiratory  rate
sns.displot(data,x='respiratory_rate',hue='gender')


data


# checking maximum and minimum value of body_temperature
print(data['body_temperature'].min())
print(data['body_temperature'].max())
# Normal Body Temperature of the Human Body | DrSafeHandsThe normal body temperature range is
# generally considered to be between 97°F (36.1°C) and 99°F (37.2°C)


#  checking the number patients who's body_temperature is in between 97 and 99 

data[(data.body_temperature>97) & (data.body_temperature<98)]
# there are 5854 paitents hows body_temperature is normal and laying in betweeen 97 and 99 degree



data[data.body_temperature>99]   
#  ther are  9937 patients which have  high body_temperature



data[data.body_temperature<97]   
#  ther are  1456 patients which have  low body_temperature


# distribution  of body_temperature
sns.displot(data,x='body_temperature',hue='gender')


data



# checking maximum and minimum value of oxygen_saturation
print(data['oxygen_saturation'].min())

print(data['oxygen_saturation'].max())


#A normal oxygen saturation level, measured with a pulse oximeter, typically falls between 95% and 100%
# checking the number patients who's oxygen_saturation is in between 95% and 100%

data[(data.oxygen_saturation>95) & (data.oxygen_saturation<100)]
# there are 8034 paitents hows body_temperature is normal and laying in betweeen 95% and 100%



data[data.oxygen_saturation>100]   
#  ther are  zero patients which have  high oxygen_saturation


data[data.oxygen_saturation<95]   
#  there are 19936 patients having low range of oxygen_saturation value 


# distribution  of oxygen_saturation
sns.displot(data,x='oxygen_saturation',hue='gender')
  


data


# checking maximum and minimum value of glucose_fasting
print(data['glucose_fasting'].min())

print(data['glucose_fasting'].max())


# What Is The Normal Range Of Fasting Blood Sugar Level ...A normal fasting blood glucose level is considered to be between 70 and 100 mg/dL (3.9 and 5.6 mmol/L). 

data[(data.glucose_fasting>70) & (data.glucose_fasting<100)]

# there are 79021 paitents hows glucose_fasting is normal and laying in betweeen 70 and 100



data[data.glucose_fasting>100]   
#  ther are  21565 patients which have  high glucose_fasting


data[data.glucose_fasting<70]   
#  there are zero patients having low range of glucose_fasting


# distribution  of glucose_fasting
sns.displot(data,x='glucose_fasting',hue='gender')


# checking maximum and minimum value of cholesterol_total
print(data['cholesterol_total'].min())

print(data['cholesterol_total'].max())


# A total cholesterol level of less than 200 mg/dL is considered normal for adults

data[data.cholesterol_total<200]

# there are 10023 paitents hows cholesterol_total is normal.



data[data.cholesterol_total>200]   
#  ther are  19769 patients which have  high cholesterol_total


 # distribution graph of   cholesterol_total
sns.displot(data,x='cholesterol_total',hue='gender')


data
































data['pulse_pressure'] = data['blood_pressure_systolic'] - data['blood_pressure_diastolic']


data


# Mean Arterial Pressure (MAP)
# MAP = DBP + (1/3) * (SBP - DBP)
# PP=SBP-DBP
# MAP = DBP + (1/3) * (PP)
data['map'] = data['blood_pressure_diastolic'] + (data['pulse_pressure'] / 3)


data


data
















