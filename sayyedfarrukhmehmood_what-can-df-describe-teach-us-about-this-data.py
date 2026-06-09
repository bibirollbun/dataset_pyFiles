%load_ext cudf.pandas


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv',index_col='id')
df.head()


summary_stats=df.describe()
summary_stats


age_stats=summary_stats.Age
age_stats


sns.histplot(df['Age'],kde=True,bins=30)
plt.title("Distribution of Age")
plt.show()


sns.boxplot(x=df['Age'])
plt.title('Age Percentiles')
plt.show()


height_stats=summary_stats.Height
height_stats


sns.histplot(df['Height'],kde=True,bins=30)
plt.title("Distribution of Height")
plt.show()


sns.boxplot(x=df['Height'])
plt.title('Height Percentiles')
plt.show()


weight_stats=summary_stats['Weight']
weight_stats


sns.histplot(df['Weight'],kde=True,bins=30)
plt.title("Distribution of Weight")
plt.show()


sns.boxplot(x=df['Weight'])
plt.title('Weight Percentiles')
plt.show()


duration_stats=summary_stats['Duration']
duration_stats


sns.histplot(df['Duration'],kde=True,bins=10)
plt.title("Distribution of Duration")
plt.show()


sns.boxplot(x=df['Duration'])
plt.title('Duration Percentiles')
plt.show()


heart_rate_stats=summary_stats['Heart_Rate']
heart_rate_stats


sns.histplot(df['Heart_Rate'],kde=True,bins=30)
plt.title("Distribution of Heart Rate")
plt.show()


sns.boxplot(x=df['Heart_Rate'])
plt.title('Heart Rate Percentiles')
plt.show()


temp_stats=summary_stats['Body_Temp']
temp_stats


sns.histplot(df['Body_Temp'],kde=True, bins=30)
plt.title('Body Temperature Distribution')
plt.show()


sns.boxplot(x=df['Body_Temp'])
plt.title('Body Temperature Percentiles')
plt.show()


cal_stats=summary_stats['Calories']
cal_stats


sns.histplot(df['Calories'],kde=True, bins=30)
plt.title('Calories Distribution')
plt.show()


sns.boxplot(x=df['Calories'])
plt.title('Calories Percentiles')
plt.show()

