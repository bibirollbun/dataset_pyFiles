import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')


df = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
df.head()


df.shape


null_columns = df.isnull().sum()
null_percentage = null_columns[null_columns>0]/df.shape[0]*100
null_percentage.map(lambda x : f'{x:.2f}%')


column_types = df.dtypes.astype('str')
column_types


int_cols = column_types[column_types.isin(['float64','int64'])].index.tolist()
cat_cols = column_types[column_types=='object'].index.tolist()
target = 'Listening_Time_minutes'


sns.heatmap(df[int_cols].corr(),annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)


cat_df = df[cat_cols+[target]]
cat_df.head()


unique_day = cat_df['Publication_Day'].unique()
dic = {}
pointer = 0
for day in unique_day:
    dic[day]=pointer
    pointer+=1
dic


print(cat_df['Episode_Sentiment'].unique())
sentiment_df = pd.get_dummies(df['Episode_Sentiment'],prefix='Episode_Sentiment' ,prefix_sep='_',dtype = 'int64')
sentiment_df.head()


print(cat_df['Publication_Time'].unique())
time_dic = {'Morning':0,
            'Afternoon':1,
            'Evening':2,
            'Night':3}


cat_df['Publication_day_encode'] = cat_df['Publication_Day'].map(dic)
cat_df['Pulication_time_encode'] = cat_df['Publication_Time'].map(time_dic)

cat_df = pd.concat([cat_df,sentiment_df],axis=1)
cat_df


cat_df.columns


cat_df[['Listening_Time_minutes',
       'Publication_day_encode', 'Pulication_time_encode',
       'Episode_Sentiment_Negative', 'Episode_Sentiment_Neutral',
       'Episode_Sentiment_Positive']].corr()

