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


import numpy as np
import pandas as pd
import warnings


warnings.filterwarnings("ignore")


df = pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
df


df.isnull().sum()


df.info()


cat = df.select_dtypes("object").columns
cat


for i in cat:
    df[i]= df[i].astype("category")

df.info()


num = df.select_dtypes(["int64" , "float64"]).columns
num


for col in df.columns:
    print(df[col].value_counts())
    print("************************************+**")


import plotly.express as px


cat


num


px.pie(df , names= "Premium Amount" , title= 'Annual Income' )


def null_percentage(col):
    col_nun = df[col].isnull().sum()
    na_per = (col_nun / df.shape[0]) * 100
    return na_per

for i in df.columns:
    if null_percentage(i) > 0 :
        print(f"{i} Null : {null_percentage(i).round(2)} % | {df[i].dtype}")


#df['Age'] = df['Age'].fillna(0)
#df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Marital Status'] = df['Marital Status'].fillna(df['Marital Status'].mode()[0])
#df.dropna(subset = ['Vehicle Age'] , inplace= True)


#px.histogram(df , x = 'Age' , color = 'Premium Amount')


def detect_outlier(col):
    Q1 = np.quantile(col , 0.25)
    Q3 = np.quantile(col , 0.75)
    IQR = Q3 - Q1 
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5*IQR
    k = df[i]
    outliers = k[ (( k < lower ) | (k > upper )  ) ]
    return outliers


outliers_dic = {}
for i in num:
   outliers_dic[i] = detect_outlier(df[i])
   
   print(outliers_dic)
   print("****************************************************************************")    #???????????


for i in num:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    K = df[i]
    outliers = K[ ((K > upper) | (K < lower))]
    print(outliers)
    print("********************************************************")    #?????????




