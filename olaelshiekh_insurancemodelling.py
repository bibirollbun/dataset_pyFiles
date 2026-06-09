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
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import warnings



# Ignore all warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e12/sample_submission.csv')


train_data


train_data.info()


train_data.isnull().sum()


# missing percentage
missings = train_data.isnull().mean()*100
missings


# handling Age ( mean, mode, median)
sns.boxplot(x=train_data['Age'])
plt.title("BoxPlot of Age")
plt.show()



print(train_data['Age'].mean())
print(train_data['Age'].median())


train_data.describe()


for col in train_data.columns:
    print(train_data[col].value_counts(normalize=True))
    print("-------------")


train_data.info()


del train_data['id']


train_data['Number of Dependents'] = train_data['Number of Dependents'].astype('category')
train_data['Previous Claims'] = train_data['Previous Claims'].astype('category')


train_data.info()


# N


num = train_data.select_dtypes(include=['int64', 'float64']).columns
cat = train_data.select_dtypes(include=['object', 'category']).columns
print(num.tolist())
print(cat.tolist())



"""
numerical_columns = ['Age' ,'Annual Income' ,'' ]
cat 
train_data.fillna(train_data.mean(), inplace=True)
#gender, Marital Status
"""


for col in num:
    plt.figure(figsize=(15,8))
    sns.histplot(x=train_data[col] , kde=True , bins = 25, data =train_data)
    plt.title(f"Histogram of {col}")
    plt.show()


mean = train_data['Health Score'].mean()


std = train_data['Health Score'].std()


train_data['Health Score'].isna().sum()



#np.random.uniform(mean-std ,mean+std , size = train_data['Health Score'].isna().sum() )



train_data.describe()


#test_data.fillna(test_data.mean() , inplace=True )
# missing percentage
for col in num:
    print(f"Column Name : {col}")
    print(f" Number of Missing Before Cleaning {train_data[col].isnull().mean()*100}")
    
    m = train_data[col].mean()
    s = train_data[col].std()
    si = train_data[col].isna().sum()

    #pd.Series
    train_data[col] = train_data[col].fillna(pd.Series(np.random.uniform(m-s , m+s , size = int(si))
                                                      , index = train_data[train_data[col].isna()].index ) )
    
    print(f"Number of missing after Handling : {train_data[col].isnull().mean()*100}")
    print("________")
    
    #loc , iloc


for col in num:
    plt.figure(figsize=(15,8))
    sns.histplot(x=train_data[col] , kde=True , bins = 25 , data= train_data)
    plt.title(f"Histogram of {col}")
    plt.show()


train_data['Marital Status'].mode()[0]


for col in cat:
    print(f"Column Name : {col}")
    print(train_data[col].isnull().mean()*100)
    
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    print(f"Number of missing after Handling : {train_data[col].isnull().mean()*100}")
    print("________")
    
    


num


# supplot
plt.figure(figsize=(20,15))
for i , col in enumerate(num , start=1):
    plt.subplot(5,2 , i)
    sns.boxplot(x=train_data[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()    
plt.show()


for col in num :
    fig = px.histogram( train_data , x = col , title = f'Histogram for {col} distribution')
    fig.show()



num


for col in num :
    print(f"Column Name : {col}")
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - ( 1.5 *IQR )
    upper = Q3 + (1.5 * IQR )
    print (f" Q1 : {Q1} , Q3 : {Q3} \nLower bound is : {lower} \nUpper Bound is : {upper}")
    outlier = train_data[(train_data[col] < lower) | (train_data[col] > upper ) ]
    print(f"Number of outlier : {outlier.shape[0]}")
    print("____________________")


# Define function to detect outlier 
def mad_outliers( column_name , threshold = 3.5 ): # threshold = 3.5 Defult value
    med = train_data[column_name].median()
    abs_deviation = abs ( train_data[column_name] - med )
    mad = abs_deviation.median()
    modified_z_score = 0.6745 * (train_data[column_name] - med ) / mad
    outliers =  train_data[(modified_z_score < -threshold) | (modified_z_score > threshold )]
    print(f"Outlier size in {column_name} is : {outliers.shape[0]}")
    return outliers

for col in num :
    mad_outliers(col )



df = train_data.copy()
df


def mad_outliers( column_name , threshold = 3.5 ): # threshold = 3.5 Defult value
    med = df[column_name].median()
    abs_deviation = abs ( df[column_name] - med )
    mad = abs_deviation.median()
    modified_z_score = 0.6745 * (df[column_name] - med ) / mad
    outliers =  (modified_z_score < -threshold) | (modified_z_score > threshold )
    print(f"Outlier size in {column_name} is : {outliers.shape[0]}")
    return outliers


income_outlier = mad_outliers('Annual Income')
income_outlier


df = df[~income_outlier]
df


cat


from sklearn.preprocessing import LabelEncoder


LE = LabelEncoder()
for col in cat :
    train_data[col] = LE.fit_transform(train_data[col])

train_data


num


from sklearn.preprocessing import StandardScaler


SC = StandardScaler()

for col in num :
    train_data[[col]] = SC.fit_transform(train_data[[col]])

train_data


X = train_data.drop('Premium Amount' , axis = 1)
y = train_data['Premium Amount']


X


y


from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.25 , random_state = 42)


print("X_train : " , X_train.shape)
print("X_test : " , X_test.shape)
print("y_train : " , y_train.shape)
print("y_test : " , y_test.shape)




