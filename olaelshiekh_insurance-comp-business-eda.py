import numpy as np 
import pandas as pd 
import warnings
import plotly.express as px


warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
df


df.isnull().sum()


df.info()


count_age_na = df['Age'].isnull().sum() # Ammar
count_age_na


( count_age_na/1200000 ) * 100 


df.shape[0]


for i in df.columns :
    col_nun = df[i].isnull().sum() 
    na_per = ( col_nun / df.shape[0] ) * 100 

    print(f"missing in {i} is : {na_per.round(2)} % ")


df['Policy Start Date'] = df['Policy Start Date'].astype('datetime64[ns]')
df.info()


cat = df.select_dtypes('object').columns
cat # Hannan 


for i in cat:
    df[i] = df[i].astype("category")

df.info()


for i in df.columns :
    print(df[i].value_counts() )
    print("___________________")


#df['Number of Dependents'] = df['Number of Dependents'].astype('category')
df['Previous Claims'] = df['Previous Claims'].astype('category')

df.info()


df = df.drop('id' , axis=1)
df


num = df.select_dtypes(['float64' , 'int64']).columns
num


cat


def null_percentage(col) : 
    col_nun = df[col].isnull().sum() 
    na_per = ( col_nun / df.shape[0] ) * 100 
   
    return na_per

for i in df.columns :
    if null_percentage(i)> 0 :        
        print(f"{i} Null : {null_percentage(i).round(2)} %  | {df[i].dtype}")
        


df.describe().T


#df['Age'] = df['Age'].fillna(0)
#df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Marital Status'] = df['Marital Status'].fillna(df['Marital Status'].mode()[0])
#df.dropna(subset = ['Vehicle Age'] , inplace= True)



df['Annual Income'] = df['Annual Income'].fillna(df['Annual Income'].mean())


df['Number of Dependents'] = df['Number of Dependents'].fillna(-1.0)


df['Number of Dependents'].value_counts()


null_percentage('Number of Dependents')


df['Number of Dependents'] = df['Number of Dependents'].astype('category')
df['Number of Dependents'].value_counts()


df.info()





df['Occupation'] = df['Occupation'].astype(object).fillna("Unknown")


df['Occupation'].dtype


df.info()


df['Occupation'] = df['Occupation'].astype('category')


df.info()


import matplotlib.pyplot as plt

plt.hist(
    df['Health Score'],
    bins=30,                 # Plotly default-ish
    color='#636EFA',         # Plotly default blue
    edgecolor='white'
)

plt.xlabel('Health Score')
plt.ylabel('Count')
plt.title('Health Score Distribution')

plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()



col = df['Health Score']


col = col.fillna(col.mean())



plt.hist(
    col,
    bins=30,                 # Plotly default-ish
    color='#636EFA',         # Plotly default blue
    edgecolor='white'
)

plt.xlabel('Health Score')
plt.ylabel('Count')
plt.title('Health Score Distribution')

plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()



m = df['Health Score'].mean()
std = df['Health Score'].std()
print(m , std)


np.random.uniform(m-std , m+std)


mask = df['Health Score'].isna()


mask


col = df['Health Score']
col


#df['Health Score'] = df['Health Score'].fillna(np.random.uniform(m-std , m+std))


#df.loc[mask , col] = np.random.uniform(low = m-std , high = m+std , size=mask.sum())




