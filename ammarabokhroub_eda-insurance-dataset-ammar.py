import pandas as pd
import random
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , MinMaxScaler, StandardScaler


warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv') # Loading the insurance dataset into a DataFrame



df


df.info() # Checking for null values and data types


df.drop(columns=['id'], inplace=True)  # Dropping 'id' column as it is not useful for analysis
df.info()


df['Policy Start Date'] = df['Policy Start Date'].astype('datetime64[ns]')
df.info()


cat = df.select_dtypes(include='object').columns # Selecting categorical columns
num = df.select_dtypes(include=['int64', 'float64']).columns # Selecting numerical columns
print("Categorical :", cat)
print("Numerical:", num) 


for i in cat: # Converting categorical columns to 'category' data type
    df[i] = df[i].astype('category')
df.info()


num


#  Checking unique values in some columns . if it likes a categorical column or not
print(df['Health Score'].value_counts()) 
""" Checking unique values in 'Health Score' column .
df['Health Score'] = df['Health Score'].astype('category') 
it has a wide range of values """
print(df['Credit Score'].value_counts()) 
""" Checking unique values in 'Credit Score' column .
#df['Credit Score'] = df['Credit Score'].astype('category') it has a wide range of values """

print(df['Number of Dependents'].value_counts()) # Checking unique values in 'Number of Dependents' column .
#df['Number of Dependents'] = df['Number of Dependents'].astype('category') # we can convert it to categorical as it has limited unique values
df.info()


df.duplicated().sum() # Checking for duplicate rows in the dataset


df.isnull().sum() # Checking for null values in each column


count_agenull = df['Age'].isnull().sum()   # Calculate the percentaage of missing values for Age column
percentaage =  count_agenull / df.shape[0] * 100
percentaage 


cat


# colect missing values py catgorical columns and numerical columns
cat_missing = []
num_missing=[]
for i in cat:
    if df[i].isnull().sum() > 0 :
        cat_missing.append(i) 
for i in num:
    if df[i].isnull().sum() > 0 :
        num_missing.append(i) 

print(cat_missing)
print(num_missing)


# defind function Calculate the percentaage of missing values in col

def  Percen_DfNull(col):
    count_agenull = df[col].isnull().sum()   
    percentaage =  count_agenull / df.shape[0] * 100
    return percentaage
print(" percentaage null for  catgorical columns ********************************")
for i in cat_missing:
    print(i,"= ",Percen_DfNull(i).round(4))
print(" \npercentaage null for  numerical columns ********************************")
for i in num_missing:
    print(i,"= ",Percen_DfNull(i).round(4))


for i in cat:
    plt.figure(figsize=(8,5))
    df[i].value_counts().plot(kind='bar', color='lightgreen')
    plt.title(i)
    plt.xlabel('catgory')
    plt.ylabel('count')
    plt.show()


plt.figure(figsize=(8,5))
df['Number of Dependents'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Number of Dependents')
plt.xlabel('catgory')
plt.ylabel('count')
plt.show()



plt.figure(figsize=(8,5))
df['Previous Claims'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Previous Claims')
plt.xlabel('catgory')
plt.ylabel('count')
plt.show()



for i in num_missing:
    plt.figure(figsize=(8,5))             
    plt.hist(df[i], bins=30, color='skyblue', edgecolor='black') 
    plt.title(i)  
    plt.xlabel('values')                       
    plt.ylabel('counter')                       
    plt.grid(axis='y', alpha=0.75)            
    plt.show()


# catgorical Columns
df['Marital Status'] = df['Marital Status'].fillna(df['Marital Status'].mode()[0])  #fill missing values py mode()[0]


# numerical Columns
df['Number of Dependents'] = df['Number of Dependents'].fillna(-1.0) #fill missing values py -1.0


df['Number of Dependents'].value_counts()


# convert to categorical type
df['Number of Dependents'] = df['Number of Dependents'].astype('category') 
df['Number of Dependents'].info()


df['Occupation'] = df['Occupation'].cat.add_categories(["Unknown"]) # add new category


df['Occupation'] = df['Occupation'].fillna('Unknown')  # fill missing values py new category


df['Occupation'].isnull().sum()


print(Percen_DfNull('Customer Feedback'))


df['Customer Feedback'].value_counts()


# fill nan values py MODE 
df['Customer Feedback'] = df['Customer Feedback'].fillna(df['Customer Feedback'].mode()[0])



df['Customer Feedback'].isna().sum()


# remove 'Number of Dependents' and 'Previous Claims' from numerical columns list
num = num.drop(['Number of Dependents','Previous Claims'])
num


df['Previous Claims'].unique()


# Previous Claims column is like catgorical column
# first convert to object colem
# than fill nan values 

#df['Previous Claims'] = df['Previous Claims'].astype('object')
df['Previous Claims'] = df['Previous Claims'].fillna(-1.0)
df['Previous Claims'] =df['Previous Claims'].astype('category')






df['Previous Claims'].isnull().sum()


# append 'Number of Dependents' and 'Previous Claims' to catgorical columns
cat=cat.append(pd.Index(['Number of Dependents','Previous Claims']))
cat


for i in num:
    if i in num_missing:
        print(f"{i} null percent =",Percen_DfNull(i))
    print(df[i].value_counts())
    print("***************************")


#df['Age'] = df['Age'].fillna(df['Age'].mean())  # numerical
#df['Annual Income'] = df['Annual Income'].fillna(df['Annual Income'].mean())  # numerical


col = df['Health Score']


m = round(col.mean(),2)
std = round(col.std(),2)
print(std,m)


# fill nan values py random values betwen m-std and m+std 
for i in num:
    if df[i].isnull().sum() > 0:
        m = df[i].mean()
        std = df[i].std()
        missing_idx = df[df[i].isna()].index
        rand_values = np.random.uniform(m - std, m + std, size=len(missing_idx))
        df.loc[missing_idx, i] = rand_values
        print(f"{i}: Filled {len(missing_idx)} missing values")
    else:
        print(f"{i}: No missing values")


for i in num:
    plt.figure(figsize=(8,5))             
    plt.hist(df[i], bins=30, color='skyblue', edgecolor='black') 
    plt.title(i)  
    plt.xlabel('values')                       
    plt.ylabel('count')                       
    plt.grid(axis='y', alpha=0.75)            
    plt.show()



for i in cat:
    plt.figure(figsize=(8,5))
    df[i].value_counts().plot(kind='bar', color='lightgreen')
    plt.title(i)
    plt.xlabel('catgory')
    plt.ylabel('count')
    plt.show()



cat


num



"""fig = px.histogram(
                    df,
                    x= 'Premium Amount',
                    y= 'Annual Income',
                    color="Marital Status",
                    nbins=20,
                    opacity=0.5,
                    title="Premium Distribution and Annual Income by Marital Status" )

fig.show()"""


""" The proportions of the categories in the column 'Policy Type' are approximately equal, 
              therefore there is no direct or strong effect on the target column."""
"""fig = px.histogram(
                    df,
                    x='Premium Amount',
                    color='Policy Type',
                   
                    nbins=20,
                    opacity=0.5,
                    title="Premium Distribution by Policy Type" )

fig.show()"""


""" The proportions of the categories in the column 'Gender' are approximately equal, 
              therefore there is no direct or strong effect on the target column."""
"""fig = px.histogram(
    df,
    x='Premium Amount',
   
    color="Gender",
    nbins=20,
    opacity=0.5,
    title="Premium Distribution by Gende" )

fig.show()"""


"""fig = px.histogram(
    df,
    x='Premium Amount',
    color="Insurance Duration",
    nbins=20,
    opacity=0.5,
    title="Premium Distribution by Insurance Duration" )

fig.show()"""


ins = df['Insurance Duration'].sort_values()
ins


fig = px.histogram(
    df,
    x='Premium Amount',
    color=ins,
    nbins=20,
    opacity=0.5,
    title="Premium Distribution by Insurance Duration" )

fig.show()


df.info()


df.groupby('Previous Claims')['Policy Type'].value_counts()


month = df['Policy Start Date'].dt.month
plt.figure(figsize=(8,5))
df.groupby(month)['Premium Amount'].max().plot(marker='o', linestyle='-', color='teal')
plt.title('Premium Amount py month')
plt.xlabel('month')
plt.ylabel('Premium Amount')
plt.show()


year = df['Policy Start Date'].dt.year
plt.figure(figsize=(8,5))
df.groupby(year)['Premium Amount'].max().plot(marker='o', linestyle='-', color='teal')
plt.title('Premium Amount py year')
plt.xlabel('year')
plt.ylabel('Premium Amount')
plt.show()


df.groupby(year)['Policy Type'].value_counts()


cat



"""for i in cat:
     fig = px.pie(df,names= i , title=f'{i}')
     fig.show()
   """


num


NumCorr=[]
for i in num:
    NumCorr.append(df[i].corr(df['Premium Amount']))

NumCorr


"""We have a very large number of values, so the scatter is not clearly visible. Therefore, 
we use this method to group the values into sets to reduce the large number, 
making the drawings appear clearer."""

q = pd.qcut(df['Annual Income'], q=20)
grouped = df.groupby(q)['Premium Amount'].mean()
# >>>>>>>>>> 1 <<<<<<<<<<<<
grouped.plot(kind='line', marker='o')
plt.title('Mean Premium Amount by Annual Income quantiles')
plt.show()



# x = group for 'Annual Income'
grouped.index


q = pd.qcut(df['Annual Income'], q=100)
grouped = df.groupby(q)['Premium Amount'].mean()
# >>>>>>>>>>>> 2 <<<<<<<<<<
plt.figure(figsize=(8,5))
plt.scatter(grouped.index.astype(str), grouped.values)

plt.title('Mean Premium Amount by Annual Income quantiles')
plt.xlabel('Annual Income Quantiles')
plt.ylabel('Mean Premium Amount')

plt.show()



fig=px.box(df,y='Age',title='outliers for Age')
fig.show()


num


"""for i in num:
    fig=px.box(df,y=i,title=f'outliers for {i}')
    fig.show()"""


outlier = df[df['Annual Income']< 1000]

outlier.groupby('Occupation')['Annual Income'].agg(['mean','max','min'])


q1 =df['Annual Income'].quantile(0.25)
q3 =df['Annual Income'].quantile(0.75)
IQR = q3 - q1
lower = q1 - 1.5*IQR
upper = q3 + 1.5*IQR
outliers = df[(df['Annual Income'] < lower) | (df['Annual Income'] > upper)]
outliers


"""A better and more comprehensive understanding of the nature of the data is necessary 
to determine whether the values â€‹â€‹are outliers or not."""
"""for i in num:
    q1 =A better and more comprehensive understanding of the nature of the data is necessary to determine whether the values â€‹â€‹are outliers or not.df[i].quantile(0.25)
    q3 =df[i].quantile(0.75)
    IQR = q3 - q1
    lower = q1 - 1.5*IQR
    upper = q3 + 1.5*IQR
    outliers = df[(df[i] < lower) | (df[i] > upper)]
outliers"""


cat


num


# convert date columen to year , month and day columens 
df['year']=df['Policy Start Date'].dt.year
df['month']=df['Policy Start Date'].dt.month
df['day'] = df['Policy Start Date'].dt.day
df = df.drop('Policy Start Date',axis = 1)


# ENCODEING
le = LabelEncoder()
for i in cat:
    df[i]=le.fit_transform(df[i])



df


df.info()


# Splitting Data
X = df.drop('Premium Amount',axis = 1)
Y = df['Premium Amount']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



x_train


y_train


"""df['year']=df['Policy Start Date'].dt.year
df['month']=df[' Policy Start Date'].dt.month
df['day'] = df[' Policy Start Date'].dt.day"""


# scalling Data

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)



model = LinearRegression()
model.fit(x_train_scaled, y_train)




y_pred = model.predict(x_test)


mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2 Score:", r2)

