import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler 


df = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
df


test_data = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
test_data


df.isnull().sum()


test_data.isnull().sum()


print(df.duplicated().sum())
print(test_data.duplicated().sum())



df.info()


for col in df.columns :
    print(df[col].value_counts(normalize=True))
    print("________________")
    


df['loan_status'] = df['loan_status'].replace({0 : "No" , 1 : "Yes"})


#df['loan_status']= df['loan_status'].astype('category')


df.info()


df.isnull().sum()


df.select_dtypes(include='object')


# change data type of object to category
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')

df.info()
    


# Dictionary Comprehension
test_data = test_data.astype( {col : 'category' for col in test_data.select_dtypes(include='object').columns })
test_data.info()


del df['id']


num = df.select_dtypes(include= ('int64' , 'float64')).columns
num


cat = df.select_dtypes(include= 'category').columns
cat
                       


px.histogram( df , x = 'person_age' , title = 'Histogram for Ages distribution')



for col in num :
    fig = px.histogram( df , x = col , color='loan_status', title = f'Histogram for {col} distribution')
    fig.show()



# supplot
plt.figure(figsize=(20,15))
for i , col in enumerate(num , start=1):
    plt.subplot(4,2 , i)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()    
plt.show()


# IQR ( Numerical Continues Values ) 
# Q1, Q2 , Q3 , IQR = Q3 - Q1 , Lower, upper

for col in num :
    print(f"Column Name : {col}")
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - ( 1.5 *IQR )
    upper = Q3 + (1.5 * IQR )
    print (f" Q1 : {Q1} , Q3 : {Q3} \nLower bound is : {lower} \nUpper Bound is : {upper}")
    outlier = df[(df[col] < lower) | (df[col] > upper ) ]
    print(f"Number of outlier : {outlier.shape[0]}")
    print("____________________")
    


df.columns


data = df['person_age']


med = data.median()


data.describe()


sns.kdeplot(df['person_age'] , label = data , fill=True )


# median , mad , MZ 
abs_deviation = abs (data - med )
abs_deviation


mad = abs_deviation.median()
mad


modified_z_score = 0.6745 * (data - med ) / mad
modified_z_score


sns.kdeplot(modified_z_score , label = data , fill=True )


modified_z_score.describe()


threshold = 3.5
outliers = df[(modified_z_score < -3.5) | (modified_z_score > 3.5) ]
outliers
# outlier = abs(modified_z_score > 3.5 )


#data_with_no_outliers = df[(modified_z_score > -3.5) & (modified_z_score < 3.5) ]
#data_with_no_outliers


# Define function to detect outlier 
def mad_outliers( column_name , threshold = 3.5 ): # threshold = 3.5 Defult value
    med = df[column_name].median()
    abs_deviation = abs ( df[column_name] - med )
    mad = abs_deviation.median()
    modified_z_score = 0.6745 * (df[column_name] - med ) / mad
    outliers =  df[(modified_z_score < -threshold) | (modified_z_score > threshold )]
    print(f"Outlier size in {column_name} is : {outliers.shape[0]}")
    return outliers

for col in num :
    mad_outliers(col , 4)

# store modified_z_score for each column in the data in a new dataframe 
# Kde plot for the new datframe ( suplots )
# drop outlier based on the observation we take from each kde plot 



for col in num :
    mad_outliers(col , 2.5)


for col in num :
    mad_outliers(col) # threshold = 3.5 Defult value


test_data


def mad_outliers( column_name , threshold = 3.5 ):
    med = test_data[column_name].median()
    abs_deviation = abs ( test_data[column_name] - med )
    mad = abs_deviation.median()
    modified_z_score = 0.6745 * (test_data[column_name] - med ) / mad
    outliers =  test_data[(modified_z_score < -threshold) | (modified_z_score > threshold )]
    print(f"Outlier size in {column_name} is : {outliers.shape[0]}")

for col in test_data.select_dtypes(include=('float64' , 'int64')).columns :
    mad_outliers(col , 3.5)

# for test data ( DON'T DROP )


df1 = df.copy()
df1


def mad_outliers( column_name , threshold = 3.5 ): # threshold = 3.5 Defult value
    med = df1[column_name].median()
    abs_deviation = abs ( df1[column_name] - med )
    mad = abs_deviation.median()
    modified_z_score = 0.6745 * (df1[column_name] - med ) / mad
    outliers =  (modified_z_score < -threshold) | (modified_z_score > threshold )
    #print(f"Outlier size in {column_name} is : {outliers.shape[0]}")
    return outliers


# deleting Outlier from column person_age 
age_outlier =  mad_outliers('person_age')
age_outlier


df1 = df1[~age_outlier]
df1
#dropna : drop null values 


num


# supplot
# Plotting after handling outlire in age ONLYYY
plt.figure(figsize=(20,15))
for i , col in enumerate(num , start=1):
    plt.subplot(4,2 , i)
    sns.boxplot(x=df1[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()    
plt.show()


px.histogram( df , x = 'person_age', color='loan_status' , title = 'Histogram for Ages distribution')



px.histogram( df1 , x = 'person_age' ,  color='loan_status' , title = 'Histogram for Ages distribution')



# person_income
income_outlier =  mad_outliers('person_income')
df1 = df1[~income_outlier]
fig1 = px.histogram( df , x = 'person_income', color='loan_status' , title = 'Histogram for Income distribution')
fig1.show()
fig2 = px.histogram( df1 , x = 'person_income', color='loan_status' , title = 'Histogram for Income distribution')
fig2.show()


df1


# supplot
# Plotting after handling outlire in age ONLYYY
plt.figure(figsize=(20,15))
for i , col in enumerate(num , start=1):
    plt.subplot(4,2 , i)
    sns.boxplot(x=df1[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()    
plt.show()


# cb_person_cred_hist_length
income_outlier =  mad_outliers('cb_person_cred_hist_length')
df1 = df1[~income_outlier]
fig1 = px.histogram( df , x = 'cb_person_cred_hist_length', color='loan_status' , title = 'Histogram for cp_history distribution')
fig1.show()
fig2 = px.histogram( df1 , x = 'cb_person_cred_hist_length', color='loan_status' , title = 'Histogram for cp_history distribution')
fig2.show()


df1


# supplot
# Plotting after handling outlire in age ONLYYY
plt.figure(figsize=(20,15))
for i , col in enumerate(num , start=1):
    plt.subplot(4,2 , i)
    sns.boxplot(x=df1[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()    
plt.show()


# Define function to detect outlier 
def mad_outliers( column_name , threshold = 3.5 ): # threshold = 3.5 Defult value
    med = df1[column_name].median()
    abs_deviation = abs ( df1[column_name] - med )
    mad = abs_deviation.median()
    modified_z_score = 0.6745 * (df1[column_name] - med ) / mad
    outliers = (modified_z_score < -threshold) | (modified_z_score > threshold )
    print(f"Outlier size in {column_name} is : {outliers.shape[0]}")
    return outliers

for col in num : 
    income_outlier =  mad_outliers(col, 4.5)
    df1 = df1[~income_outlier]
    fig1 = px.histogram( df , x =col, color='loan_status' , title = f'Histogram for {col} distribution before Outlier Handling')
    fig1.show()
    fig2 = px.histogram( df1 , x = col, color='loan_status' , title = f'Histogram for {col} distribution After Outlier Handling')
    fig2.show()


# supplot
# Plotting after handling outlire in age ONLYYY
plt.figure(figsize=(20,15))
for i , col in enumerate(num , start=1):
    plt.subplot(4,2 , i)
    sns.boxplot(x=df1[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()    
plt.show()
df1


# dropna , fill na 
# drop outlier , smoothing outlier


df2 = df.copy()
df2


# Preprocessing pipeline 
# columntransformer [ encoder , sclaer, fill , drop , type casting ]



# CI/CD 
# Model Optimization ( Parameter Tuning , Hyper-Paramter " PreProcessing steps" Tuning)

