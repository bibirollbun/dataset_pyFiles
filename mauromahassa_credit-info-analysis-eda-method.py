#imports
import missingno as msno
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 


df=pd.read_csv("application_train.csv")
df.head()


df.info()


df.describe()


#Checking missing values
data.isnull().sum().head(50)


msno.matrix(df.sample(500))


total = df.isnull().sum().sort_values(ascending=False) # Total rows that are null in each collumn
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False) # Percentage of what's missing and total
missing_df = pd.concat([total, percent], axis=1, keys=['Total', 'missing_ratio'])
missing_df.head(50)


# How many loans have been payed?
colors = ['#1b9e77', '#a9f971', '#fdaa48','#6890F0','#A890F0']
paid_unpaid = df["TARGET"].value_counts().plot(kind='bar',color = colors)

print("1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample, 0 - all other cases")
print(a1)


# Gender os dataset
gender_dataset = df["CODE_GENDER"].value_counts().plot(kind='pie',autopct = '%1.0f%%',title='Gender distribution in the dataset')
df["CODE_GENDER"].value_counts()


# Gender, those who pay
gender_no_pay = df.loc[df['TARGET']==1,'CODE_GENDER']
gender_no_pay.value_counts().plot(kind='pie',autopct = '%1.0f%%',title='Gender distribution for clients with paying difficulties')


gender_no_pay.value_counts().plot(kind='bar',title='Gender distribution for clients with paying difficulties',color = colors[2:])


#Gender, those who don't pay
# Gender, those who pay
gender_pay = df.loc[df['TARGET']==0,'CODE_GENDER']
gender_pay.value_counts().plot(kind='bar', title='Gender distribution for clients without paying difficulties ')


gender_pay.value_counts().plot(kind='pie', autopct = '%1.0f%%',title='Gender distribution without paying difficuties')


# Those who own a family

family_pay = df.loc[df['TARGET']==1,'NAME_FAMILY_STATUS']
family_pay.value_counts().plot(kind='pie',autopct = '%1.0f%%',title='Family status distribution of clients with difficulties paying')


# Those who own a family

family_not_pay = df.loc[df['TARGET']==0,'NAME_FAMILY_STATUS']
family_not_pay.value_counts().plot(kind='pie',autopct = '%1.0f%%',title='Family status distribution of clients without difficulties paying')


# How many children

family_pay = df.loc[df['TARGET']==0,'CNT_CHILDREN']
family_pay.value_counts().plot(kind='bar',color=colors,title='Number of children distribution for no difficulties paying clients')


# How many children

family_not_pay = df.loc[df['TARGET']==1,'CNT_CHILDREN']
family_not_pay.value_counts().plot(kind='bar',color=colors,title='Number of children distribution for without difficulties paying clients')


# how many Family members client have
family_members = df['CNT_FAM_MEMBERS'].value_counts().plot(kind='bar',color=colors,title='Number of family members distribution')


# how many Family members client have
family_no_pay = df.loc[df['TARGET']==1,'CNT_FAM_MEMBERS']
family_no_pay.value_counts().plot(kind='bar',color=colors,title='Number of family members distribution for with difficulties paying clients')


# how many Family members client have

family_pay = df.loc[df['TARGET']==0,'CNT_FAM_MEMBERS']
family_pay.value_counts().plot(kind='bar',color=colors,title='Number of family members distribution for without difficulties paying clients')


def distribution(column, colors, difficulties,title,graph_type='bar'):
    # how many Family members client have
    distribution = df.loc[df['TARGET']==difficulties,column]
    return distribution.value_counts().plot(kind='bar',color=colors,title=title)


# Income
distribution('NAME_INCOME_TYPE', colors,1, 'bar', 'INCOME TYPE for members with dificulties')



# Income
distribution('NAME_INCOME_TYPE', colors,0, 'bar', 'INCOME TYPE for members without dificulties')



#Occupation type
distribution('OCCUPATION_TYPE', colors,1, 'bar', 'Occupation TYPE for members with dificulties')



#Occupation type
distribution('OCCUPATION_TYPE', colors,0, 'bar', 'Occupation TYPE for members without dificulties')




distribution('NAME_EDUCATION_TYPE', colors,1, 'bar', 'Occupation TYPE for members with dificulties')




distribution('NAME_EDUCATION_TYPE', colors,0, 'bar', 'Occupation TYPE for members without dificulties')



#NAME_HOUSING_TYPE

distribution('NAME_EDUCATION_TYPE', colors,0, 'bar', 'Name housing TYPE for members without dificulties')




distribution('NAME_EDUCATION_TYPE', colors,1, 'bar', 'Occupation TYPE for members with dificulties')




distribution('NAME_EDUCATION_TYPE', colors,1, 'bar', 'Occupation TYPE for members with dificulties')





