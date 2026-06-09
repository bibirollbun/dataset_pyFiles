import pandas as pd
df = pd.read_csv("train.csv")
df


# Checking the no. of rows and columns in our dataset
df.shape


# Checking the no. of unique values in each dataset
df.nunique()


# Null Values in Dataset
df.isnull().sum()


df=df.drop("density" , axis=1)
df.head()


df=df.drop("BIRADS" , axis=1)
df.head()


# Checking the datatype of each column
df.info()


# Extracting age column
df.iloc[:,5:6]


# check for duplicated data in dataset
df.duplicated().sum()


import numpy as np
from sklearn.impute import SimpleImputer
# Handling Null Values using SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(df.iloc[:,5:6].values)
df.iloc[:,5:6] = imputer.transform(df.iloc[:,5:6].values)


# Changing the datatype
# Convert image_id and patient_id to string
df["image_id"] = df["image_id"].astype("str")
df["patient_id"] = df["patient_id"].astype("str")


df.isnull().sum()


# Dropping unrelated column
df.drop("site_id",axis=1,inplace=True)
df.head()


df.isna().sum()


# Correlation between various columns in a dataset
numeric_df = df.select_dtypes(include=['number'])
numeric_df.corr()


import seaborn as sns
sns.heatmap(numeric_df.corr())


import matplotlib.pyplot as plt
plt.subplot(1,2,1)
sns.distplot(df["age"]) 

plt.subplot(1,2,2)
sns.boxplot(df["machine_id"])  


# Checking for outlier in the dataset
df.describe()


df['age'].value_counts()


sns.distplot(df['age'])


sns.boxplot(df['age'])


Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
min_value = Q1-1.5*IQR
max_value = Q3+1.5*IQR
print("min_value",min_value," ","max_value=",max_value)


df[(df['age']<min_value) | (df['age']>max_value)]


df[df['age']==89]


df[(df['age']<29)].shape


df.groupby("age")['cancer'].sum().sort_values(ascending=False)


bins=[20,30,40,50,60,70,80,90]
df['age-group'] =  pd.cut(df['age'], bins=bins)
sns.countplot(x='age-group', data=df)
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Distribution of Age Groups')
plt.show()


df.groupby('age')['cancer'].sum()


# Removing people of age group less than 29 from our dataset
df = df[(df["age"]>min_value)]


grouped_df = df.groupby('age-group')['cancer'].count().reset_index()
grouped_df.plot(x='age-group',y='cancer',kind='bar')


df.groupby("age-group")["cancer"].count().sort_values(ascending=False)


df.groupby("implant")["cancer"].count().sort_values(ascending=False)


df.groupby('implant')['cancer'].count().plot(kind='bar')


# No of patients not having implant but cancer positive
len(df[(df['implant']==0) & (df['cancer']==1)])


# No of patients having implant and cancer positive
len(df[(df['implant']==1) & (df['cancer']==1)])


df.groupby("biopsy")["cancer"].count().sort_values(ascending=False)


# No of patients who had biopsy and also tested positive for cancer
len(df[(df['cancer']==1) & (df['biopsy']==1)])


sns.countplot(x='biopsy', hue='cancer', data=df)
plt.xlabel('Biopsy')
plt.ylabel('Count')
plt.title('Biopsy vs. Cancer')
plt.show()


df.groupby("invasive")["cancer"].count().sort_values(ascending=False)


len(df[(df['invasive']==1) & (df['cancer']==1)])


len(df[(df['invasive']==0) & (df['cancer']==1)])


sns.countplot(x='invasive', hue='cancer', data=df)


# There are 11907 unique patients in the dataset
len(df['patient_id'].unique())


df.shape


# Group columns having same patient_id
patient_summary = df.groupby('patient_id')['image_id'].count().sort_values(ascending=False)
patient_summary


# There are no duplicate images in our dataset
len(df['image_id'].unique())


cancer_per_patient = df.groupby("patient_id")['cancer'].max().values
negative_count = (cancer_per_patient==0).sum()
positive_count = (cancer_per_patient==1).sum()
print(f'There are {negative_count} no.of unique patients negative with cancer.')
print(f'There are {positive_count} no.of unique patients positive with cancer.')


# Train-Test Split
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
X = df_train.drop('cancer', axis=1)  
y = df_train['cancer']  


df_test




