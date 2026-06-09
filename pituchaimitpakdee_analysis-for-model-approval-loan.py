#Just run this, you will get the dataset that kaggle provide for you in this compitition :D
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt



train_df = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")
sample_df = pd.read_csv("/kaggle/input/playground-series-s4e10/sample_submission.csv")


train_df


test_df 


# we can see the imbalance class for this one
train_df.loan_status.value_counts()



train_df.cb_person_default_on_file.unique()


#checking nan value exist
train_df.isnull().sum()


test_df.isnull().sum()


print(train_df.describe())


print(train_df.describe(include = "all")) # As you can see I used include="all" to quick summary the types of each varieables


train_df.dtypes



# EDA is the process of analyzing and summarizing a dataset to understand its structure, detect patterns, and identify potential issues before applying machine learning models.
# For this datasets we have 2 data types, which are numerical and categorical data types.
# Numerical seem not pretty hard but for the categorical types of data. how to deal with it if we have to put it in model? 
# but first let's focus on finding the relationship to loan_status


#Correlation can be only use with numerical data types 
correlation_matrix = train_df[['person_income', 'person_age','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length','loan_status']].corr()

loan_status_corr = correlation_matrix['loan_status'].sort_values(ascending=False)
print(loan_status_corr)


train_df['person_income']


#Let's check the distribution of person income first
pd.options.display.float_format = '{:.2f}'.format
person_income_describe = train_df['person_income'].describe(include = 'all')
person_income_describe


plt.figure(figsize=(10, 7))
plt.hist(train_df['person_income'], bins=55)
plt.title("Person Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()

