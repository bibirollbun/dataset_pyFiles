import pandas as pd
import numpy as np
import math, copy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


#Uploading and visualizing the target training dataset
train_targ=pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx')
train_targ


#Uploading and visualizing the categorical training dataset
train_cat=pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx')
train_cat


test_cat=pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')


#Uploading and visualizing the quantitative training dataset
train_quant=pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx')
train_quant


test_quant=pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')


#Uploading and visualizing the functional training dataset
train_func=pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv')
train_func


test_func=pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')


for col in train_cat.select_dtypes(include='number').columns:
    train_cat[col] = train_cat[col].astype('category')

# Creating a list of all of the columns except the first
columns_to_encode = train_cat.columns[1:].tolist()

# encoding categorical data
data_encoded = pd.get_dummies(train_cat[columns_to_encode], drop_first=True)
data_encoded = data_encoded.map(lambda x: 1 if x is True else (0 if x is False else x))

train_cat = pd.concat([train_cat.drop(columns=columns_to_encode), data_encoded], axis=1)


for col in test_cat.select_dtypes(include='number').columns:
    test_cat[col] = test_cat[col].astype('category')

# Creating a list of all of the columns except the first
columns_to_encode = test_cat.columns[1:].tolist()

# encoding categorical data
data_encoded = pd.get_dummies(test_cat[columns_to_encode], drop_first=True)
data_encoded = data_encoded.map(lambda x: 1 if x is True else (0 if x is False else x))

# Combine encoded columns with the rest of the DataFrame
test_cat= pd.concat([test_cat.drop(columns=columns_to_encode), data_encoded], axis=1)




train_cat_func = pd.merge(train_cat, train_func, on = 'participant_id')
train_df=pd.merge(train_cat_func, train_quant, on = 'participant_id')
train_df


test_cat_func = pd.merge(test_cat, test_func, on = 'participant_id')
test_df=pd.merge(test_cat_func, test_quant, on = 'participant_id')
test_df


train_df.fillna(0)


train_df['Sex_F']=train_targ['Sex_F']
train_df['ADHD_Outcome']=train_targ['ADHD_Outcome']

fem=train_df[train_df['Sex_F']==1]
male=train_df[train_df['Sex_F']==0]

adhd=train_df[train_df['ADHD_Outcome']==1]
no_adhd=train_df[train_df['ADHD_Outcome']==0]



sns.boxplot(x=train_targ['Sex_F'], y=train_df['SDQ_SDQ_Hyperactivity'])
plt.show()
t_test, p_value=stats.ttest_ind(male['SDQ_SDQ_Hyperactivity'], fem['SDQ_SDQ_Hyperactivity'])

alpha = 0.05


if p_value < alpha:
    print("Reject the null hypothesis")
    print(p_value)
else:
    print("Fail to reject the null hypothesis")


sns.boxplot(x=train_targ['ADHD_Outcome'], y=train_df['SDQ_SDQ_Hyperactivity'])
plt.show()
t_test, p_value=stats.ttest_ind(adhd['SDQ_SDQ_Hyperactivity'], no_adhd['SDQ_SDQ_Hyperactivity'])

alpha = 0.05


if p_value < alpha:
    print("Reject the null hypothesis")
    print(p_value)
else:
    print("Fail to reject the null hypothesis")


sns.boxplot(x=train_targ['ADHD_Outcome'], y=train_df['SDQ_SDQ_Generating_Impact'])
plt.show()
t_test, p_value=stats.ttest_ind(adhd['SDQ_SDQ_Generating_Impact'], no_adhd['SDQ_SDQ_Generating_Impact'])

alpha = 0.05


if p_value < alpha:
    print("Reject the null hypothesis")
    print(p_value)
else:
    print("Fail to reject the null hypothesis")


sns.boxplot(x=train_targ['Sex_F'], y=train_df['SDQ_SDQ_Externalizing'])
plt.show()
t_test, p_value=stats.ttest_ind(male['SDQ_SDQ_Externalizing'], fem['SDQ_SDQ_Externalizing'])

alpha = 0.05


if p_value < alpha:
    print("Reject the null hypothesis")
    print(p_value)
else:
    print("Fail to reject the null hypothesis")


sns.boxplot(x=train_targ['ADHD_Outcome'], y=train_df['SDQ_SDQ_Externalizing'])
plt.show()
t_test, p_value=stats.ttest_ind(adhd['SDQ_SDQ_Externalizing'], no_adhd['SDQ_SDQ_Externalizing'])

alpha = 0.05


if p_value < alpha:
    print("Reject the null hypothesis")
    print(p_value)
else:
    print("Fail to reject the null hypothesis")


sns.boxplot(x=train_targ['Sex_F'], y=train_df['SDQ_SDQ_Emotional_Problems'])
plt.show()
t_test, p_value=stats.ttest_ind(male['SDQ_SDQ_Emotional_Problems'], fem['SDQ_SDQ_Emotional_Problems'])

alpha = 0.05


if p_value < alpha:
    print("Reject the null hypothesis")
    print(p_value)
else:
    print("Fail to reject the null hypothesis")


sns.boxplot(x=train_targ['ADHD_Outcome'], y=train_df['SDQ_SDQ_Emotional_Problems'])
plt.show()
t_test, p_value=stats.ttest_ind(adhd['SDQ_SDQ_Emotional_Problems'], no_adhd['SDQ_SDQ_Emotional_Problems'])

alpha = 0.05


if p_value < alpha:
    print("Reject the null hypothesis")
    print(p_value)
else:
    print("Fail to reject the null hypothesis")


sns.boxplot(x=train_targ['ADHD_Outcome'], y=train_df['SDQ_SDQ_Difficulties_Total'])
plt.show()
t_test, p_value=stats.ttest_ind(adhd['SDQ_SDQ_Difficulties_Total'], no_adhd['SDQ_SDQ_Difficulties_Total'])

alpha = 0.05


if p_value < alpha:
    print("Reject the null hypothesis")
    print(p_value)
else:
    print("Fail to reject the null hypothesis")


sns.boxplot(x=train_targ['ADHD_Outcome'], y=train_df['SDQ_SDQ_Conduct_Problems'])
plt.show()
t_test, p_value=stats.ttest_ind(adhd['SDQ_SDQ_Conduct_Problems'], no_adhd['SDQ_SDQ_Conduct_Problems'])

alpha = 0.05


if p_value < alpha:
    print("Reject the null hypothesis")
    print(p_value)
else:
    print("Fail to reject the null hypothesis")


sns.boxplot(x=train_targ['ADHD_Outcome'], y=train_df['APQ_P_APQ_P_ID'])
plt.show()
t_test, p_value=stats.ttest_ind(adhd['APQ_P_APQ_P_ID'], no_adhd['APQ_P_APQ_P_ID'])

alpha = 0.05


if p_value < alpha:
    print("Reject the null hypothesis")
    print(p_value)
else:
    print("Fail to reject the null hypothesis")


sns.boxplot(x=train_targ['ADHD_Outcome'], y=train_df['APQ_P_APQ_P_OPD'])
plt.show()

t_test, p_value=stats.ttest_ind(adhd['APQ_P_APQ_P_OPD'], no_adhd['APQ_P_APQ_P_OPD'])

alpha = 0.05

# Interpret the results
if p_value < alpha:
    print("Reject the null hypothesis")
    print(p_value)
else:
    print("Fail to reject the null hypothesis")


sns.boxplot(x=train_targ['Sex_F'], y=train_df['APQ_P_APQ_P_PP'])
plt.show()
t_test, p_value=stats.ttest_ind(fem['APQ_P_APQ_P_PP'], male['APQ_P_APQ_P_PP'])

alpha = 0.05

# Interpret the results
if p_value < alpha:
    print("Reject the null hypothesis")
    print(p_value)
else:
    print("Fail to reject the null hypothesis")


sns.boxplot(x=train_targ['Sex_F'], y=train_df['APQ_P_APQ_P_OPD'])
plt.show()

t_test, p_value=stats.ttest_ind(fem['APQ_P_APQ_P_OPD'], male['APQ_P_APQ_P_OPD'])

alpha = 0.05

# Interpret the results
if p_value < alpha:
    print("Reject the null hypothesis")
    print(p_value)
else:
    print("Fail to reject the null hypothesis")


def mean_func(group):
    train_mean=pd.DataFrame()
    for col in train_df.loc[ :, '0throw_1thcolumn':'198throw_199thcolumn']:
        train_mean[col]=group[col]
    

    return train_mean.mean().to_frame()
    



df_adhd=mean_func(adhd)
df_adhd=df_adhd.reset_index()



df_adhd[['row', 'columns']] = df_adhd['index'].str.split('_', expand=True)



df_adhd['value']=df_adhd[0]

df_new=df_adhd.loc[:,'row':'value']

price= int ( ''.join(filter(str.isdigit, df['row']) ) )
price


pivoted = df_new.pivot(index=int(filter(str.isdigit,df_new["row"])), columns="columns", values='value')

sns.heatmap(pivoted.fillna(0).tail())

