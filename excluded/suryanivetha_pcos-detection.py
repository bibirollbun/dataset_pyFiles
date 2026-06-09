import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


train_data =  pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')
test_data  =  pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')



train_data.head()


test_data.head()


train_data.head()


#understanding the data
train_data.info()


#exploring the age column
train_data['Age'].unique()


train_data['Age'].value_counts()


#Replacing the values
train_data.replace({'Age' : {'15-20' : 'Less than 20', 
                             '30-25' : '25-30',
                             '35-44' : '40-45',
                             '30-40':'35-40',
                             'Less than 20-25' : 'Less than 20'
                            }}, inplace =True)


#corrected format
print(train_data['Age'].value_counts())


#exploring the hormonal imbalance
train_data['Hormonal_Imbalance'].value_counts()


train_data.replace({'Hormonal_Imbalance' : {'No, Yes, not diagnosed by a doctor':'No',
                                            'Yes Significantly' : 'Yes'
                                           }},inplace =True)


#corrected format
train_data['Hormonal_Imbalance'].value_counts()


#Exploring the hirsutism
train_data['Hirsutism'].value_counts()


train_data.replace({'Hirsutism' : {'No, Yes, not diagnosed by a doctor':'No',
                                           }},inplace =True)


#Corrected format
train_data['Hirsutism'].value_counts()


#Exploring Conception_Difficulty
train_data['Conception_Difficulty'].value_counts()


train_data.replace({'Conception_Difficulty' : {'Yes, diagnosed by a doctor':'Yes',
                                            'No, Yes, not diagnosed by a doctor' : 'No'
                                           }},inplace =True)


#corrected format
train_data['Conception_Difficulty'].value_counts()


#Exploring Insulin_Resistance
train_data['Insulin_Resistance'].value_counts()


train_data.replace({'Insulin_Resistance' : {
                                            'No, Yes, not diagnosed by a doctor' : 'No'
                                           }},inplace =True)


#corrected formate for Insulin_Resistance
train_data['Insulin_Resistance'].value_counts()


#Exploring Exercise_Frequency
train_data['Exercise_Frequency'].value_counts()


train_data.replace({'Exercise_Frequency' : {'Rarely' : 'Low',
                                            '1-2 Times a Week' : 'Low',
                                            '3-4 Times a Week' : 'Moderate',
                                            '6-8 Times a Week' : 'High',
                                            '6-8 hours'        : 'Moderate',
                                            'Less than usual'  : 'Low',
                                            'Less than 6 hours' : 'Low' 
                                           }},inplace =True)


#corrected format
train_data['Exercise_Frequency'].value_counts()


#Exploring Exercise_Type
train_data['Exercise_Type'].value_counts()


#just making it perfect format
train_data['Exercise_Type'] = train_data['Exercise_Type'].str.split('(').str[0].str.strip()


#corrected format
train_data['Exercise_Type'].value_counts()


#Analysing the Exercise Duration
train_data['Exercise_Duration'].value_counts()


train_data.replace({'Exercise_Duration' : {'45 minutes' : 'More than 30 minutes',
                                            '20 minutes' : 'Less than 30 minutes',
                                           'Less than 6 hours' : 'More than 30 minutes',
                                           '30 minutes to 1 hour' : 'More than 30 minutes'  
                                          }},inplace = True)


#corrected format
train_data['Exercise_Duration'].value_counts()


#mapping
def mapping(x):

    if x == 'Not Applicable':
        return 'No'
    elif x  == 'Less than 30 minutes':
        return 'Short'
    elif x == '30 minutes':
        return 'Moderate'
    elif x  == 'More than 30 minutes':
        return 'Long'
    else:
        pass


train_data['Exercise_Duration'] = train_data['Exercise_Duration'].apply(mapping)


train_data['Exercise_Duration'].value_counts()


#explore the sleep hours
train_data['Sleep_Hours'].value_counts()


#mapping
def mapping_sleep(x):

    if x == '6-8 hours':
        return 'Good'
    elif x  == 'Less than 6 hours':
        return 'Poor'
    elif x == '9-12 hours' or x == 'More than 12 hours':
        return 'Over'
    elif x  == '3-4 hours':
        return 'Over'
    else:
        pass


train_data['Sleep_Hours'] = train_data['Sleep_Hours'].apply(mapping_sleep)


#corrected format
train_data['Sleep_Hours'].value_counts()


#Exercise Benefit exploration
train_data['Exercise_Benefit'].value_counts()


#mapping
def mapping_exe(x):

    if x == 'Somewhat':
        return 'Moderate'
    elif x  == 'Not at All':
        return 'No'
    elif x == 'Yes Significantly':
        return 'High'
    elif x  == 'Not Much':
        return 'Low'
    else:
        pass


train_data['Exercise_Benefit'] = train_data['Exercise_Benefit'].apply(mapping_exe)


#correct format
train_data['Exercise_Benefit'].value_counts()


train_data.head()


test_data.info()


test_data.head()


#analysing the age column in test data
test_data['Age'].value_counts()


#Replacing the values
test_data.replace({'Age' : { '15-20' : 'Less than 20', 
                             '30-25' : '25-30',
                             '35-44' : '40-45',
                             '30-40':'35-40',
                             'Less than 20-25' : 'Less than 20',
                             '30-30' : '25-30',
                             'Less than 20)' : 'Less than 20',
                             '25-25' : '20-25',
                            '50-60': '45 and above',
                             '22-25' : '20-25',
                              '20' : 'Less than 20',
                             '45-49' : '45 and above'
                              
                            
                            }}, inplace =True)


#fixing it
test_data['Age'].value_counts()


#analysing Hormonal_Imbalance
test_data['Hormonal_Imbalance'].value_counts()


#analysing the Hyperandrogenism
test_data['Hyperandrogenism'].value_counts()


#analysing teh hirsutism
test_data['Hirsutism'].value_counts()


#analysing the Conception_Difficulty
test_data['Conception_Difficulty'].value_counts()


#fixing it
test_data.loc[test_data['Conception_Difficulty'] == 'Somewhat', 'Conception_Difficulty'] = 'Yes'


#fixing Conception_Difficulty
test_data['Conception_Difficulty'].value_counts()


#Analysing the Insulin_Resistance
test_data['Insulin_Resistance'].value_counts()


#fixing it
test_data.loc[test_data['Insulin_Resistance'] == 'Yes Significantly','Insulin_Resistance'] = 'Yes'


#showing it
test_data['Insulin_Resistance'].value_counts()


#analysing the Exercise_Frequency
test_data['Exercise_Frequency'].value_counts()


test_data.replace({'Exercise_Frequency' : {'Rarely' : 'Low',
                                            '1-2 Times a Week' : 'Low',
                                            '3-4 Times a Week' : 'Moderate',
                                            '6-8 Times a Week' : 'High',
                                            '6-8 hours'        : 'Moderate',
                                            'Less than usual'  : 'Low',
                                            'Less than 6 hours' : 'Low',
                                            'Yes' : 'Moderate',
                                            'Daily' : 'High',
                                            'Less than 6-8 Times a Week' : 'Moderate',
                                            '30-35' : 'Low',
                                            'Somewhat' : 'Low',
                                            '1/2 Times a Week' : 'Low'
                                           }},inplace =True)


#seeing it
test_data['Exercise_Frequency'].value_counts()


#Exploring Exercise_Type
test_data['Exercise_Type'].value_counts()


#just making it perfect format
test_data['Exercise_Type'] = test_data['Exercise_Type'].str.split('(').str[0].str.strip()


test_data['Exercise_Type'].value_counts()


test_data.loc[test_data['Exercise_Type'] == 'Yes', 'Exercise_Type'] = 'Strength training'
test_data.loc[test_data['Exercise_Type'] == 'Yes Significantly', 'Exercise_Type'] = 'Strength training'
test_data.loc[test_data['Exercise_Type'] == 'No', 'Exercise_Type'] = 'No Exercise'
test_data.loc[test_data['Exercise_Type'] == 'Sleep_Benefit', 'Exercise_Type'] = 'Strenght training'
test_data.loc[test_data['Exercise_Type'] == 'Not Applicable', 'Exercise_Type'] = 'No Exercise'
test_data.loc[test_data['Exercise_Type'] == 'Strength', 'Exercise_Type'] = 'Strength training'


test_data['Exercise_Type'].value_counts()


#anlayse Exercise_Duration 
test_data['Exercise_Duration'].value_counts()


#fixing it
test_data.replace({'Exercise_Duration' : {'45 minutes' : 'More than 30 minutes',
                                            '20 minutes' : 'Less than 30 minutes',
                                           'Less than 6 hours' : 'More than 30 minutes',
                                           '30 minutes to 1 hour' : 'More than 30 minutes',
                                           'Strength training' : 'More than 30 minutes',
                                           '6-8 hours' : 'More than 30 minutes',
                                           'Less than 20 minutes' : 'Less than 30 minutes',
                                           'No Exercise' : 'Not Applicable',
                                           '3-4 Times a Week' : 'More than 30 minutes',
                                           '20 minutes' : 'Less than 30 minutes',
                                           'Less than 6 hours' : 'More than 30 minutes',
                                           'Not Much' : 'Less than 30 minutes',
                                           '1-2 Times a Week' : 'Less than 30 minutes',
                                           '40 minutes' : 'Mora than 30 minutes',
                                           'Strenght training' : 'More than 30 minutes'
                                          
                                          }},inplace = True)


test_data['Exercise_Duration'].value_counts()


#mapping
def mapping_te(x):

    if x == 'Not Applicable':
        return 'No'
    elif x  == 'Less than 30 minutes':
        return 'Short'
    elif x == '30 minutes':
        return 'Moderate'
    elif x  == 'More than 30 minutes':
        return 'Long'
    else:
        pass


test_data['Exercise_Duration'] = test_data['Exercise_Duration'].apply(mapping_te)


#seeing it
test_data['Exercise_Duration'].value_counts()


#analysing Sleep_Hours
test_data['Sleep_Hours'].value_counts()


#mapping
def mapping_sleep_2(x):

    if x == '6-8 hours' or x == '6-8 Times a Week':
        return 'Good'
    elif x  == 'Less than 6 hours':
        return 'Poor'
    elif x == '9-12 hours' or x == 'More than 12 hours' or x == '6-12 hours':
        return 'Over'
    elif x  == '3-4 hours':
        return 'Over'
    elif x == 'Strength training' or x == 'Strenght training':
        return 'Good'
    elif x == 'No Exercise' or x == '20 minutes':
        return 'Poor'
    else:
        pass


test_data['Sleep_Hours'] = test_data['Sleep_Hours'].apply(mapping_sleep_2)


test_data['Sleep_Hours'].value_counts()


#Exercise Benefit exploration
test_data['Exercise_Benefit'].value_counts()


#mapping
def mapping_exe_2(x):

    if x == 'Somewhat':
        return 'Moderate'
    elif x  == 'Not at All' or x == 'Not Much' or x == 'No Exercise':
        return 'No'
    elif x == 'Yes Significantly':
        return 'High'
    elif x  == 'Not Much' or x == 'Strength training' or x == 'Strenght training':
        return 'Low'
    else:
        pass


test_data['Exercise_Benefit'] = test_data['Exercise_Benefit'].apply(mapping_exe_2)


#seeing it
test_data['Exercise_Benefit'].value_counts()


test_data.head()


#missing value count in train data
train_miss = train_data.isna().sum()
plt.figure(figsize = (10,6))
sns.barplot(x = train_miss.values  , y = train_miss.index)
plt.show()
print('**********')
print(train_miss)


#missing values ratio in training data 
print(((train_data.isna().sum())/len(train_data)) * 100)


#missing values in test_data
test_miss = test_data.isna().sum()
plt.figure(figsize = (10,6))
sns.barplot(x = test_miss.values, y = test_miss.index)
plt.show()

print("****************")
print(test_miss)


#Missing value ratios for test data
print(((test_data.isna().sum())/ len(test_data)) * 100)


#understanding duplicates
print(np.any(train_data.duplicated()))
print(np.any(test_data.duplicated()))


#understanding the weight_kg
plt.figure(figsize = (10,6))
sns.displot(data = train_data, x = 'Weight_kg', kind = 'hist', kde=True)
plt.show()


#Analysed all the categroical columns
count  = ['Age', 'PCOS', 'Hormonal_Imbalance','Hyperandrogenism', 'Hirsutism', 'Conception_Difficulty',
       'Insulin_Resistance', 'Exercise_Frequency', 'Exercise_Type',
       'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']


fig, axes = plt.subplots(4, 3, figsize=(12, 14))  # Adjust grid size based on the number of columns
axes = axes.flatten()  # Flatten the 2D array to 1D for easy iteration

for i, col in enumerate(count):
    sns.countplot(data=train_data, x=col, ax=axes[i])
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels

plt.tight_layout()
plt.show()


#all categorical columns with respective to age
count  = ['PCOS', 'Hormonal_Imbalance','Hyperandrogenism', 'Hirsutism', 'Conception_Difficulty',
       'Insulin_Resistance', 'Exercise_Frequency', 'Exercise_Type',
       'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']

fig, axes = plt.subplots(4, 3, figsize=(12, 14))  
axes = axes.flatten()  

for i, col in enumerate(count):
    sns.countplot(data=train_data, x='Age',hue = col, ax=axes[i])
    axes[i].set_title(col+"  Analyses with Age")
    axes[i].tick_params(axis='x', rotation=45)  
    
for j in range(len(count), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#all categorical data with respective to pcos
count  = ['Age', 'Hormonal_Imbalance','Hyperandrogenism', 'Hirsutism', 'Conception_Difficulty',
       'Insulin_Resistance', 'Exercise_Frequency', 'Exercise_Type',
       'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']


fig, axes = plt.subplots(4, 3, figsize=(12, 14))  
axes = axes.flatten()  

for i, col in enumerate(count):
    sns.countplot(data=train_data, x=col, hue = 'PCOS' ,ax=axes[i])
    axes[i].set_title(col + " Analyse with respective to PCOS", fontsize=10)
    axes[i].tick_params(axis='x', rotation=45)  


for j in range(len(count), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#Understand how weights effect these columns
count  = ['Age', 'PCOS', 'Hormonal_Imbalance','Hyperandrogenism', 'Hirsutism', 'Conception_Difficulty',
       'Insulin_Resistance', 'Exercise_Frequency', 'Exercise_Type',
       'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']

fig, axes = plt.subplots(4, 3, figsize=(12, 14))  
axes = axes.flatten()  

for i, col in enumerate(count):
    sns.histplot(data=train_data, x='Weight_kg', bins = 6, hue = col, multiple = 'dodge', ax=axes[i])
    axes[i].set_title(col + " Effects on Weights")
    axes[i].tick_params(axis='x', rotation=45)  

plt.tight_layout()
plt.show()


#Analysing columns with respective to Hormonal_Imbalance
count  = ['Age', 'PCOS','Hyperandrogenism', 'Hirsutism', 'Conception_Difficulty',
       'Insulin_Resistance', 'Exercise_Frequency', 'Exercise_Type',
       'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']

fig, axes = plt.subplots(4, 3, figsize=(12, 14))  
axes = axes.flatten()  

for i, col in enumerate(count):
    sns.countplot(data=train_data, x='Hormonal_Imbalance', hue = col, ax=axes[i])
    axes[i].set_title(col+ " Effect on Hormonal_Imbalance", fontsize = 10)
    axes[i].tick_params(axis='x', rotation=45)  

for j in range(len(count), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#Analysing columns with respective to Hyperandrogenism
count  = ['Age', 'PCOS','Hormonal_Imbalance', 'Hirsutism', 'Conception_Difficulty',
       'Insulin_Resistance', 'Exercise_Frequency', 'Exercise_Type',
       'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']

fig, axes = plt.subplots(4, 3, figsize=(12, 14))  
axes = axes.flatten()  

for i, col in enumerate(count):
    sns.countplot(data=train_data, x='Hyperandrogenism', hue = col, ax=axes[i])
    axes[i].set_title(col+ " Effect on Hyperandrogenism", fontsize = 10)
    axes[i].tick_params(axis='x', rotation=45)  

for j in range(len(count), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#Analysing columns with respective to Hirsutism
count  = ['Age', 'PCOS','Hormonal_Imbalance', 'Hyperandrogenism', 'Conception_Difficulty',
       'Insulin_Resistance', 'Exercise_Frequency', 'Exercise_Type',
       'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']

fig, axes = plt.subplots(4, 3, figsize=(12, 14))  
axes = axes.flatten()  

for i, col in enumerate(count):
    sns.countplot(data=train_data, x='Hirsutism', hue = col, ax=axes[i])
    axes[i].set_title(col+ " Effect on Hirsutism", fontsize = 10)
    axes[i].tick_params(axis='x', rotation=45)  

for j in range(len(count), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#Analysing columns with respective to Conception_Difficulty
count  = ['Age', 'PCOS','Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism',
       'Insulin_Resistance', 'Exercise_Frequency', 'Exercise_Type',
       'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']

fig, axes = plt.subplots(4, 3, figsize=(12, 14))  
axes = axes.flatten()  

for i, col in enumerate(count):
    sns.countplot(data=train_data, x='Conception_Difficulty', hue = col, ax=axes[i])
    axes[i].set_title(col+ " Effect on Conception_Difficulty", fontsize = 10)
    axes[i].tick_params(axis='x', rotation=45)  

for j in range(len(count), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#Analysing columns with respective to Insulin_Resistance
count  = ['Age', 'PCOS','Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism',
       'Conception_Difficulty', 'Exercise_Frequency', 'Exercise_Type',
       'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']

fig, axes = plt.subplots(4, 3, figsize=(12, 14))  
axes = axes.flatten()  

for i, col in enumerate(count):
    sns.countplot(data=train_data, x='Insulin_Resistance', hue = col, ax=axes[i])
    axes[i].set_title(col+ " Effect on Insulin_Resistance", fontsize = 10)
    axes[i].tick_params(axis='x', rotation=45)  

for j in range(len(count), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#Analysing columns with respective to Exercise_Frequency
count  = ['Age', 'PCOS','Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism',
       'Conception_Difficulty', 'Insulin_Resistance', 'Exercise_Type',
       'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']

fig, axes = plt.subplots(4, 3, figsize=(12, 14))  
axes = axes.flatten()  

for i, col in enumerate(count):
    sns.countplot(data=train_data, x='Exercise_Frequency', hue = col, ax=axes[i])
    axes[i].set_title(col+ " Effect on Exercise_Frequency", fontsize = 10)
    axes[i].tick_params(axis='x', rotation=45)  

for j in range(len(count), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



#Analysing columns with respective to Exercise_Type	
count  = ['Age', 'PCOS','Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism',
       'Conception_Difficulty', 'Insulin_Resistance', 'Exercise_Frequency',
       'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']

fig, axes = plt.subplots(4, 3, figsize=(12, 14))  
axes = axes.flatten()  

for i, col in enumerate(count):
    sns.countplot(data=train_data, x='Exercise_Type', hue = col, ax=axes[i])
    axes[i].set_title(col+ " Effect on Exercise_Type", fontsize = 10)
    axes[i].tick_params(axis='x', rotation=45)  

for j in range(len(count), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



#Analysing columns with respective to Exercise_Duration	
count  = ['Age', 'PCOS','Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism',
       'Conception_Difficulty', 'Insulin_Resistance', 'Exercise_Frequency',
       'Exercise_Type', 'Sleep_Hours', 'Exercise_Benefit']

fig, axes = plt.subplots(4, 3, figsize=(12, 14))  
axes = axes.flatten()  

for i, col in enumerate(count):
    sns.countplot(data=train_data, x='Exercise_Duration', hue = col, ax=axes[i])
    axes[i].set_title(col+ " Effect on Exercise_Duration", fontsize = 10)
    axes[i].tick_params(axis='x', rotation=45)  

for j in range(len(count), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



#Analysing columns with respective to Sleep_Hours		
count  = ['Age', 'PCOS','Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism',
       'Conception_Difficulty', 'Insulin_Resistance', 'Exercise_Frequency',
       'Exercise_Type', 'Exercise_Duration', 'Exercise_Benefit']

fig, axes = plt.subplots(4, 3, figsize=(12, 14))  
axes = axes.flatten()  

for i, col in enumerate(count):
    sns.countplot(data=train_data, x='Sleep_Hours', hue = col, ax=axes[i])
    axes[i].set_title(col+ " Effect on Sleep_Hours", fontsize = 10)
    axes[i].tick_params(axis='x', rotation=45)  

for j in range(len(count), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#Analysing columns with respective to Exercise_Benefit		
count  = ['Age', 'PCOS','Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism',
       'Conception_Difficulty', 'Insulin_Resistance', 'Exercise_Frequency',
       'Exercise_Type', 'Exercise_Duration', 'Sleep_Hours']

fig, axes = plt.subplots(4, 3, figsize=(12, 14))  
axes = axes.flatten()  

for i, col in enumerate(count):
    sns.countplot(data=train_data, x='Exercise_Benefit', hue = col, ax=axes[i])
    axes[i].set_title(col+ " Effect on Exercise_Benefit", fontsize = 10)
    axes[i].tick_params(axis='x', rotation=45)  

for j in range(len(count), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


#Outlier Analysis for the weights
sns.catplot(data = train_data, y = 'Weight_kg', kind = 'box')


#droping missing values
t_data = train_data.dropna()


#handling outliers
q1,q3 = np.quantile(t_data['Weight_kg'],[0.25,0.75])
iqr   = q3 - q1

lower_boundary  = q1 - (1.5 * iqr)
upper_boundary  = q3 + (1.5 * iqr)

t_data.loc[t_data['Weight_kg'] > upper_boundary, ['Weight_kg'] ] = upper_boundary
t_data.loc[t_data['Weight_kg'] < lower_boundary, ['Weight_kg'] ] = lower_boundary

sns.catplot(data = t_data, y = 'Weight_kg', kind  = 'box')


#representing all categorical data in numerical 
from sklearn.preprocessing import LabelEncoder

count = ['Age', 'PCOS', 'Hormonal_Imbalance',
       'Hyperandrogenism', 'Hirsutism', 'Conception_Difficulty',
       'Insulin_Resistance', 'Exercise_Frequency', 'Exercise_Type',
       'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']
for i in count:
    
    le  = LabelEncoder()
    t_data[i] = le.fit_transform(t_data[i])


#understanding the distribution
sns.displot(data = t_data, x = 'Weight_kg', kde = True)

from scipy.stats import skew
print(skew(t_data['Age']))


t_data.head()


#making the test data id column good
te_data  = test_data.reset_index()
te_data  = te_data.drop(['ID'], axis = 1)


te_data.rename(columns  = {'index' : 'ID'}, inplace = True)


#making weight to be in good format
te_data.loc[(te_data['Weight_kg'] == 'Strength training') | (te_data['Weight_kg'] == 'Strenght training'), ['Weight_kg']] = np.nan
te_data.loc[te_data['Weight_kg'] == 'No Exercise', ['Weight_kg']] = np.nan


#converting from str to float
te_data['Weight_kg'] = te_data['Weight_kg'].astype('float')


#handling missing values
from sklearn.impute import SimpleImputer
simp  =  SimpleImputer(missing_values = np.nan, strategy= 'median')
te_data[['Weight_kg']] = simp.fit_transform(te_data[['Weight_kg']])


#handling np.nan
from sklearn.impute import SimpleImputer
ob  =  SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

for k in te_data.select_dtypes(include = ['object']):
    te_data[[k]] = ob.fit_transform(te_data[[k]])


#handling None
from sklearn.impute import SimpleImputer
ob  =  SimpleImputer(missing_values = None, strategy = 'most_frequent')

for k in te_data.select_dtypes(include = ['object']):
    te_data[[k]] = ob.fit_transform(te_data[[k]])


te_data.isna().sum()


te_data.info()


#just removing the ID column 
te_data_2 = te_data.drop(['ID'],axis = 1)


##convert all the categorical data into numerical
from sklearn.preprocessing import LabelEncoder
leb = LabelEncoder()
for k in te_data_2.select_dtypes(include = ['object']):
    te_data_2[k] = leb.fit_transform(te_data[k])


te_data_2.head()


X = t_data.drop(['ID','PCOS'], axis  = 1)
y = t_data['PCOS']


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)


#logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(x_train,y_train)


y_pred = lr.predict(x_test)
y_pred


y_prob  = lr.predict_proba(x_test)


from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_prob[:,1]))


#Implementing the Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)


#test the model
rf_prob = rfc.predict_proba(x_test)
roc_auc_score(y_test,rf_prob[:,1])


#Implementing the Adaboost classifier
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
ada.fit(x_train,y_train)
ada_prob = ada.predict_proba(x_test)
roc_auc_score(y_test,ada_prob[:,1])


#Implementing the BaggingClassifier
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier()
bagging.fit(x_train,y_train)
bagging_prob = bagging.predict_proba(x_test)
roc_auc_score(y_test,bagging_prob[:,1])


#Implementing the ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
extra = ExtraTreesClassifier()
extra.fit(x_train,y_train)
extra_prob = extra.predict_proba(x_test)
roc_auc_score(y_test,extra_prob[:,1])


#implementing SVM
from sklearn.svm import LinearSVC
svc =  LinearSVC()
svc.fit(x_train,y_train)
svc_pred = svc.predict(x_test)
accuracy_score(svc_pred, y_test)


#Implementing the GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gdb = GradientBoostingClassifier()
gdb.fit(x_train,y_train)
gdb_prob = gdb.predict_proba(x_test)
roc_auc_score(y_test,gdb_prob[:,1])


#Implementing the decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_prob = dt.predict_proba(x_test)
roc_auc_score(y_test,dt_prob[:,1])


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# List of models and names
models = [lr, rfc, ada, bagging, extra, svc, gdb,dt]
names = ['Logistic Regression',
         'Random Forest Classifier',
         'AdaBoosting Classifier',
         'Bagging Classifier',
         'ExtraTrees Classifier',
         'Support Vector Machines',
         'Gradient Boosting Classifier',
        'Decision Tree Classifier']

# Create subplots
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10, 14))
ax = ax.flatten()

# Loop through models
for i, model in enumerate(models):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)  # Use y_test for correct labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax[i])
    ax[i].set_title(f'The confusion matrix for {names[i]}', fontsize=12)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Add more space between rows if needed

# Save the figure as an image file (e.g., PNG)
plt.savefig('confusion_matrices.png', dpi=300) 

plt.show()



#calculating the prediction probabailities  using logisitic regression
predictions = lr.predict_proba(te_data_2)
predictions


Values = pd.DataFrame(predictions[:,1])
Values


Values.rename(columns  = {0 : 'PCOS'}, inplace =True)


output  = pd.concat([te_data['ID'], Values['PCOS']],axis = 1)
output


output.to_csv('submission.csv', index=False)




