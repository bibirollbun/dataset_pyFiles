# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import Train Dataset
df_train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv', index_col = 'id')

# Import Test Dataset
df_test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv', index_col = 'id')

# Display Dataset
print('Train Dataset')
display(df_train.tail(5))

print("Test Dateset")
display(df_test.head(5))


# concat all dataset
df_all = pd.concat((df_train.loc[:,'Gender':'MTRANS'],
                  df_test.loc[:,'Gender':'MTRANS']))

df_all.head()


# Statistical Information Train Dataset
display(df_all.info())
print('--------------------')
print('Shape of the Dataset')
display(df_all.shape)
print('--------------------')
print('Dataset Description')
display(df_all.describe())


# Check Null data
print('Total Null Dataset: ')
display(df_all.isnull().sum())

print(' ')

# Check Duplicated Data
print('Total Duplicated Dataset: ')
df_all.duplicated().sum()


# create plot function
import matplotlib.pylab as plt
import seaborn as sns
sns.set(style = 'whitegrid')
def pie_plot(data, feature, hue = None):
    if data[feature].dtypes == 'Object' or data[feature].dtypes == 'O':
        fig = plt.figure(figsize = (6,6))
        val = data[feature].value_counts().values
        cat = data[feature].value_counts().index
        plt.pie(val, labels = cat, autopct = '%1.1f%%')
        plt.title(f'Distribution Plot of {feature}')
        fig.tight_layout()
        plt.show()
    else:
        fig = plt.figure(figsize = (8,5))
        sns.kdeplot(data, x = feature, hue = hue, fill = True)
        plt.title(f"Distribution Plot of Obesity based on {feature} ")
        fig.tight_layout()
        plt.show()


pie_plot(df_train, 'NObeyesdad')


# FAVC Plot
pie_plot(df_all, 'FAVC')


# FVCV pie plot
print('#Train Dataset')
display(pie_plot(df_train, 'FCVC', 'NObeyesdad')) #Train Dataset
print(" ")
print('#All Dataset')
display(pie_plot(df_all, 'FCVC')) # All Dataset


# Gender Distribution
print('All Dataset')
display(pie_plot(df_all, 'Gender'))# All Dataset
print(' ')
print('Train Dataset')
display(pie_plot(df_train, 'Gender')) # Train Dataset


# NCP Distribution
print('Data Train')
display(pie_plot(df_train, 'NCP', 'NObeyesdad')) # Train Dataset
print(" ")
print('All Dataset')
display(pie_plot(df_all, 'NCP')) # All Dataset


# CAEC Distribution
print('Train Data')
display(pie_plot(df_train, 'CAEC'))
print(" ")
print('All Data')
display(pie_plot(df_all, 'CAEC'))


# SMOKE Distribution
display(pie_plot(df_all, 'SMOKE'))


# SCC Data
print('Train Data')
display(pie_plot(df_train, 'SCC'))


print('Data Train')
display(pie_plot(df_train, 'CH2O', 'NObeyesdad'))
print(" ")
print('ALL Data')
display(pie_plot(df_all, 'CH2O'))


print('Data Train')
display(pie_plot(df_train, 'FAF', 'NObeyesdad'))
print(" ")
print('All Data')
display(pie_plot(df_all, 'FAF'))



print('Data Train')
display(pie_plot(df_train, 'TUE', 'NObeyesdad'))
print(" ")
print("All Data")
display(pie_plot(df_all, 'TUE'))


print('Data Train')
display(pie_plot(df_all, 'SCC'))


display(pie_plot(df_train, 'MTRANS'))


# Weight
pie_plot(df_train, 'Weight', 'NObeyesdad')


# Height
pie_plot(df_train, 'Height', 'NObeyesdad')


pie_plot(df_train, 'family_history_with_overweight')


# create body mass index
df_train['bmi'] = df_train['Weight']/(df_train['Height']*df_train['Height'])
df_all['bmi'] = df_all['Weight']/(df_all['Height']*df_all['Height'])


# Plot body mass index
pie_plot(df_train, 'bmi', 'NObeyesdad')


# grouping obesity based on its bmi score
bmi_obese = df_train.groupby('NObeyesdad')['bmi'].median().sort_values(ascending = True)
val = bmi_obese.values
ind = bmi_obese.index

# create barplot
fig = plt.figure(figsize = (8,4))
ax = sns.barplot(x = val, y = ind)
for container in ax.containers:
    ax.bar_label(container, fmt = '%.00f', label_type = 'edge', padding = 3)
plt.title('Obesity based on BMI (Median)')
plt.ylabel('Obesity Category')
plt.xlabel('Body Mass Index')
plt.show()


age_obese = df_train.groupby('NObeyesdad')['Age'].mean().sort_values(ascending = False)
val_age = age_obese.values
ind_age = age_obese.index

fig = plt.figure(figsize = (8,4))
ax = sns.barplot(x = val_age, y = ind_age)
for container in ax.containers:
    ax.bar_label(container, fmt = '%.00f', label_type = 'edge', padding = 3)
plt.title('Obesity based on Age (Mean)')
plt.ylabel('Obesity Category')
plt.xlabel('Age')
plt.show()


trans_data = df_train.groupby(['MTRANS', 'NObeyesdad'])['NObeyesdad'].count().unstack().fillna(0)
fig = plt.figure(figsize = (17,8))
sns.heatmap(data = trans_data, annot = True, cmap = 'YlGnBu', linewidth = 0.8, fmt = 'g')

#Customize title and label
plt.title('Transportation Type based on Obesity Category', fontsize = 18)
plt.xlabel("Obesity Category", fontsize = 15)
plt.ylabel('Transportation Type', fontsize = 15)
plt.xticks(rotation = 0, fontsize = 13)
plt.yticks(rotation = 0, fontsize = 13)
fig.tight_layout()
plt.show()


avg_bmi_gender = df_train.groupby('Gender')['bmi'].mean().reindex(['Male', 'Female'])
avg_bmi_val = avg_bmi_gender.values
avg_bmi_index = avg_bmi_gender.index

fig, axs = plt.subplots(1,2,figsize = (10,5), sharey=True)
sns.barplot(x = avg_bmi_val, y = avg_bmi_index, ax = axs[0])
axs[0].set_title('Average BMI based on Gender')
sns.violinplot(data = df_train, x = 'bmi', y = 'Gender', ax = axs[1])
axs[1].set_title('BMI Distribution Plot based on Gender')
plt.suptitle('BMI Analysis by Gender')
fig.tight_layout()
plt.show()


gender_data = df_train.groupby(['NObeyesdad', 'Gender'])['Gender'].count().unstack().fillna(0)

#Customize title and label
sns.heatmap(data = gender_data, annot = True, cmap = 'YlGnBu', linewidth = 0.8, fmt = 'g')
plt.title('Obesity Category based on Gender', fontsize = 18)
plt.xlabel("Gender", fontsize = 15)
plt.ylabel('Obesity Category', fontsize = 15)
plt.xticks(rotation = 0, fontsize = 13)
plt.yticks(rotation = 0, fontsize = 13)
fig.tight_layout()
plt.show()


obese_alc = df_train.groupby(['NObeyesdad', 'CALC'])['CALC'].count().unstack().fillna(0)

# create heatmap
sns.heatmap(data = obese_alc, annot = True, cmap = 'YlGnBu', linewidth = 0.5, fmt = 'g')
plt.title('Alcohol Consumption Frequency based on Obesity')
plt.xlabel("Alcohol Consumption", fontsize = 15)
plt.ylabel('Obesity Category', fontsize = 15)
plt.xticks(rotation = 0, fontsize = 13)
plt.yticks(rotation = 0, fontsize = 13)
fig.tight_layout()
plt.show()


# Using Contingency Table (Pandas Cross Tab)
gender_obese = pd.crosstab(df_train['Gender'], df_train['NObeyesdad'], normalize = 'index')
gender_obese


# using Contingency Table + Chi-Squared Test
from scipy.stats import chi2_contingency
import warnings
def crosstab_chi2 (features ,data = df_train, target = 'NObeyesdad'):
    p_val = {}
    for cols in features:
        try:
            cross_tab = pd.crosstab(data[cols], data[target])
            chi2, p, dof, expected = chi2_contingency(cross_tab)
            p_val[cols] = p
        except Exception as e:
            p_val[cols] = f"Error: {e}"
    return pd.Series(p_val, name = 'p_value')

data_objects = df_train.select_dtypes(include = 'object')
crosstab_chi2(data_objects.columns)


df_all


# Object in the train dataset

def encoded_data(data, drop_column = None):
    copy_data = data.copy()
    objects = copy_data.select_dtypes(include = 'object')
    if 'NObeyesdad' in objects.columns:
        copy_data.drop(columns = ['NObeyesdad'], inplace = True)
        objects = copy_data.select_dtypes(include = 'object')
    else:
        objects = copy_data.select_dtypes(include = "object")
    data = pd.get_dummies(data, columns = objects.columns, drop_first = True)
    return data

train_enc = encoded_data(df_train)
test_enc = encoded_data(df_test)
all_enc = encoded_data(df_all)



# Check Missing Data for all dataset
missing_train = set(test_enc.columns) - set(train_enc.columns)
missing_test = set(train_enc.columns) - set(test_enc.columns)
missing_all = set(train_enc.columns) - set(all_enc.columns)

print('Missing in train data: ', missing_train)
print('Missing in test data: ', missing_test)
print('Missing in all data: ', missing_all)


all_enc.iloc[:len(train_enc)]


# importlogistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

X = all_enc.iloc[:len(train_enc)]
y = train_enc.NObeyesdad

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)
params = {"fit_intercept": True,
         "max_iter":200,
         "penalty": 'l2',
         "C": 0.01}
logreg = LogisticRegression(**params)

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print(classification_report(y_pred, y_test))


pred_data = all_enc.iloc[len(train_enc):]
pred = logreg.predict(pred_data)
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e2/sample_submission.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')
submission = pd.DataFrame({'id':test.id, 'NObeyesdad':pred})
submission.to_csv('/kaggle/working/submission.csv', index = False)


