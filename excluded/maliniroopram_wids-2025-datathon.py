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


import pandas as pd
from sklearn.model_selection import train_test_split


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


sample_submission = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')
test_connectome = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')
test_cat = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')
test_quant = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')
train_connectome = pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv')
train_solutions = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx')
train_cat = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx')
train_quant = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx')


train_cat.head()


train_cat.shape


train_cat.dtypes


train_cat.isnull().sum()


# Fill NAs

for col in train_cat.columns:
    if train_cat[col].isna().sum() > 0:  # Check if the column has NaN values
        if train_cat[col].dtype in ['float64', 'int64']:  # Ensure it's numeric
            train_cat[col] = train_cat[col].fillna(train_cat[col].mean())  # Avoid inplace
        else:
            print(f"Skipping non-numeric column: {col}")


train_cat.isnull().sum()


train_quant.head()


train_quant.shape


train_quant.dtypes


train_quant.isnull().sum()


# Fill NAs

for col in train_quant.columns:
    if train_quant[col].isna().sum() > 0:  # Check if the column has NaN values
        if train_quant[col].dtype in ['float64', 'int64']:  # Ensure it's numeric
            train_quant[col] = train_quant[col].fillna(train_quant[col].mean())  # Avoid inplace
        else:
            print(f"Skipping non-numeric column: {col}")


train_quant.isnull().sum()


train_connectome.head()


train_connectome.shape


train_connectome.dtypes


train_connectome.isnull().sum()


train_merged = pd.merge(train_cat, train_quant, on='participant_id')
train_merged = pd.merge(train_merged, train_connectome, on='participant_id')
train_merged = pd.merge(train_merged, train_solutions, on='participant_id')
train_merged.shape


#Code from WiDS Workshop
sdq_vars = [col for col in train_merged.columns if col.startswith('SDQ')]


plt.figure(figsize=(14, 10))
for i, var in enumerate(sdq_vars):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='ADHD_Outcome', y=var, data=train_merged)
    plt.title(var.split('SDQ_SDQ_')[1])
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 10))
for i, var in enumerate(sdq_vars):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='Sex_F', y=var, data=train_merged)
    plt.title(var.split('SDQ_SDQ_')[1])
plt.tight_layout()
plt.show()


apq_vars = [col for col in train_merged.columns if 'APQ_P_APQ_P' in col]


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, var in enumerate(apq_vars):
    sns.boxplot(x='ADHD_Outcome', y=var, data=train_merged, ax=axes[i])
    var_name = var.replace('APQ_P_APQ_P_', '')
    axes[i].set_title(f'{var_name} by ADHD Status')
    axes[i].set_xlabel('ADHD Diagnosis (1=Yes, 0=No)')

# Add descriptive labels for ADHD status charts
labels = [
    "CP: Physical discipline differences",
    "ID: Discipline consistency variations",
    "INV: Parental involvement by ADHD status",
    "OPD: Alternative discipline strategies",
    "PM: Monitoring practices across groups",
    "PP: Positive reinforcement patterns"
]

for i, label in enumerate(labels):
    axes[i].set_ylabel(label, rotation=90, ha='right')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, var in enumerate(apq_vars):
    sns.boxplot(x='Sex_F', y=var, data=train_merged, ax=axes[i])
    var_name = var.replace('APQ_P_APQ_P_', '')
    axes[i].set_title(f'{var_name} by Sex')
    axes[i].set_xlabel('Sex (1=Female, 0=Male)')

# Add descriptive labels for sex comparison charts
labels = [
    "CP: Physical discipline by gender",
    "ID: Discipline consistency across sexes",
    "INV: Parental involvement differences",
    "OPD: Alternative discipline strategies",
    "PM: Monitoring practices by sex",
    "PP: Positive reinforcement patterns"
]

for i, label in enumerate(labels):
    axes[i].set_ylabel(label, rotation=90, ha='right')

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1,2, figsize=(5,3))

for col, ax in zip(train_solutions.drop('participant_id',axis=1), axs):
    counts = train_solutions[col].value_counts()
    ax.pie(counts, labels=counts.index,
           autopct='%1.1f%%',
           startangle=90)
    ax.set_title(f'{col}')


female_1 = train_merged[(train_merged['Sex_F']==1)]
female_1.shape


#Concatenate female dataframe to training data
train_oversample = pd.concat([female_1[100:], train_merged], ignore_index=True)
train_oversample.shape


train_oversample['Sex_F'].value_counts()


#Create new solutions data that includes the duplicate rows
oversample_solutions = train_oversample[['ADHD_Outcome', 'Sex_F']]
oversample_solutions.shape


columns = ['APQ_P_APQ_P_CP',
 'APQ_P_APQ_P_ID',
 'APQ_P_APQ_P_INV',
 'APQ_P_APQ_P_OPD',
 'APQ_P_APQ_P_PM',
 'APQ_P_APQ_P_PP',
 'ColorVision_CV_Score',
 'EHQ_EHQ_Total',
 'MRI_Track_Age_at_Scan']


train_oversample.drop(columns=columns, inplace=True)
train_oversample.shape


test_merged = pd.merge(test_cat, test_quant, on='participant_id')
test_merged = pd.merge(test_merged, test_connectome, on='participant_id')
test_merged.shape


#Fill NAs of test data

for col in test_merged.columns:
    if test_merged[col].isna().sum() > 0:  # Check if the column has NaN values
        if test_merged[col].dtype in ['float64', 'int64']:  # Ensure it's numeric
            test_merged[col] = test_merged[col].fillna(test_merged[col].mean())  # Avoid inplace
        else:
            print(f"Skipping non-numeric column: {col}")


#Drop columns
test_merged.drop(columns=columns, inplace=True)
test_merged.shape


# Load data
x_train = train_oversample
test = test_merged

y_train = oversample_solutions

# Set index
x_train.set_index('participant_id', inplace=True)
test.set_index('participant_id', inplace=True)

x_train.drop(columns=['ADHD_Outcome', 'Sex_F'], inplace=True)


print(x_train.shape, test.shape, y_train.shape)


#Split the data into train and test sets
X_train, X_test, y_train, y_test= train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


from sklearn.preprocessing import StandardScaler

#Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score


#Set model and parameters
multioutput_nn = MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, solver='sgd'), n_jobs=-1)


#Fit training data to model
multioutput_nn.fit(X_train_scaled, y_train)


#Get predictions
multioutput_pred_nn = multioutput_nn.predict(X_test_scaled)


#F1 ADHD, Sex
f1 = f1_score(y_test, multioutput_pred_nn, average=None)
print('f1: ', f1)


#Average F1
f1 = f1_score(y_test, multioutput_pred_nn, average='micro')
print('f1: ', f1)


final_pred_nn = multioutput_nn.predict(test)


sample_submission['ADHD_Outcome'] = final_pred_nn[:,0]
sample_submission['Sex_F'] = final_pred_nn[:,1]
sample_submission.to_csv('submission2.csv',index=False)

