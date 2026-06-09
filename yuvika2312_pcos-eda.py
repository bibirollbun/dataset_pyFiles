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


full_test = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')
full_train= pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')
full_test.head(),full_train.head()


full_train.info()


full_train.describe()


full_train.nunique()


full_train.isna().sum()


full_train.duplicated().sum()


full_train['PCOS'].value_counts()


full_train_pcos = full_train[full_train['PCOS']=='Yes']


import matplotlib.pyplot as plt
category_counts = full_train_pcos['Exercise_Duration'].value_counts()

plt.figure(figsize=(12, 5))
plt.bar(category_counts.index, category_counts.values, color='skyblue')
plt.xlabel('Duration')
plt.ylabel('numbe of people')
plt.title('Bar Chart of Exercise Duration for people with pcos')
plt.show()


category_counts = full_train_pcos['Conception_Difficulty'].value_counts()

plt.figure(figsize=(6,4))
plt.bar(category_counts.index, category_counts.values, color='skyblue')
plt.xlabel('Duration')
plt.ylabel('numbe of people')
plt.title('Bar Chart of Conception Difficulty for people with pcos')
plt.show()


data = full_train_pcos['Age'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90)
plt.title("Age Distribution for people with PCOS")


full_train_pcos.nunique()


full_train_pcos['Hyperandrogenism'].value_counts()


full_train_pcos['Insulin_Resistance'].value_counts()


full_train_pcos['Hirsutism'].value_counts()


crosstab = pd.crosstab(full_train['Hirsutism'], full_train['PCOS'])

# Stacked bar plot
crosstab.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='viridis')
plt.title('Stacked Bar Plot of Hirsutism and PCOS')
plt.xlabel('Hirsutism')
plt.ylabel('Count')
plt.legend(title='PCOS')
plt.show()



crosstab = pd.crosstab(full_train['Age'], full_train['PCOS'])

# Stacked bar plot
crosstab.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='viridis')
plt.title('Stacked Bar Plot of Insulin Resistance and PCOS')
plt.xlabel('Insulin Resistance')
plt.ylabel('Count')
plt.legend(title='PCOS')
plt.show()



full_train['Age'].value_counts(),full_test['Age'].value_counts()


import seaborn as sns
crosstab = pd.crosstab(full_train['Exercise_Duration'], full_train['PCOS'])

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(crosstab, annot=True, cmap='coolwarm', fmt='d')
plt.title('Heatmap of Exercise_Duration and PCOS')
plt.xlabel('PCOS')
plt.ylabel('Exercie Duration')
plt.show()


full_train['Exercise_Duration'].value_counts(),full_test['Exercise_Duration'].value_counts()


X = full_train.drop(columns=["PCOS","ID"],axis=1)
full_train['PCOS']=full_train['PCOS'].map({'Yes':1,'No':0})
y=full_train['PCOS']
test_data = full_test.drop(columns=["ID"],axis=1)


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

numerical_features = ['Weight_kg']  
categorical_features = ['Hormonal_Imbalance','Hyperandrogenism','Hirsutism','Conception_Difficulty',
                        'Insulin_Resistance','Exercise_Frequency','Exercise_Type','Exercise_Duration',
                        'Sleep_Hours','Exercise_Benefit','Age']         


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  
    ('scaler', StandardScaler())                
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  
    ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))     
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_train = preprocessor.fit_transform(X)
X_test = preprocessor.transform(test_data)



from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y)


from xgboost import XGBClassifier
model = XGBClassifier(random_state=42)
model.fit(X_train_balanced,y_train_balanced)
predictions =  model.predict_proba(X_test)[:, 1]
output = pd.DataFrame({'ID': full_test.ID, 'PCOS': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")




