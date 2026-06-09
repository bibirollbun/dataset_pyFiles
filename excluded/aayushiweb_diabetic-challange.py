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


train=pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
train.head()


test=pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
test.head()


train.info()


train.duplicated().sum()


test.duplicated().sum()


train.describe().T


numeric_data = train.select_dtypes(include=[np.number]).columns.tolist()


# To specifically select the pandas 'category' dtype
categorical_data = train.select_dtypes(include='category').columns.tolist()


numeric_data


categorical_data


import matplotlib.pyplot as plt
# create a boxplot
boxplot = train.boxplot()
plt.show()


train['gender'].value_counts(dropna=False)





gender                              
ethnicity                           
education_level
income_level
smoking_status                    
employment_status            


train["gender"].tail(20)


categorical_cols = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'employment_status',
    'smoking_status',
]



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)


X_encoded = ct.fit_transform(train)


X_encoded


feature_names = ct.get_feature_names_out()
X_encoded = pd.DataFrame(X_encoded, columns=feature_names)
X_encoded


train.head()


categorical_data = train.select_dtypes(include='object').columns.tolist()
categorical_data


train.columns.tolist()


categorical_data=['gender',
 'ethnicity',
 'education_level',
 'income_level',
 'smoking_status',
 'employment_status']


cols_to_drop = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status'
]

train = train.drop(columns=cols_to_drop)


train


final_df = pd.concat([train, X_encoded], axis=1)


final_df.shape



final_df.columns





# cols_to_drop = [
#     'gender',
#     'ethnicity',
#     'education_level',
#     'income_level',
#     'smoking_status',
#     'employment_status'
# ]

# test = test.drop(columns=cols_to_drop)










