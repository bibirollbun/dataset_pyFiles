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


data=pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')


data.info()


data.isnull().sum()


#dropping the columns with too much missing values
train=data.drop(['Occupation','Previous Claims','Credit Score','Number of Dependents','id','Policy Start Date'],axis=1)
columns_to_check = ['Vehicle Age', 'Insurance Duration']  # Specify the columns you want to check for missing values
train = train.dropna(subset=columns_to_check)


#splitting the training and testing data
X_train=train.drop('Premium Amount', axis=1)
y_train=train['Premium Amount']


#filling in the missing data
X_train['Gender'] = X_train['Gender'].fillna(X_train['Gender'].mode())
X_train['Annual Income'] = X_train['Annual Income'].fillna(X_train['Annual Income'].median())
X_train['Marital Status'] = X_train['Marital Status'].fillna(X_train['Marital Status'].mode()[0])
X_train['Customer Feedback'] = X_train['Customer Feedback'].fillna('Not Available')
X_train['Health Score'] = X_train['Health Score'].interpolate(method='linear')
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())



X_train.head()



import seaborn as sns
sns.countplot(data=X_train, x='Policy Type', order=X_train['Policy Type'].value_counts().index, hue=X_train['Smoking Status'])



sns.countplot(data=X_train, x='Smoking Status', order=X_train['Smoking Status'].value_counts().index, hue=X_train['Gender'])



sns.countplot(data=X_train, x='Property Type', order=X_train['Property Type'].value_counts().index)



subset=X_train[['Age','Health Score','Vehicle Age','Insurance Duration']]
subset=subset.sample(n=5000, random_state=42)
sns.pairplot(subset)


X_test=pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder 
from sklearn.pipeline import Pipeline

education_order = sorted(X_train['Education Level'].dropna().unique())  # Sort unique values
policy_order = sorted(X_train['Policy Type'].dropna().unique())         # Sort unique values
feedback_order = sorted(X_train['Customer Feedback'].dropna().unique())
exercise_order = sorted(X_train['Exercise Frequency'].dropna().unique())

nominal_transformer = OneHotEncoder(handle_unknown='ignore')  # For nominal columns
ordinal_transformer = OrdinalEncoder(categories=[education_order, policy_order, feedback_order, exercise_order])  # For ordinal columns
numerical_transformer = StandardScaler()  # For scaling numerical column
preprocessor = ColumnTransformer(
    transformers=[
        # Apply OneHotEncoder to nominal columns
        ('nominal', nominal_transformer, ['Gender', 'Marital Status', 'Location', 'Smoking Status', 'Property Type']),
        
        # Apply OrdinalEncoder to ordinal columns
        ('ordinal', ordinal_transformer, ['Education Level', 'Policy Type', 'Customer Feedback', 'Exercise Frequency']),
        
        # Apply StandardScaler to Annual Income
        ('numerical', numerical_transformer, ['Annual Income']),
    ],
    remainder='passthrough'  # Keeps any other columns untouched (optional)
)

# Create a pipeline with the preprocessor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Fit and transform the dataset
X_train_transformed = pipeline.fit_transform(X_train)



from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=5, random_state=42,oob_score=True)
regr.fit(X_train_transformed, y_train)


X_test=X_test.drop(['Occupation','Previous Claims','Credit Score','Number of Dependents','id','Policy Start Date'],axis=1)



X_test['Annual Income'] = X_test['Annual Income'].fillna(X_test['Annual Income'].median())
X_test['Marital Status'] = X_test['Marital Status'].fillna(X_test['Marital Status'].mode()[0])
X_test['Customer Feedback'] = X_test['Customer Feedback'].fillna('Not Available')
X_test['Health Score'] = X_test['Health Score'].interpolate(method='linear')
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].median())



X_test.dropna(inplace=True)
X_test.isnull().sum()


education_order = sorted(X_test['Education Level'].dropna().unique())  # Sort unique values
policy_order = sorted(X_test['Policy Type'].dropna().unique())         # Sort unique values
feedback_order = sorted(X_test['Customer Feedback'].dropna().unique())
exercise_order = sorted(X_test['Exercise Frequency'].dropna().unique())

nominal_transformer = OneHotEncoder(handle_unknown='ignore')  # For nominal columns
ordinal_transformer = OrdinalEncoder(categories=[education_order, policy_order, feedback_order, exercise_order])  # For ordinal columns
numerical_transformer = StandardScaler()  # For scaling numerical column
preprocessor = ColumnTransformer(
    transformers=[
        # Apply OneHotEncoder to nominal columns
        ('nominal', nominal_transformer, ['Gender', 'Marital Status', 'Location', 'Smoking Status', 'Property Type']),
        
        # Apply OrdinalEncoder to ordinal columns
        ('ordinal', ordinal_transformer, ['Education Level', 'Policy Type', 'Customer Feedback', 'Exercise Frequency']),
        
        # Apply StandardScaler to Annual Income
        ('numerical', numerical_transformer, ['Annual Income']),
    ],
    remainder='passthrough'  # Keeps any other columns untouched (optional)
)

# Create a pipeline with the preprocessor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Fit and transform the dataset
X_test_transformed = pipeline.fit_transform(X_test)



from sklearn.metrics import mean_squared_error, r2_score

# Access the OOB Score
oob_score = regr.oob_score_
print(f'Out-of-Bag Score: {oob_score}')

# Making predictions on the same data or new data
predictions = regr.predict(X_train_transformed)

# Evaluating the model
mse = mean_squared_error(y_train, predictions)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_train, predictions)
print(f'R-squared: {r2}')

