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


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train.shape


print(train.info())
print(test.info())


train.describe()



test.describe()


train.shape


test.shape


train.head()


test.head()


train.isnull().sum()


test.isnull().sum()


train['date'] = pd.to_datetime(train['date'], format="%Y-%m-%d", dayfirst=True)

train['year'] = train['date'].dt.year
train['quarter'] = train['date'].dt.quarter
train['month'] = train['date'].dt.month

train.info()


train.head()


train = train.drop(columns='date')


test['date'] = pd.to_datetime(test['date'], format="%Y-%m-%d", dayfirst=True)

test['year'] = test['date'].dt.year
test['quarter'] = test['date'].dt.quarter
test['month'] = test['date'].dt.month

test.info()


test.head()


test = test.drop(columns='date')


train_id = train['id']
train = train.drop(columns='id')
test_id = test['id']
test = test.drop(columns='id')


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


train.nunique()


test.nunique()


numeric_columns_train = train.select_dtypes(include=['number'])
categorical_columns_train = train.select_dtypes(include=['object'])

numeric_columns_test = test.select_dtypes(include=['number'])
categorical_columns_test = test.select_dtypes(include=['object'])


# Set figure size for better readability
plt.figure(figsize=(15, 30))

# List of colors for the plots
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#8e44ad']

# Plot each categorical column
for i, column in enumerate(categorical_columns_train.columns):
    counts = train[column].value_counts().sort_index()
    plt.subplot((len(categorical_columns_train.columns) + 1) // 2, 2, i + 1)
    sns.barplot(x=counts.index, y=counts.values, color=colors[i % len(colors)])
    plt.title(column.replace('_', ' ').title())
    plt.xlabel(column)
    plt.ylabel('Count')
    # Annotate counts above each bar
    for j, count in enumerate(counts):
        plt.text(j, count + (max(counts) * 0.01), str(count), ha='center', va='bottom', fontsize=9)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# Set figure size for better readability
plt.figure(figsize=(15, 30))

# List of colors for the plots
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#8e44ad']

# Plot each categorical column
for i, column in enumerate(categorical_columns_test.columns):
    counts = test[column].value_counts().sort_index()
    plt.subplot((len(categorical_columns_test.columns) + 1) // 2, 2, i + 1)
    sns.barplot(x=counts.index, y=counts.values, color=colors[i % len(colors)])
    plt.title(column.replace('_', ' ').title())
    plt.xlabel(column)
    plt.ylabel('Count')
    # Annotate counts above each bar
    for j, count in enumerate(counts):
        plt.text(j, count + (max(counts) * 0.01), str(count), ha='center', va='bottom', fontsize=9)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 10))
sns.heatmap(numeric_columns_train.corr(), annot=True, cmap='coolwarm', fmt='.2f')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 10))
sns.heatmap(numeric_columns_test.corr(), annot=True, cmap='coolwarm', fmt='.2f')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


train = train.drop(columns= 'year')
test = test.drop(columns='year')


train.head()


test.head()


# Select numeric columns only

# Set figure size for better readability
plt.figure(figsize=(16, 10))

# Iterate through each numeric column and create boxplot
sns.boxplot(data=train, x='num_sold')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# Select numeric columns only

# Set figure size for better readability
plt.figure(figsize=(16, 10))

# Iterate through each numeric column and create boxplot
sns.kdeplot(data=train, x='num_sold')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer , StandardScaler
from sklearn.pipeline import Pipeline


ft = FunctionTransformer(func=np.log1p)


from sklearn.preprocessing import OneHotEncoder


ohe = OneHotEncoder(drop='first',sparse_output=False)


num_sold_pipeline = Pipeline([
    ('imputer', KNNImputer()),  # Impute missing values first
    ('scaler', StandardScaler())  # Then scale the data
])


num_sold_transformer = ColumnTransformer(transformers=[
    ('num_sold', num_sold_pipeline, ['num_sold']),  # Apply the pipeline to num_sold
    ('encode', OneHotEncoder(), ['country', 'store', 'product'])  # One-hot encoding for categorical features
], remainder='passthrough')


train_transformed = num_sold_transformer.fit_transform(train)


train_transformed = pd.DataFrame(train_transformed,columns = num_sold_transformer.get_feature_names_out())


train_transformed


x = train_transformed.drop(columns='num_sold__num_sold')
y = train_transformed['num_sold__num_sold']


from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=365)


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score ,KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor



model1 = LinearRegression()
model2 = GradientBoostingRegressor()
model3 = DecisionTreeRegressor()
model4 = KNeighborsRegressor()


ensemble_model = VotingRegressor(estimators=[
    ('lr', model1),
    ('gb', model2),
    ('dt', model3),
    ('knn', model4)
])


# pipe = Pipeline([
#     ('preprocessor',num_sold_transformer),
#     ('ensemble_model',ensemble_model)
# ])


ensemble_model.fit(x_train,y_train)


y_pred = ensemble_model.predict(x_test)


r2_score(y_test , y_pred)


kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(ensemble_model, x, y, cv=kfold, scoring='r2')


# Print cross-validation results
print("Cross-validation r2 scores:", scores)
print(f"Mean r2 score: {scores.mean():.4f}")
print(f"Standard deviation of r2 scores: {scores.std():.4f}")


test


transformer = ColumnTransformer(transformers=[
    ('encode', OneHotEncoder(), ['country', 'store', 'product'])  # One-hot encoding for categorical features
], remainder='passthrough')


test_transformed = transformer.fit_transform(test)


test_transformed = pd.DataFrame(test_transformed,columns = transformer.get_feature_names_out())


test_transformed


ensemble_model.fit(x_train,y_train)
predictions = ensemble_model.predict(test_transformed)  # Use predict() method instead of calling the model

# Assuming 'id' is defined; you need to ensure it has the same length as predictions
output = pd.DataFrame({'id': test_id , 'num_sold': predictions})

# Save the output to a CSV file
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

