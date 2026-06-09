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


#Loading the datasets:
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


print(train.describe())
print(test.describe())


print(train.info())


# Personality Distribution
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Personality', data=train)
plt.title("Target Class Distribution")
plt.xticks([0, 1], ['Extrovert', 'Introvert'])
plt.show()

# Feature Correlation
# Visualize Feature Correlation - Numeric Columns Only
plt.figure(figsize=(10, 8))
numeric_data = train.select_dtypes(include='number')  # select only numeric columns
sns.heatmap(numeric_data.corr(), cmap='coolwarm', annot=True)
plt.title("Correlation Heatmap")
plt.show()

#Boxplots for numerical values in the training dataset:
for col in train.select_dtypes(include = 'number'):
    plt.figure()
    sns.boxplot(x='Personality', y=col, data=train)
    plt.title(f'{col} by Personality')
    plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(train.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()


numerical_features = train.select_dtypes(include=['int64', 'float64']).columns.drop(['id'])
categorical_features = train.select_dtypes(include=['object']).columns.drop(['Personality'])


for col in numerical_features:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(test[col].median())

for col in categorical_features:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])


def eda_plots(df):
    print("\nClass Distribution:\n", df['Personality'].value_counts())
    
    for col in numerical_features:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()

    for col in categorical_features:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col, hue='Personality')
        plt.title(f"{col} vs Personality")
        plt.show()

eda_plots(train)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])



X_train = train.drop(['id', 'Personality'], axis = 1)
X_test = test.drop('id', axis = 1)
Y_train = train['Personality']

logreg_pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter = 1000))
])

#Fitting the model
logreg_pipeline.fit(X_train, Y_train)


y_pred_logreg = logreg_pipeline.predict(X_test)
submission = pd.DataFrame({
    "id" : test['id'],
    "Personality" : y_pred_logreg
})

print(submission)
submission.to_csv("submission.csv", index=False)

