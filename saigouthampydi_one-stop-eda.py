# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ignoring the warnings, coz i hate them 
import warnings 
warnings.filterwarnings('ignore')


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


numerical_features=train.select_dtypes(include=['number']).columns
categorical_features=train.select_dtypes(include=['object']).columns



train[numerical_features]=train[numerical_features].fillna(train[numerical_features].mean())

for feature in categorical_features:
    train[feature]=train[feature].fillna(train[feature].mode()[0])


train.info()


for col in numerical_features[1:]:
    plt.figure(figsize=(10, 6))
    sns.histplot(train[train['Personality']=='Introvert'][col],color='red',stat='density',alpha=0.6,label='Introvert',bins='auto')
    sns.histplot(train[train['Personality']=='Extrovert'][col],color='blue',stat='density',alpha=0.6,label='Extrovert',bins='auto')
    plt.title(f'Distribution of {col} by Personality Type')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend(title='Personality')
    plt.show()


for col in categorical_features:
        plt.figure(figsize=(12, 7)) 
        sns.countplot(data=train, x=col, hue='Personality', palette={'Introvert': 'red', 'Extrovert': 'blue'}, alpha=0.7)
        plt.title(f'Count of {col} by Personality Type')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Personality')
        plt.tight_layout()
        plt.show()


train['Personality'].value_counts()/train.shape[0]*100


train[numerical_features[1:]].corr()


sns.heatmap(train[numerical_features[1:]].corr())
plt.title('Correlation Matrix')
plt.show()


#Start with the 'Introvert' class
train[train['Personality']=='Introvert'].describe()


#Moving onto the 'Extrovert' class
train[train['Personality']=='Extrovert'].describe()


#Box plots for the 'Introvert' class
for col in numerical_features[1:]:
    plt.boxplot(train[train['Personality']=='Introvert'][col])
    plt.title(f'Boxplot of {col}')
    plt.show()


#Box plots for the 'Introvert' class
for col in numerical_features[1:]:
    plt.boxplot(train[train['Personality']=='Extrovert'][col])
    plt.title(f'Boxplot of {col}')
    plt.show()


df_cleaned=train.copy()
classes = train['Personality'].unique()
for feature in numerical_features[1:]:
    print(f"\nProcessing feature: '{feature}'")

    for class_val in classes:
        print(f"  Processing class: '{class_val}' for feature '{feature}'")

        # Filter data for the current class
        class_data = df_cleaned[df_cleaned['Personality'] == class_val]
        feature_data = class_data[feature]

        # Calculate Q1, Q3, and IQR for the current feature within the current class
        Q1 = feature_data.quantile(0.25)
        Q3 = feature_data.quantile(0.75)
        IQR = Q3 - Q1

        # Calculate outlier fences
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"    Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"    Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")

        # Identify outliers for the current feature and class
        outliers_indices = df_cleaned[
            (df_cleaned['Personality'] == class_val) &
            ((df_cleaned[feature] < lower_bound) | (df_cleaned[feature] > upper_bound))
        ].index

        if not outliers_indices.empty:
            print(f"    Identified {len(outliers_indices)} outliers for '{feature}' in class '{class_val}'.")
            # Drop the outliers from the main cleaned DataFrame
            df_cleaned.drop(outliers_indices, inplace=True)
            print(f"    Removed outliers. New size for class '{class_val}': {df_cleaned[df_cleaned['Personality'] == class_val].shape[0]}")
        else:
            print(f"    No outliers found for '{feature}' in class '{class_val}'.")



train.shape


df_cleaned.shape


df_cleaned[df_cleaned['Personality']=='Extrovert'].describe()


df_cleaned[df_cleaned['Personality']=='Introvert'].describe()


for col in numerical_features[1:]:
    plt.boxplot(df_cleaned[df_cleaned['Personality']=='Introvert'][col])
    plt.title(f'Boxplot of {col}')
    plt.show()


for col in numerical_features[1:]:
    plt.boxplot(df_cleaned[df_cleaned['Personality']=='Extrovert'][col])
    plt.title(f'Boxplot of {col}')
    plt.show()


from sklearn.preprocessing import StandardScaler

result = df_cleaned.copy()

for col in categorical_features:
    dummies = pd.get_dummies(df_cleaned[col], prefix=f'{col}', drop_first=True)
    result = pd.concat([result, dummies], axis=1)

result.drop(categorical_features, axis='columns', inplace=True)
result.drop('id', axis='columns', inplace=True)

numerical_columns = [col for col in result.columns if col not in dummies.columns]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(result[numerical_columns])

scaled_df = pd.DataFrame(scaled_data, columns=numerical_columns, index=result.index)

final_df = pd.concat([scaled_df, result[dummies.columns]], axis=1)

final_df.head()


final_df.to_csv('final_df.csv')

