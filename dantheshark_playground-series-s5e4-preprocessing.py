from IPython.display import display, Markdown
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
import math
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import pandas as pd 
import scipy.stats as ss
import seaborn as sns
import os
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Decide between local or kaggle cloud storage         
KAGGLE_ENV = 'kaggle' in os.listdir('/')
data_path = '/kaggle/input' if KAGGLE_ENV else '../kaggle/input'

# This is a good idea to work only locally. But If you wanna ran your NB also at kaggle... this is not working.
# # Pull the dataset from kaggle, it is concat dataset train + original dataset
# dataset_name = 'dantheshark/s4-e11-train-concat'
# if KAGGLE_ENV:
#     kaggle.api.dataset_download_files(dataset_name, path="../kaggle/input/", unzip=True)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    
for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
        print(os.path.join(dirname, filename)) 


# Load the data
train_original = pd.read_csv(data_path + '/playground-series-s5e4/train.csv')
test_original = pd.read_csv(data_path + '/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv(data_path + '/playground-series-s5e4/sample_submission.csv')

original_data = pd.read_csv(data_path + '/podcast-listening-time-prediction-dataset/podcast_dataset.csv')


#Concat train and the original data set
train_original.drop('id', axis=1, inplace=True) #id is not needed for training
train = train_original.copy()
train = pd.concat([train, original_data],ignore_index=True)

test = test_original.copy()
test.drop('id', axis=1, inplace=True) #id is not needed for testing


train.iloc[train_original.shape[0]-5:train_original.shape[0]+5].head(10)


# Save concat Dataset
train.to_csv('train_concat.csv', index=False)


train.info()


def convert_object_to_category(df, max_unique_values=50):
    """
    Converts all object columns to category
    if the number of unique values is below max_unique_values.
    """
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() <= max_unique_values:
            df[col] = df[col].astype('category')
            print(f"Converted {col} to category")
    return df


convert_object_to_category(train)


train.info()


# # Convert Float to Int, not needed float
# columns_to_convert = [] # list of columns to convert

# for col in columns_to_convert:
#     train[col] = pd.to_numeric(train[col], errors='coerce').astype('Int64')
#     train[col] = pd.to_numeric(train[col], errors='coerce').astype('Int64')


numeric_features = train.select_dtypes(include=['int64', 'float64']).columns


train.info()


plt.figure(figsize=(12, 6))
sns.boxplot(data=train[numeric_features])
plt.xticks(rotation=90)
plt.show()


train.describe()


# Check what kind of Episodes are they?
train_long_episodes = train[train['Episode_Length_minutes'] >= 150]
train_long_episodes.head()
# Lets delete it... not sure about it.


# Lets have a look..
train_long_episodes = train[train['Number_of_Ads'] >= 10]
train_long_episodes.head()
# Lets delete it... not sure about it.


train = train[train['Episode_Length_minutes'] <= 150]
train = train[train['Number_of_Ads'] <= 10]


train.info()


plt.figure(figsize=(12, 6))
sns.boxplot(data=train[numeric_features])
plt.xticks(rotation=90)
plt.show()


train.describe()


print(train.duplicated().sum())
print(train.duplicated(subset=['Podcast_Name', 'Episode_Title']).sum())
# Be careful, playground datasets are artifically created, so It is possible we have variatnion in the data.
# Conclusion: Only delete the 1:1 duplicates.
train = train.drop_duplicates()



print(train.duplicated().sum())


def show_general_stats(df):
    display(Markdown('### General Stats'))
    display(df.describe())
    display(Markdown('### Data Types'))
    display(df.dtypes)
    display(Markdown('### Missing Values'))
    display(df.isnull().sum())
    display(Markdown('### Shape'))
    display(df.shape)
    display(Markdown('### Head'))
    display(df.head(100))
    display(Markdown('### Tail'))
    display(df.tail(100))
    display(Markdown('### Sample'))
    display(df.sample(100))
    display(Markdown('### '))


show_general_stats(train)


# 1. Guest_Popularity: missing = no guest (0), add flag
train['has_guest'] = (~train['Guest_Popularity_percentage'].isna()).astype(int)
train['Guest_Popularity_percentage'].fillna(0, inplace=True)

# 2. Episode_Length: Median
train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].mean(), inplace=True)

# 3. Listening_Time: only data with data!
train = train[train['Listening_Time_minutes'].notna()]

# 4. Number_of_Ads: 0
train['Number_of_Ads'].fillna(0, inplace=True)


train.isnull().sum()


test.info()


convert_object_to_category(test)


test.info()


test.info()


numeric_features = test.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(12, 6))
sns.boxplot(data=test[numeric_features])
plt.xticks(rotation=90)
plt.show()


test.describe()


# Check what kind of Episodes are they?
test_long_episodes = test[test['Episode_Length_minutes'] >= 150]
test_long_episodes.head()
# Lets delete it.. way to huge!


test = test[test['Episode_Length_minutes'] <= 150]


plt.figure(figsize=(12, 6))
sns.boxplot(data=test[numeric_features])
plt.xticks(rotation=90)
plt.show()


test_number_of_ads = test[test['Number_of_Ads'] > 10]
test_number_of_ads.head()
# Does not look plausible, lets delete it.



test = test[test['Number_of_Ads'] <= 10]
test.info()


plt.figure(figsize=(12, 6))
sns.boxplot(data=test[numeric_features])
plt.xticks(rotation=90)
plt.show()


print(test.duplicated().sum())
# print(train.duplicated(subset=['Podcast_Name', 'Episode_Title']).sum())
# # Be careful, playground datasets are artifically created, so It is possible we have variatnion in the data.
# # Conclusion: Only delete the 1:1 duplicates.
# train = train.drop_duplicates()


show_general_stats(test)


# 1. Guest_Popularity: missing = no guest (0), add flag
test['has_guest'] = (~test['Guest_Popularity_percentage'].isna()).astype(int)
test['Guest_Popularity_percentage'].fillna(0, inplace=True)


test.isnull().sum()


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = train.copy()

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_feature = encoder.fit_transform(df[['Episode_Title']])
df_encoded = pd.DataFrame(encoded_feature, columns=encoder.get_feature_names_out(['Episode_Title']))

df = df.drop(columns=['Episode_Title'])
df = pd.concat([df, df_encoded], axis=1)

# Final DataFrame
train = df.copy()

# Show stats
show_general_stats(train)


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = test.copy()

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_feature = encoder.fit_transform(df[['Episode_Title']])
df_encoded = pd.DataFrame(encoded_feature, columns=encoder.get_feature_names_out(['Episode_Title']))

df = df.drop(columns=['Episode_Title'])
df = pd.concat([df, df_encoded], axis=1)

# Final DataFrame
test = df.copy()

# Show stats
show_general_stats(test)


if KAGGLE_ENV:
    train.to_csv('/kaggle/working/s5-e4-train_preprocessed.csv', index=False)
else:
    train.to_csv( '../kaggle/working/s5-e4-train_preprocessed.csv', index=False)


if KAGGLE_ENV:
    test.to_csv('/kaggle/working/s5-e4-test_preprocessed.csv', index=False)
else:
    test.to_csv( '../kaggle/working/s5-e4-test_preprocessed.csv', index=False)

