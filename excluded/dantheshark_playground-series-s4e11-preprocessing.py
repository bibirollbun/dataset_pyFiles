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
train_original = pd.read_csv(data_path + '/playground-series-s4e11/train.csv')
test_original = pd.read_csv(data_path + '/playground-series-s4e11/test.csv')
sample_submission = pd.read_csv(data_path + '/playground-series-s4e11/sample_submission.csv')
original_data = pd.read_csv(data_path + '/depression-surveydataset-for-analysis/final_depression_dataset_1.csv')

train_concat_data = pd.read_csv(data_path + '/s4-e11-train-concat/s4-e11-train-concat.csv')
test_concat_data = pd.read_csv(data_path + '/s4-e11-test-concat/s4-e11-test-concat.csv')


train_concat_data.head()


test_concat_data.head()



# Convert Float to Int, not needed float
columns_to_convert = ['Work Pressure', 'Job Satisfaction', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress']

for col in columns_to_convert:
    train_concat_data[col] = pd.to_numeric(train_concat_data[col], errors='coerce').astype('Int64')
    test_concat_data[col] = pd.to_numeric(test_concat_data[col], errors='coerce').astype('Int64')  
    



def get_categorical_numerical_features(df):
    # Get Numeric & Categorical Features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric_features, categorical_features
numeric_features, categorical_features = get_categorical_numerical_features(train_concat_data)


# plt.figure(figsize=(12, 6))
# sns.boxplot(data=train_concat_data[numeric_features])
# plt.xticks(rotation=90)
# plt.show()


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


show_general_stats(train_concat_data)
show_general_stats(test_concat_data)


# def cramers_v(x, y):
#     confusion_matrix = pd.crosstab(x, y)
#     chi2 = ss.chi2_contingency(confusion_matrix)[0]
#     n = confusion_matrix.sum().sum()
#     phi2 = chi2 / n
#     r, k = confusion_matrix.shape
#     phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
#     rcorr = r - ((r-1)**2)/(n-1)
#     kcorr = k - ((k-1)**2)/(n-1)
#     return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# cramers_v(train_concat_data['FEATURE'], train_concat_data['Depression'])


# imputer = KNNImputer(n_neighbors=3)  # k=5 nearest neighbors
# print("Start KNN-Imputation...")
# train_original[numeric_features] = imputer.fit_transform(train_original[numeric_features])# only for numerical data
# print(train_original)


# List of columns (only int) with missing values
columns_to_convert = ['Work Pressure', 'Job Satisfaction',  'Work/Study Hours', 'Financial Stress']#, 'Study Satisfaction']# Had problems here, messed up the data 

train_preprocessed = train_concat_data.copy()

imputer = IterativeImputer(
    estimator=RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
    max_iter=200,
    random_state=42,
    min_value=0,
    imputation_order="ascending" 
)
df_subset = train_preprocessed[columns_to_convert]
df_imputed_values = imputer.fit_transform(df_subset)
train_preprocessed[columns_to_convert] = np.round(df_imputed_values).astype(int)

print(imputer.n_iter_)
show_general_stats(train_preprocessed)


columns_to_convert = ['Study Satisfaction'] # a lot of missing values! over 114k! better we do the this calculation seperated

df_train = train_preprocessed.copy()

imputer = IterativeImputer(
    estimator=RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
    max_iter=200,
    random_state=42,
    min_value=0,
    imputation_order="ascending" 
)

df_subset = df_train[columns_to_convert]
df_imputed_values = imputer.fit_transform(df_subset)
df_train[columns_to_convert] = np.round(df_imputed_values).astype(int)

print(imputer.n_iter_)
show_general_stats(df_train)


# List of columns (only int) with missing values
columns_to_convert = ['Work Pressure', 'Job Satisfaction',  'Work/Study Hours', 'Financial Stress']#, 'Study Satisfaction']# Had problems here, messed up the data 

test_preprocessed = test_concat_data.copy()

imputer = IterativeImputer(
    estimator=RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
    max_iter=200,
    random_state=42,
    min_value=0,
    imputation_order="ascending" 
)
df_subset = test_preprocessed[columns_to_convert]
df_imputed_values = imputer.fit_transform(df_subset)
test_preprocessed[columns_to_convert] = np.round(df_imputed_values).astype(int)

print(imputer.n_iter_)
show_general_stats(test_preprocessed)


columns_to_convert = ['Study Satisfaction'] # a lot of missing values! over 114k! better we do the this calculation seperated

df_test = test_preprocessed.copy()

imputer = IterativeImputer(
    estimator=RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
    max_iter=200,
    random_state=42,
    min_value=0,
    imputation_order="ascending" 
)

df_subset = df_test[columns_to_convert]
df_imputed_values = imputer.fit_transform(df_subset)
df_test[columns_to_convert] = np.round(df_imputed_values).astype(int)

print(imputer.n_iter_)
show_general_stats(df_test)


# if KAGGLE_ENV:
#     train.to_csv('/kaggle/working/s4-e11-train-concat-imputed.csv', index=False)
# else:
#     train.to_csv( '../kaggle/working/' + '/s4-e11-train-concat-imputed.csv', index=False)


df_train.head(100)


df_test.head(100)


# Check the unique values in the column 'Degree'
num_unique_degrees = df_train["Degree"].nunique()
unique_degrees = df_train["Degree"].unique()

print(f"Number of unique categories in the Degree column': {num_unique_degrees}")
print(f"Unique categories in the 'Degree' column': {unique_degrees}")


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = df_train.copy()

# Binary Encoding for "Have you ever had suicidal thoughts ?"
df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})

# One-Hot - Encoding for "Working Professional or Student"
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_feature = encoder.fit_transform(df[['Working Professional or Student']])
df_encoded = pd.DataFrame(encoded_feature, columns=encoder.get_feature_names_out(['Working Professional or Student']))

df = df.drop(columns=['Working Professional or Student'])
df = pd.concat([df, df_encoded], axis=1)

# One-Hot-Encoding for "Degree"
encoder_degree = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
degree_encoded = encoder_degree.fit_transform(df2[["Degree"]])
degree_columns = encoder_degree.get_feature_names_out(["Degree"])
df_degree = pd.DataFrame(degree_encoded, columns=degree_columns, index=df2.index)

# Final DataFrame
train_final = df.drop(columns=["Degree"], errors='ignore').join(df_degree)

# Show stats
print("OneHotEncoded Degree Data")
show_general_stats(train_final)


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = df_test.copy()

# Binary Encoding for "Have you ever had suicidal thoughts ?"
df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})

# One-Hot - Encoding for "Working Professional or Student"
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_feature = encoder.fit_transform(df[['Working Professional or Student']])
df_encoded = pd.DataFrame(encoded_feature, columns=encoder.get_feature_names_out(['Working Professional or Student']))

df = df.drop(columns=['Working Professional or Student'])
df = pd.concat([df, df_encoded], axis=1)

# One-Hot-Encoding for "Degree"
encoder_degree = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
degree_encoded = encoder_degree.fit_transform(df2[["Degree"]])
degree_columns = encoder_degree.get_feature_names_out(["Degree"])
df_degree = pd.DataFrame(degree_encoded, columns=degree_columns, index=df2.index)

# Final DataFrame
test_final = df.drop(columns=["Degree"], errors='ignore').join(df_degree)

# Show stats
print("OneHotEncoded Degree Data")
show_general_stats(test_final)


train_final.head(100)


test_final.head(100)


if KAGGLE_ENV:
    train_final.to_csv(data_path + '/s4-e11-train-concat-final/s4-e11-train-concat-final.csv', index=False)
else:
    train_final.to_csv(data_path +  '/s4-e11-train-concat-final/s4-e11-train-concat-final.csv', index=False)


if KAGGLE_ENV:
    test_final.to_csv(data_path +'/s4-e11-test-concat-final/s4-e11-test-concat-final.csv', index=False)
else:
    test_final.to_csv(data_path +'/s4-e11-test-concat-final/s4-e11-test-concat-final.csv', index=False)

