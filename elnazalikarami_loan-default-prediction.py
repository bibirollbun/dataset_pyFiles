# # Imports

import numpy as np 
import pandas as pd 
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_rows', 100)
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(font_scale = 2)




# # # Loading Data and Data Cleaning

# Read in data into a dataframe 
train_data = pd.read_csv('/kaggle/input/loan-default-prediction/train_v2.csv.zip')

# Display top of dataframe
train_data.head()



# # Data Types and Missing Values
train_data.shape

# See the column data types and non-missing values
train_data.info()


# Reading test data.
test_data = pd.read_csv('/kaggle/input/loan-default-prediction/test_v2.csv.zip',index_col='id')
test_data.head()


# Showing information about test data.
test_data.info()


# Data type
train_data.select_dtypes(include=['object']).head()


# Statistics for each column
train_data.describe()


# # Missing Values

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    


missing_values_table(train_data).head(50)


numeric_cols = train_data.select_dtypes(include='number').columns
train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].mean())

#data.fillna(data.select_dtypes(include='number').mean(), inplace=True)


missing_values_table(train_data).head(50)


train_data.dropna(inplace=True)
missing_values_table(train_data)


train_data.shape


# Identify numeric columns
numeric_cols = train_data.select_dtypes(include='number').columns.tolist()


numeric_cols = [col for col in numeric_cols if col not in ['id', 'loss']]

# Fill missing values only for those columns
train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].mean())
test_data[numeric_cols] = test_data[numeric_cols].fillna(train_data[numeric_cols].mean())  # use train mean for test






#test = test_data[numeric_cols[:-1]] # chosing only the numric columns from test data (after excluding loss data).
#train = train_data[numeric_cols] # chosing only the numric columns from train data 

train = train_data[numeric_cols]
test = test_data[numeric_cols]


#  Keep 'loss' in the training data for modeling
features = [col for col in numeric_cols if col not in ['id']]  # exclude 'id' but keep 'loss'
train = train_data[features + ['loss']]  # include 'loss' back explicitly

#  Drop rows where 'loss' is missing
train.dropna(axis=0, subset=['loss'], inplace=True)

# Split into X and y
X_train_full = train.drop(['loss'], axis=1)
y_train_full = train['loss']

#  Train-validation split
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42)



# A numrical transformer.
num_trans = Pipeline(steps = [
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

#numeric_cols.remove('loss')

# A preprocessor that combines the two previous transformers.
preprocessor = ColumnTransformer(transformers = [
    ('num', num_trans, numeric_cols)
],
    remainder = "drop")


X_train_trans = preprocessor.fit_transform(X_train) # Preprocessing train data.
X_valid_trans = preprocessor.transform(X_valid) # Preprocessing validation data.


X_train_trans.shape # chincking the shape of the preprocessed data.


# creating a function that transform the preprocessed data that is in numpy array form to DataFrame form.
def array_to_df(arr):
    return pd.DataFrame(data = arr,columns = X_train.columns)


X_train_trans = array_to_df(X_train_trans)
X_valid_trans = array_to_df(X_valid_trans)


# creating a LogisticRegression model that will help us choosing the right features.
from sklearn.linear_model import LogisticRegression

my_model = LogisticRegression().fit(X_train_trans, y_train)



# Geting the most effictive features on predictions.
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(X_valid_trans.head(1000), y_valid.head(1000))
eli5.show_weights(perm, feature_names = X_valid_trans.columns.tolist())


# Choosing inly the important features
from sklearn.feature_selection import SelectFromModel

sel = SelectFromModel(perm, threshold=0.001, prefit=True)
X_train_super_trans = sel.transform(X_train_trans)
X_valid_super_trans = sel.transform(X_valid_trans)


# checking the shape of the filtered data.
X_train_super_trans.shape


train.shape


# Creating functions to get the cross validation score.
from sklearn.model_selection import cross_val_score

def cross_val(X_train_super_trans, y_train, model):
    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(estimator = model, X = X_train_super_trans, y = y_train, cv = 5)
    return accuracies.mean()

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    
    # Train the model
    model.fit(X_train_super_trans, y_train)
    
    # evalute
    model_cross = cross_val(X_train_super_trans, y_train, model)
    
    # Return the performance metric
    return model_cross


# # Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
random_cross = fit_and_evaluate(random)

print('Random Forest Performance on the test set: Cross Validation Score = %0.4f' % random_cross)


# Preprocessing and filtering the whole train and test data after checking its Cross Validation Score.
new_train = preprocessor.fit_transform(X_train_full)
new_test = preprocessor.transform(test)

new_train = sel.transform(new_train)
new_test = sel.transform(new_test)


# reading the sample submission dataset
submission = pd.read_csv('../input/loan-default-prediction/sampleSubmission.csv')
submission.head()
id	


# fitting the RandomForestClassifier with the whole train data to make it's predictions more accurate.
random.fit(new_train,y_train_full)

submission.loss = random.predict(new_test)

submission.head(40)


 submission.loss.unique()


# save output for submission
submission.to_csv('submission.csv', index=False, header=True)

