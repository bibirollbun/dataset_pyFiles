# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns


#loading the data
train_set = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_set = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


#seeing the first five rows
train_set[:5]  #can also do train_set.head() or train_set.head(10)


#summary of the pandas dataframe
train_set.info()


# seeing numerical attributes
train_set.describe()


# Counting the loan_paid_back column to see if the data is imbalanced

train_set['loan_paid_back'].value_counts(normalize=True)


# Seeing the correlation matrix (-1 to 1)
tdf = train_set.corr(numeric_only=True)
tdf['loan_paid_back'].sort_values(ascending=False)


# Visualization - using pandas wrapper of matplotlib
# Viz 1 - The shapes of numerical ones
train_set.hist(bins=50, figsize=(12,8))
plt.show()


# Viz 2 - Categorical vs. Target

sns.barplot(x='grade_subgrade',y='loan_paid_back',data=train_set,order=sorted(train_set['grade_subgrade'].unique()))


sns.barplot(x='loan_purpose',y='loan_paid_back',data=train_set)


# Creating the two sets - X (questions) and Y (answers)
# droping more than one columns requires giving them in []
# the axis = 0/1 tells it to delete row/column

features = train_set.drop(['loan_paid_back','id'],axis=1)
target = train_set['loan_paid_back']


# just seeing if it has 2 columns less
features.shape


# Now Splitting the dataset into train and val
from sklearn.model_selection import train_test_split


#We pass features and target together (if done separately they will be randomized separately which confuses the model - no right answers)
# random_state fixed insures that every time the same random sequence is used 
# stratify=target ensures the 80/20 contains the same imbalance (not all 0s in 20 or something like that)
x_train,x_val, y_train, y_val = train_test_split(features,
                                                 target,
                                                 test_size = 0.2,
                                                 random_state=42,
                                                 stratify=target)


print(x_train.shape)
print(y_train.shape)


# making two list of columns
# could also be done with .drop and .[] methods but this is fine as its adapts automatically no need to manually name
num_cols = x_train.select_dtypes(include=np.number).columns
cat_cols = x_train.select_dtypes(include='object').columns


cat_cols


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# making the pipelines
# they will be run by columntransformers (managers that will pass the required column in reqr pipeline)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


# Now making the master pipeline assembling the two

from sklearn.compose import ColumnTransformer

preprocessing = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])


x_train_prep = preprocessing.fit_transform(x_train)
x_train_prep.shape


# doing the same for validation set, but only transform this time not fit_transform
# bcz fit_transform will recalculate the mean/median for the val data which will come different and model will get confused as to what is scaled to 0
x_val_prep = preprocessing.transform(x_val)


# let's see what new columns have the encoder made

newnames = preprocessing.get_feature_names_out()
newnames


#toarray is used to unzip the sparse matrix made by sklearn
pd.DataFrame(x_train_prep[:3].toarray(), columns = newnames)


from sklearn.linear_model import LogisticRegression


log_reg = LogisticRegression(random_state=42)

log_reg.fit(x_train_prep,y_train)


# it outputs two columns complementry of each other we are slicing off the first one
y_proba = log_reg.predict_proba(x_val_prep)[:,1]
y_proba


from sklearn.metrics import roc_auc_score


# Moment of truth
score = roc_auc_score(y_val, y_proba)
print(f"My AUC score : {score}")


#preparing the test data
x_test = test_set.drop(['id'], axis=1)
x_test_prep = preprocessing.transform(x_test)


# testing on this data now
test_proba = log_reg.predict_proba(x_test_prep)[:, 1]
test_proba


submission = pd.DataFrame({
    'id': test_set['id'],
    'loan_paid_back': test_proba
    })
submission.head()


# Saving the final file to csv
submission.to_csv('submission.csv', index=False)

