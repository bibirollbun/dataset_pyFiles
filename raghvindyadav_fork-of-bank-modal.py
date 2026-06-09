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


from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

# Remove the id column from train and test

train.drop("id",axis=1,inplace=True)
train.head()


# Describe training data set.
train.describe()


# Check null values
train.isnull().sum()


test_data.isnull().sum()


# Columns names with their unique values.
for cname in train.columns:
    if train[cname].dtype == "object":
        print(cname, train[cname].unique())


# Let make default, housing and loan columns to numeric
cat_numeric_cols = ['default','housing','loan']

for col in cat_numeric_cols:
    train[col]=train[col].map({'yes':1 ,'no':0})
    
for col in cat_numeric_cols:
    test_data[col]=test_data[col].map({'yes':1 ,'no':0})
    

train.head()


X = train.drop(columns=['y'])
y = train['y']


# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X.columns if
                    X[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X.columns if 
                X[cname].dtype in ['int64', 'float64']]
print(categorical_cols,numerical_cols)


# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('oridinal', OrdinalEncoder()),
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)


# Define model
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=40,
    learning_rate=0.05,
    n_estimators=5000,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.8,
)


my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])


# Preprocessing of training data, fit model 
my_pipeline.fit(X, y,)


# ROC AUC score
from sklearn.metrics import roc_auc_score
y_pred_proba = my_pipeline.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)

print(auc_score)


test = test_data.drop(columns=['id'])
test_pred_proba= my_pipeline.predict_proba(test)[:, 1]
test_pred_proba = np.clip(test_pred_proba, 0, 1)


output = pd.DataFrame({
    'id': test_data['id'],
    'y': test_pred_proba
})
output.to_csv('submission.csv', index=False)

