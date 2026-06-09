import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import gc

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV


import warnings
warnings.filterwarnings("ignore")


train_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
print('The shape of train_trans is: ', train_trans.shape)
print('the shape of train_id is: ', train_id.shape)

train = train_trans.merge(train_id, on = 'TransactionID', how = 'left' )
print('The shape of the merged training set is :', train.shape)


print('Before Memory Reduction, ',train.memory_usage().sum()/ 1024**2)

# Loop through columns for float and int dtypes
for col in train.columns:
    # Convert float64 to float32
    if train[col].dtype == 'float64':
        train[col] = train[col].astype('float32')
     # Convert int64 to int 32   
    elif train[col].dtype == 'int64':
        train[col] = train[col].astype('int32')

print('After Memory Reduction, ',train.memory_usage().sum()/ 1024**2)


del train_id
del train_trans
gc.collect()


print(round((train.isFraud.sum()/len(train))*100,4),'% of the the transactions are fraud')


train.isnull().sum().to_frame().T


check = train.isnull().sum()/len(train)

cols = check[check >= .75].index

print(len(cols))


new_feats = [col for col in train.columns if col not in cols ]
print(len(new_feats))


# List created from Data page on competition 
cat_features = ['ProductCD',
                'card1','card2','card3','card4','card5','card6',
                'addr1','addr2',
                'P_emaildomain','R_emaildomain',
                'M1','M2','M3','M4','M5','M6','M7','M8','M9',
                'DeviceType','DeviceInfo']

# use list comprehension for id columns 
id_feats = [f'id_{x}' for x in range(12,39)]

# combine lists for final categorical feature list
cat_features = cat_features + id_feats
# Make all Cat features objects
train[cat_features] = train[cat_features].astype('str')
print(len(cat_features))


num_features = [col for col in train.columns if col not in cat_features + ['TransactionID','isFraud'] ]

print(len(num_features))


new_cat = [col for col in new_feats if col in cat_features]
new_num = [col for col in new_feats if col in num_features]
print(len(new_cat))
print(len(new_num))


cat_100 = [col for col in train[new_cat] if train[col].nunique() < 100]
print(len(cat_100))
print(cat_100)


train[cat_100] = train[cat_100].astype('str')


features = new_num + cat_100


num_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = -999)), # replaces missing values with mean of col
        ('scaler', StandardScaler()) # scales data
    ]
)

cat_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer( strategy = 'constant', fill_value = 'Missing')),
        ('encoder', OneHotEncoder(handle_unknown = 'ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_transformer, new_num),
        ('cat', cat_transformer, cat_100)
    ]
)


train[cat_100].dtypes


preprocessor.fit(train[features])


x_train = preprocessor.transform(train[features])
y_train = train.isFraud
print(x_train.shape)
print(y_train.shape)


x_sample, x_valid, y_sample, y_valid = train_test_split(x_train, y_train, test_size=.2,
                                                       stratify = y_train)

print(x_sample.shape)
print(x_valid.shape)
print(y_sample.shape)
print(y_valid.shape)


hgb_clf = HistGradientBoostingClassifier(learning_rate=.01, max_iter= 1000, max_leaf_nodes=32, 
                                         max_depth=64,
                                         scoring = 'roc_auc', random_state=42 )


%%time
hgb_clf.fit(x_sample,y_sample)


sample_preds = hgb_clf.predict_proba(x_sample)[:,1]
sample_auc_roc = roc_auc_score(y_sample, sample_preds)
print("AUC-ROC Train Score:", sample_auc_roc)


val_preds = hgb_clf.predict_proba(x_valid)[:,1]
val_auc_roc = roc_auc_score(y_valid, val_preds)
print("AUC-ROC Train Score:", val_auc_roc)


test_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
test_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

test = test_trans.merge(test_id, on='TransactionID', how = 'left')
test.columns = [col.replace('-', '_') for col in test.columns]


print('Before Memory Reduction, ',test.memory_usage().sum()/ 1024**2)

for col in test.columns:
    if test[col].dtype == 'float64':
        test[col] = test[col].astype('float32')
    elif test[col].dtype == 'int64':
        test[col] = test[col].astype('int32')

print('After Memory Reduction, ', test.memory_usage().sum()/ 1024**2)


del test_trans
del test_id
gc.collect()


test.shape


test[cat_100] = test[cat_100].astype('str')


x_test = preprocessor.transform(test[features])


x_test.shape


test_preds = hgb_clf.predict_proba(x_test)[:,1]


submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
submission.isFraud = test_preds
submission.to_csv('hgb_submission.csv', index = False, header=True)

