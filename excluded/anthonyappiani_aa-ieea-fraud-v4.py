import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import gc

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV



import warnings
warnings.filterwarnings("ignore")


train_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
print('The shape of train_trans is: ', train_trans.shape)
print('the shape of train_id is: ', train_id.shape)


train_trans.head()


train_id.head()


train = train_trans.merge(train_id, on = 'TransactionID', how = 'left' )
print('The shape of the merged training set is :', train.shape)


train.dtypes.unique()


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
print(len(cat_features))


cat_10 = [col for col in train[cat_features] if train[col].nunique() < 11]
print(len(cat_10))
print(cat_10)


train[cat_10].dtypes


train['id_32'] = train['id_32'].astype('str')


num_features = [col for col in train.columns if col not in cat_features + ['TransactionID','isFraud']]

print(len(num_features))


features = num_features + cat_10
print(len(features))


y_train = train['isFraud']


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
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_10)
    ]
)


preprocessor.fit(train[features])


x_train = preprocessor.transform(train)
print(x_train.shape)
print(y_train.shape)


x_sample, x_valid, y_sample, y_valid = train_test_split(x_train, y_train, test_size=.2,
                                                       stratify = y_train)

print(x_sample.shape)
print(x_valid.shape)
print(y_sample.shape)
print(y_valid.shape)


%%time
dt_clf = DecisionTreeClassifier(max_depth = 24, min_samples_leaf = 64,random_state=42)
dt_clf.fit(x_sample,y_sample)


sample_probs = dt_clf.predict_proba(x_sample)[:,1]
auc_roc_sample = roc_auc_score(y_sample, sample_probs)
print("AUC-ROC Train Score:", auc_roc_sample)


dt_val_pred = dt_clf.predict_proba(x_valid)[:,1]
auc_roc = roc_auc_score(y_valid, dt_val_pred)
print("AUC-ROC Valid Score:", auc_roc)


%%time
rf_clf = RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs= -1)
rf_clf.fit(x_sample,y_sample)


y_proba_rf = rf_clf.predict_proba(x_sample)[:,1]
auc_roc = roc_auc_score(y_sample, y_proba_rf)
print("AUC-ROC Train Score:", auc_roc)


valid_pred = rf_clf.predict_proba(x_valid)[:, 1]
auc_roc = roc_auc_score(y_valid, valid_pred)

print("AUC-ROC Valid Score:", auc_roc)



%%time
lr_clf = LogisticRegression(max_iter=100, solver='saga', penalty='elasticnet',l1_ratio=0)
lr_clf.fit(x_sample,y_sample)


lr_pred = lr_clf.predict_proba(x_sample)[:,1]
lr_roc = roc_auc_score(y_sample, lr_pred)
print("AUC-ROC Train Score:", lr_roc)


lr_val_pred = lr_clf.predict_proba(x_valid)[:,1]
lr_val_roc = roc_auc_score(y_valid,lr_val_pred)
print("AUC-ROC Valid Score:", lr_val_roc)


test_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
test_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

test = test_trans.merge(test_id, on='TransactionID', how = 'left')
test.columns = [col.replace('-', '_') for col in test.columns]
test['id_32'] = test['id_32'].astype('str')


print('Before Memory Reduction, ',test.memory_usage().sum()/ 1024**2)

for col in test.columns:
    if test[col].dtype == 'float64':
        test[col] = test[col].astype('float32')
    elif test[col].dtype == 'int64':
        test[col] = test[col].astype('int32')

print('After Memory Reduction, ', test.memory_usage().sum()/ 1024**2)


x_test = preprocessor.transform(test)


test_probs = rf_clf.predict_proba(x_test)[:,1]


submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')


submission.isFraud = test_probs


submission.to_csv('submission.csv', index = False, header=True)

