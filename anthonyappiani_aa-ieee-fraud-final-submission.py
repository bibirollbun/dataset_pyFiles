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

import lightgbm as lgb


import warnings
warnings.filterwarnings("ignore")


train_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
print('The shape of train_trans is: ', train_trans.shape)
print('the shape of train_id is: ', train_id.shape)

train = train_trans.merge(train_id, on = 'TransactionID', how = 'left' )
print('The shape of the merged training set is :', train.shape)


print('Memory Usage Before Memory Reduction, ',train.memory_usage().sum()/ 1024**2)

# Loop through columns for float and int dtypes
for col in train.columns:
    # Convert float64 to float32
    if train[col].dtype == 'float64':
        train[col] = train[col].astype('float32')
     # Convert int64 to int 32   
    elif train[col].dtype == 'int64':
        train[col] = train[col].astype('int32')

print('Memory Usage After Memory Reduction, ',train.memory_usage().sum()/ 1024**2)


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
# Make all Cat features objects
train[cat_features] = train[cat_features].astype('object')
print(len(cat_features))


# Removing transID since it is an idividual ID and isFraud since that is the Target Variable
num_features = [col for col in train.columns if col not in cat_features + ['TransactionID','isFraud'] ]

print(len(num_features))


train.head()


# Using bar graph to be able to keep same color scheme throughout
plt.figure(figsize=[8,4])

# create list for counts and labels to plot 
counts = [train.isFraud.value_counts()[0],train.isFraud.value_counts()[1]]
labels = ['Not Fraud', 'Fraud']


plt.bar(labels, counts, label = labels, color = ['Navy', 'Red'])
plt.legend()
plt.xlabel('Not Fraud = 0, Fraud = 1')
plt.ylabel('Count')
plt.title('Count of Fraud and Non Fraud Transactions')

plt.show()



(train.isFraud.value_counts()/len(train)).to_frame()



print(round((train.isFraud.sum()/len(train))*100,4),'% of the the transactions are fraud')


train.isnull().sum().to_frame().T


missing = (train.isnull().sum()/len(train)).sort_values(ascending = False)

plt.figure(figsize=[8,4])

plt.barh(missing.index[:25],missing.values[:25], color = 'orange')
plt.xlabel('Percentage of Missing Data')
plt.ylabel('Variables')
plt.title('Top 25 Variables based on Missing Data')

plt.show()


check = train.isnull().sum()/len(train)

missing_cols = check[check >= .75].index

print(len(missing_cols))


def hist_plots(var):

    # Make sure the categorical var is a str for plotting 
    train[var] = train[var].astype('str')
    # Create fraud boolean 
    fraud = train['isFraud'] == 1
    
    plt.figure(figsize=[12,4])

    plt.subplot(1,2,1)
    # 1st hist will be total distribution. Total plots will be Navy
    plt.hist(train[var], edgecolor='black', color='Navy')
    plt.title(f'Histogram of {var}')

    plt.subplot(1,2,2)
    # 2nd hist will be of fraud accts. Fraud plots will be Red 
    plt.hist(train.loc[fraud,var], edgecolor = 'black', color = 'red')
    plt.title(f'Histogram of Fraud by {var}')

    plt.show()
    print(" ")    


train.TransactionAmt.describe()


fraud = train['isFraud'] == 1
train.loc[fraud,'TransactionAmt'].describe()


train['transactionlvl'] =  ['High' if x > 135 else 'Low' for x in train['TransactionAmt']]


hist_plots('transactionlvl')


train.boxplot(column = 'TransactionAmt', by = 'isFraud')
plt.ylim(0,500)
plt.show()


cat_10 = [col for col in train[cat_features] if train[col].nunique() <= 10 and col not in missing[:50]]

print(len(cat_10))
print(cat_10)


for var in range(0,6):
    hist_plots(cat_10[var])


for var in range(6,11):
    hist_plots(cat_10[var])


for var in range(11,16):
    hist_plots(cat_10[var])


for var in range(16,22):
    hist_plots(cat_10[var])


new_feats = [col for col in train.columns if col not in missing_cols ]
print(len(new_feats))


new_cat = [col for col in new_feats if col in cat_features]
new_num = [col for col in new_feats if col in num_features ]
print(len(new_cat))
print(len(new_num))


cat_100 = [col for col in train[new_cat] if train[col].nunique() < 100]
print(len(cat_100))
print(cat_100)


# make sure all cat vars are str type for transformer
train[cat_100] = train[cat_100].astype('str')

print(train[cat_100].dtypes.unique())


features = new_num + cat_100


num_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = -999)), # replaces missing values with -999
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


%%time
lr_clf = LogisticRegression(max_iter=500, solver='saga', penalty='elasticnet',l1_ratio=0)
lr_clf.fit(x_sample,y_sample)


lr_preds = lr_clf.predict_proba(x_valid)[:,1]
lr_auc = roc_auc_score(y_valid, lr_preds)
print("AUC-ROC Train Score:", lr_auc)


%%time 

rf_clf = RandomForestClassifier(random_state=1, n_estimators=10)

rf_parameters = {
    'max_depth': [ 16, 32, 64],
    'min_samples_leaf': [4, 8, 16, 32]
}

rf_grid = GridSearchCV(rf_clf, rf_parameters, cv=5, refit='True', n_jobs=-1, verbose=10, scoring='roc_auc')
rf_grid.fit(x_sample, y_sample)

rf_model = rf_grid.best_estimator_
rf_auc = rf_model.score(x_valid, y_valid)

print('Best Parameters:', rf_grid.best_params_)
print('Best CV Score:  ', rf_grid.best_score_)
print('Training AUC:   ', rf_model.score(x_sample, y_sample))
print('Validation AUC: ', rf_model.score(x_valid, y_valid))


hgb_clf = HistGradientBoostingClassifier(learning_rate=.01, max_iter= 1000, max_leaf_nodes=32, 
                                         max_depth=64,
                                         scoring = 'roc_auc', random_state=42 )


%%time
hgb_clf.fit(x_sample,y_sample)


sample_preds = hgb_clf.predict_proba(x_sample)[:,1]
hgb_sample_auc_roc = roc_auc_score(y_sample, sample_preds)
print("AUC-ROC Train Score:", hgb_sample_auc_roc)

val_preds = hgb_clf.predict_proba(x_valid)[:,1]
hgb_auc = roc_auc_score(y_valid, val_preds)
print("AUC-ROC Train Score:", hgb_auc)


train_data = lgb.Dataset(x_sample, label = y_sample)
valid_data = lgb.Dataset(x_valid, label = y_valid, reference= train_data)


params = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalanced': 'true',
    'boosting' : 'gbdt',
    'num_leaves': 112,
    'max_depth': -1,
    'bagging_fraction' : .5,
    'bagging_freq': 10,
    'learing_rate':.01,
    'verbose':-1
}


%%time
lgb_model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )


params = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalanced': 'true',
    'boosting' : 'gbdt',
    'num_leaves': 164,
    'max_depth': -1,
    'bagging_fraction' : .6,
    'bagging_freq': 10,
    'learing_rate':.1,
    'verbose':-1
}


%%time
lgb_model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )


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


test[cat_100] = test[cat_100].astype('str')


x_test = preprocessor.transform(test[features])


x_test.shape


test_preds = hgb_clf.predict_proba(x_test)[:,1]


submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
submission.isFraud = test_preds
submission.to_csv('hg_submission.csv', index = False, header=True)
print(submission.head())
print('File Saved and Ready to Submit')

