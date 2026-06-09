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


train_trans.head()


train_id.head()


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
train[cat_features] = train[cat_features].astype('object')
print(len(cat_features))


v_features = [f'V{x}' for x in range(1,340)]
fraud_feature = train['isFraud']


import seaborn as sns
corr_matrix = train[v_features[:26]].corr()


plt.figure(figsize=(12, 4))  # Adjust figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


corr_matrix = train[v_features[25:51]].corr()


plt.figure(figsize=(12, 4))  # Adjust figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


corr_matrix = train[v_features[51:76]].corr()


plt.figure(figsize=(12, 4))  # Adjust figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


corr_matrix = train[v_features[76:101]].corr()


plt.figure(figsize=(12, 4))  # Adjust figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


corr_matrix = train[v_features[101:126]].corr()


plt.figure(figsize=(12, 4))  # Adjust figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


corr_matrix = train[v_features[126:151]].corr()


plt.figure(figsize=(12, 4))  # Adjust figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


corr_matrix = train[v_features[151:176]].corr()


plt.figure(figsize=(12, 4))  # Adjust figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


corr_matrix = train[v_features[175:201]].corr()


plt.figure(figsize=(12, 4))  # Adjust figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


corr_matrix = train[v_features[200:226]].corr()


plt.figure(figsize=(12, 4))  # Adjust figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


corr_matrix = train[v_features[225:251]].corr()


plt.figure(figsize=(12, 4))  # Adjust figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


corr_matrix = train[v_features[250:276]].corr()


plt.figure(figsize=(12, 4))  # Adjust figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


corr_matrix = train[v_features[275:301]].corr()


plt.figure(figsize=(12, 4))  # Adjust figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


corr_matrix = train[v_features[300:]].corr()


plt.figure(figsize=(16, 8))  # Adjust figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


%%time

correlation_matrix = train[v_features].corr()

threshold = 0.85
grouped_features = []
grouped_names = []
for col in correlation_matrix.columns:
    if col not in grouped_names:
        temp_group = [col]
        for other_col in correlation_matrix.columns:
            if col != other_col and other_col not in grouped_names and abs(correlation_matrix.loc[col, other_col]) > threshold:
                temp_group.append(other_col)
                grouped_names.append(other_col)
        grouped_features.append(temp_group)

print(grouped_features)


v_features_reduced = []
for x in range(len(grouped_features)):
    feature = grouped_features[x][0]
    v_features_reduced.append(feature)

print(v_features_reduced)


num_features = [col for col in train.columns if col not in cat_features + ['TransactionID','isFraud', 'transactionlvl'] + v_features]

print(len(num_features))


cat_100 = [col for col in train[cat_features] if train[col].nunique() < 100]
train[cat_100] = train[cat_100].astype('str')
print(len(cat_100))
print(cat_100)


features = num_features + v_features_reduced + cat_100 
print(len(features))


print(isinstance(v_features_reduced, tuple))


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
        ('v', num_transformer, v_features_reduced),
        ('cat', cat_transformer, cat_100)
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


hgb_clf = HistGradientBoostingClassifier(learning_rate=.1, max_iter= 1000, max_leaf_nodes=32, 
                                         max_depth=64,
                                         scoring = 'roc_auc', random_state=42, class_weight='balanced',
                                        n_iter_no_change = 100)


%%time
hgb_clf.fit(x_sample.toarray(),y_sample)


sample_preds = hgb_clf.predict_proba(x_sample.toarray())[:,1]
hgb_sample_auc_roc = roc_auc_score(y_sample, sample_preds)
print("AUC-ROC Train Score:", hgb_sample_auc_roc)

val_preds = hgb_clf.predict_proba(x_valid.toarray())[:,1]
hgb_auc = roc_auc_score(y_valid, val_preds)
print("AUC-ROC Train Score:", hgb_auc)


train_data = lgb.Dataset(x_sample, label = y_sample)
valid_data = lgb.Dataset(x_valid, label = y_valid, reference= train_data)


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
test['id_32'] = test['id_32'].astype('str')


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


x_test = preprocessor.transform(test)
x_test.shape


test_preds = hgb_clf.predict_proba(x_test.toarray())[:,1]


submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
submission.isFraud = test_preds
submission.to_csv('hgb_submission.csv', index = False, header=True)
print(submission.head())
print('File Saved and Ready to Submit')


lgb_test_preds = lgb_model.predict(x_test)


submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
submission.isFraud = lgb_test_preds
submission.to_csv('lgb_submission.csv', index = False, header=True)
print(submission.head())
print('File Saved and Ready to Submit')

