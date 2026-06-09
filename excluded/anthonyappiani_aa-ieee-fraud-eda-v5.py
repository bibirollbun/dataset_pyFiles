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


train[cat_features] = train[cat_features].fillna('Missing')
train[cat_features] = train[cat_features].astype('str')


def hist_plots(var):

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


print(train.TransactionAmt.min())
print(train.TransactionAmt.max())
print(train.TransactionAmt.median())
print(train.TransactionAmt.mean())


train['transactionlvl'] =  ['High' if x > 135 else 'Low' for x in train['TransactionAmt']]


hist_plots('transactionlvl')


train.boxplot(column = 'TransactionAmt', by = 'isFraud')
plt.ylim(0,500)


hist_plots('ProductCD')


hist_plots('DeviceType')


hist_plots('M1')


hist_plots('M2')


hist_plots('M3')


hist_plots('M4')


hist_plots('M5')


hist_plots('M6')


hist_plots('card4')


hist_plots('card6')


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


#%%time
#dt_clf = DecisionTreeClassifier(max_depth = 24, min_samples_leaf = 64,random_state=42)
#dt_clf.fit(x_sample,y_sample)


%%time 

dt_clf = DecisionTreeClassifier(random_state=1)

dt_parameters = {
    'max_depth': [ 10, 16, 32],
    'min_samples_leaf': [ 8, 16, 24, 32, 64]
}

dt_grid = GridSearchCV(dt_clf, dt_parameters, cv=5, refit='True', n_jobs=-1, verbose=10, scoring='roc_auc')
dt_grid.fit(x_sample, y_sample)

dt_model = dt_grid.best_estimator_

print('Best Parameters:', dt_grid.best_params_)
print('Best CV Score:  ', dt_grid.best_score_)
print('Training AUC:   ', dt_model.score(x_sample, y_sample))
print('Validation AUC: ', dt_model.score(x_valid, y_valid))


sample_probs = dt_model.predict_proba(x_sample)[:,1]
auc_roc_sample = roc_auc_score(y_sample, sample_probs)
print("AUC-ROC Train Score:", auc_roc_sample)


dt_val_pred = dt_model.predict_proba(x_valid)[:,1]
dt_auc_roc = roc_auc_score(y_valid, dt_val_pred)
print("AUC-ROC Valid Score:",dt_auc_roc)


#%%time
#rf_clf = RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs= -1)
#rf_clf.fit(x_sample,y_sample)


%%time 

rf_clf = RandomForestClassifier(random_state=1, n_estimators=10)

rf_parameters = {
    'max_depth': [ 16, 32, 64, 112],
    'min_samples_leaf': [ 8, 16, 32]
}

rf_grid = GridSearchCV(rf_clf, rf_parameters, cv=5, refit='True', n_jobs=-1, verbose=10, scoring='roc_auc')
rf_grid.fit(x_sample, y_sample)

rf_model = rf_grid.best_estimator_

print('Best Parameters:', rf_grid.best_params_)
print('Best CV Score:  ', rf_grid.best_score_)
print('Training AUC:   ', rf_model.score(x_sample, y_sample))
print('Validation AUC: ', rf_model.score(x_valid, y_valid))


y_proba_rf = rf_model.predict_proba(x_sample)[:,1]
auc_roc = roc_auc_score(y_sample, y_proba_rf)
print("AUC-ROC Train Score:", auc_roc)


valid_pred = rf_model.predict_proba(x_valid)[:, 1]
rf_auc_roc = roc_auc_score(y_valid, valid_pred)

print("AUC-ROC Valid Score:", rf_auc_roc)


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


#test_probs = rf_clf.predict_proba(x_test)[:,1]


if rf_auc_roc > dt_auc_roc :
    final_model = rf_model
else: final_model = dt_model


print(final_model)


test_preds = final_model.predict_proba(x_test)[:,1]


submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')


submission.isFraud = test_preds


submission.to_csv('submission.csv', index = False, header=True)

