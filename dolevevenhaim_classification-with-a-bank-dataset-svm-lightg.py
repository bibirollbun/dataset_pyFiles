# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
import lightgbm as lgb
from lightgbm.callback import log_evaluation
import time

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


# Selecting featuers
test = test.drop(['id', 'duration'], axis=1)
numerical_features = train.select_dtypes(include ='int').drop(['y', 'id', 'duration'], axis=1)
categories_features = train.select_dtypes(exclude ='int')
y = train['y']


# EDA
train.info()
train.describe()

# check for imbalance y
y.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Target Variable (y)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# categories features frequency by loan
for cat in categories_features:
    print(f'cross tab by {cat} ')
    ct = pd.crosstab(train[cat], y, normalize='index')
    ct = ct.sort_values(by=1, ascending=False)
    print(ct.round(2), '\n')
   
# categories features attending to take a loan
for category in categories_features:
    category_group = train.groupby(category)['y'].mean().round(2).sort_values(ascending=False)
    print(category_group, '\n')
    #category_group.plot(kind='bar', title=f'{category} vs y')
    #plt.show()


# numerical features      
numeric_group = train.groupby('y')[numerical_features.columns].agg(
    ['mean', 'std', lambda col: col.std()/col.mean()]
)
numeric_group = numeric_group.rename(columns={'<lambda_0>': 'cv'})
numeric_group.T


# Plots

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
for index, col in enumerate(["balance", "pdays", "previous"]):
    ax[index].hist(train[col])
    ax[index].set_title(f"Hist for {col}")
plt.tight_layout()    
plt.show()


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
for index, col in enumerate(["balance", "pdays", "previous"]):
    ax[index].boxplot(train[col].dropna())
    ax[index].set_title(f"boxplot for {col}")
plt.tight_layout()    
plt.show()


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
for index, col in enumerate(["balance", "pdays", "previous"]):
    ax[index].violinplot(train[col])
    ax[index].set_title(f"violinplot for {col}")
plt.tight_layout()    
plt.show()


from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import seaborn as sns

for col in ["balance", "pdays", "previous"]:
    ecdf = ECDF(train[col].dropna())
    x = np.linspace(train[col].min(), train[col].max(), 500)
    
    plt.figure(figsize=(6,4))
    plt.plot(x, ecdf(x))
    plt.title(f"ECDF of {col}")
    plt.xlabel(col)
    plt.ylabel("Cumulative probability")
    plt.grid(True)
    plt.show()


train['pdays'].value_counts().sort_values(ascending=False)


# Features engineering
data = [train, test]
for df in data:
    df['was connectes'] = (df['pdays'] != -1).astype(int) 
    df['pdays_actual'] = df['pdays'].replace(-1, 0)



# numerical features
train_new_columns = train[['was connectes', 'pdays_actual']]
train_numerical_features = pd.concat([numerical_features, train_new_columns], axis=1).drop('pdays', axis=1)

test_new_columns = test[train_new_columns.columns]
test_numerical_features = pd.concat([test[numerical_features.columns], test_new_columns], axis=1).drop('pdays', axis=1)


# Transformation
skew_values = train_numerical_features.skew()
mask = skew_values.abs() > 0.8
skew_features = skew_values[mask].index

pt = PowerTransformer(method='yeo-johnson')
train_numerical_features[skew_features] = pt.fit_transform(train_numerical_features[skew_features])

# transform only on TEST to avoid date leaking
test_numerical_features[skew_features] = pt.transform(test_numerical_features[skew_features])

print('Train skew_features:')
print(train_numerical_features[skew_features].skew().sort_values(ascending=False), '\n')


print('Test skew_features:')
print(test_numerical_features[skew_features].skew().sort_values(ascending=False))



# Scaling
all_numerical_cols = train_numerical_features.columns

scaler = StandardScaler()
train_numerical_features[all_numerical_cols] = scaler.fit_transform(train_numerical_features[all_numerical_cols])
test_numerical_features[all_numerical_cols] = scaler.transform(test_numerical_features[all_numerical_cols])


# Encoding
train['poutcome'].unique()

train_categories_features = train.select_dtypes(exclude ='int').copy()
test_categories_features = test.select_dtypes(exclude ='int').copy()

edu_mapping = {
    'unknown': 0,
    'primary': 1,
    'secondary': 2,
    'tertiary': 3
}

month_mapping = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12
}


for df in (train_categories_features, test_categories_features):
    df['education'] = df['education'].map(edu_mapping)
    df['month'] = df['month'].map(month_mapping)



binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    train_categories_features[col] = (train_categories_features[col] == 'yes').astype(int)
    test_categories_features[col] = (test_categories_features[col] == 'yes').astype(int)



# Target encoding
target_encode_cols = ['contact', 'poutcome', 'job', 'marital']

target_encoder = ce.TargetEncoder(cols=target_encode_cols)

train_categories_features = target_encoder.fit_transform(train_categories_features, train['y'])
test_categories_features = target_encoder.transform(test_categories_features)


# assemble the updated features
X_train = pd.concat([train_categories_features, train_numerical_features], axis=1)
X_test = pd.concat([test_categories_features, test_numerical_features], axis=1)
y_train = train['y']


# Security check - align train with test
X_test, X_train = X_test.align(X_train, join='right', axis=1, fill_value=0)

# Security check - comparing train and test features - their names and order
X_train.columns.equals(X_test.columns)

# last check
True if (X_train.columns == X_test.columns).all() else False


# SVM model
X_small, _, y_small, _ = train_test_split(
    X_train, y_train,
    train_size=0.003,     
    random_state=42,
    stratify=y_train    
)

param_dist = {
    'C': [1, 3],
    'gamma': [0.05, 0.1],
    'kernel': ['rbf'],
}

svm_model = SVC(probability=True, random_state=42)

grid_search = RandomizedSearchCV(
    estimator=svm_model, 
    param_distributions=param_dist,
    n_iter=4,
    scoring='roc_auc', 
    cv=3,               
    n_jobs=1,           
    verbose=2,
    return_train_score=True
)

grid_search.fit(X_small, y_small)

grid_search.best_params_
grid_search.best_score_
results = pd.DataFrame(grid_search.cv_results_)
results[['params', 'mean_train_score', 'mean_test_score']]


# lightGBM model

categorical_cols = train.select_dtypes(include='object').columns

for col in categorical_cols:
    train[col] = train[col].astype('category')

for col in categorical_cols:
    test[col] = test[col].astype('category')

categorical_features = list(categorical_cols)


params = {
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 8, 12],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 400, 800],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}


X_train = train.drop(['y', 'id'], axis=1)
y_train = train['y']

SAMPLE_RATIO = 0.005  

X_sample, _, y_sample, _ = train_test_split(
    X_train, y_train,
    train_size=SAMPLE_RATIO,
    stratify=y_train,
    random_state=42
)
print("Sample size:", X_sample.shape, y_sample.shape)


lgb_model = lgb.LGBMClassifier(
    random_state=42,
    objective='binary',
    metric='auc',
    verbosity=-1,
    **{'disable_default_logging': True}
)

search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=params,
    n_iter=20,                # בניגוד ל־100+ ב־grid
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0,
    return_train_score=True
)
search.fit(
    X_sample,
    y_sample,
    categorical_feature=categorical_features,
    callbacks=[log_evaluation(period=0)]
)



results = pd.DataFrame(search.cv_results_)

stability_table = results[[
    'mean_train_score',
    'std_train_score',
    'mean_test_score',
    'std_test_score',
    'params'
]]

print(stability_table.sort_values('mean_test_score', ascending=False))

