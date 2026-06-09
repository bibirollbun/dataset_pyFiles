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


import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


# read the training and testing dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col="id")


# check the first 5 rows of training dataset
train.head()


cat_cols = train.select_dtypes(include=['object']).columns
num_cols = train.select_dtypes(include=['int64', 'float64']).columns

train.replace([np.inf, -np.inf], np.nan, inplace=True)
train[num_cols] = train[num_cols].fillna(train[num_cols].mean())

for col in cat_cols:
    if train[col].isnull().any():
        train[col] = train[col].fillna(train[col].mode()[0])


cat_cols = test.select_dtypes(include=['object']).columns
num_cols = test.select_dtypes(include=['int64', 'float64']).columns

test.replace([np.inf, -np.inf], np.nan, inplace=True)
test[num_cols] = test[num_cols].fillna(test[num_cols].mean())

for col in cat_cols:
    if test[col].isnull().any():
        test[col] = test[col].fillna(test[col].mode()[0])


print(f"The categorical value columns are: {cat_cols.values}")


le = LabelEncoder()
for col in cat_cols.values:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])


train.head()


train[num_cols] = train[num_cols].astype(int)
test[num_cols] = test[num_cols].astype(int)


train['social_energy'] = train['Social_event_attendance'] * (1 - train['Drained_after_socializing'])
train['online_social_ratio'] = train['Post_frequency'] / (train['Friends_circle_size'] + 1)
train['outside_to_alone'] = train['Going_outside'] / (train['Time_spent_Alone'] + 1)

test['social_energy'] = test['Social_event_attendance'] * (1 - test['Drained_after_socializing'])
test['online_social_ratio'] = test['Post_frequency'] / (test['Friends_circle_size'] + 1)
test['outside_to_alone'] = test['Going_outside'] / (test['Time_spent_Alone'] + 1)


train.head()


sns.histplot(train['Personality'])
plt.title('Count of Personality')
plt.show()


plt.figure(figsize=(8, 2 * len(cat_cols)))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(1, len(cat_cols), i)  
    sns.countplot(x=train[col], hue=train['Personality'], palette='husl')
    plt.title(f"{col} vs Personality count") 
    
plt.tight_layout()
plt.show()


n_plots = len(num_cols)
cols_per_row = math.ceil(n_plots / 2)

plt.figure(figsize=(4 * cols_per_row, 6))

for i, col in enumerate(num_cols, 1):
    plt.subplot(2, cols_per_row, i)
    sns.histplot(x=col, hue='Personality', data=train, fill=True, palette='husl')
    plt.title(f"{col} vs Personality count")

plt.tight_layout()
plt.show()



feature_eng_cols = ['social_energy', 'online_social_ratio', 'outside_to_alone']
plt.figure(figsize=(12, 1 * len(feature_eng_cols)))

for i, col in enumerate(feature_eng_cols, 1):
    plt.subplot(1, len(feature_eng_cols), i)  
    sns.kdeplot(x=col, hue='Personality', data=train, fill=True, palette='husl')
    plt.title(f"{col} vs Personality density") 
    
plt.tight_layout()
plt.show()


X = train.drop('Personality', axis=1)
y = train['Personality']
y = le.fit_transform(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)


X_scaled = pd.DataFrame(X_scaled, columns=list(X.columns.values))
test_scaled = pd.DataFrame(test_scaled, columns=list(test.columns.values))


X_scaled.head()


# splitting the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


param_distributions = {
    'n_estimators': [100, 150, 200, 500],
    'max_depth': [None, 5, 6, 4],  
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],  
    'criterion': ['gini', 'entropy', 'log_loss'],  
}


model = RandomForestClassifier()


random_search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_distributions, 
    n_iter=50, cv=3, 
    scoring='accuracy', 
    random_state=42, 
    n_jobs=-1)


random_search.fit(X_train, y_train)


best_params = random_search.best_params_ 
print(f'Best parameters: {best_params}')


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = oof_preds = np.empty(len(X_train), dtype=y_train.dtype)
models = []

y_train = pd.Series(y_train)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"Fold {fold + 1}")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = RandomForestClassifier(
        **best_params,
        random_state=42
    )

    model.fit(X_tr, y_tr)

    oof_preds[val_idx] = model.predict(X_val)
    acc = accuracy_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")

    models.append(model)


y_pred = model.predict(X_test)
accuracy_score(y_pred, y_test)


print(classification_report(y_pred, y_test))


test_pred = model.predict(test_scaled)
test_pred_labels = le.inverse_transform(test_pred)

sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission = pd.DataFrame({
    'id': sub['id'],
    'Personality': test_pred_labels
})

submission.to_csv('submission.csv', index=False)




