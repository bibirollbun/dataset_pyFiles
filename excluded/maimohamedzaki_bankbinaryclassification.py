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


test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test.head()


train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train.head()


sub=pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
sub.head()


train.shape


train.info()


train.duplicated().sum()


train.describe()


import matplotlib.pyplot as plt
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
train[numeric_features].hist(bins=30, figsize=(12,8),color='hotpink')
plt.tight_layout()
plt.show()


import seaborn as sns
corr_matrix = train[numeric_features].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='magma')
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(12,8))
for i, col in enumerate(numeric_features, 1):
    plt.subplot(3, 3, i)
    plt.boxplot(train[col])
    plt.title(col)
plt.tight_layout()
plt.show()


cols_to_check = ['contact', 'poutcome']

for col in cols_to_check:
    print(f"Value counts for {col}:")
    print(train[col].value_counts())
    print("-"*30)


print(train['education'].value_counts())


print(train['job'].value_counts())


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
X = train.drop(['y', 'id'], axis=1)
y = train['y']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Ordinal Encoding 
ordinal_cols = ['education','month']
ordinal_categories = [
    ['primary','secondary','tertiary','unknown'],
    ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
]
ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)
X_train[ordinal_cols] = ordinal_encoder.fit_transform(X_train[ordinal_cols])
X_val[ordinal_cols] = ordinal_encoder.transform(X_val[ordinal_cols])

#  Binary Encoding
binary_cols = ['default','housing','loan']
for col in binary_cols:
    X_train[col] = X_train[col].map({'yes':1, 'no':0})
    X_val[col] = X_val[col].map({'yes':1, 'no':0})

#  Frequency Encoding 
freq_cols = ['contact','poutcome']
freq_maps = {}
for col in freq_cols:
    freq_map = X_train[col].value_counts(normalize=True)
    freq_maps[col] = freq_map
    X_train[col] = X_train[col].map(freq_map)
    X_val[col] = X_val[col].map(freq_map)

#  One-hot Encoding 
onehot_cols = ['job','marital']
X_train = pd.get_dummies(X_train, columns=onehot_cols, drop_first=True)
X_val = pd.get_dummies(X_val, columns=onehot_cols, drop_first=True)

#  Scaling
numeric_features = ['age','balance','day','duration','campaign','pdays','previous']
scaler = RobustScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_val[numeric_features] = scaler.transform(X_val[numeric_features])

# test set 
X_test = test.drop(['id'], axis=1)

# Ordinal
X_test[ordinal_cols] = ordinal_encoder.transform(X_test[ordinal_cols])

# Binary
for col in binary_cols:
    X_test[col] = X_test[col].map({'yes':1,'no':0})

# Frequency
for col in freq_cols:
    X_test[col] = X_test[col].map(freq_maps[col]).fillna(0)  # fillna للـ unseen categories

# One-hot
X_test = pd.get_dummies(X_test, columns=onehot_cols, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Scaling
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)



from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
xgb_model = XGBClassifier(
    n_estimators=200,     
    learning_rate=0.1,    
    max_depth=4,          
    random_state=42,
    use_label_encoder=False,  
    eval_metric='logloss'      
)
xgb_model.fit(X_train, y_train)
y_val_proba = xgb_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_proba)
print("Validation AUC:", val_auc)


y_test_proba = xgb_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    'id': test['id'],
    'y': y_test_proba
})
submission.to_csv('submission.csv', index=False)
submission.head()


