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


df=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")


df.shape


df1=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


df1.shape


df.head()


df1.head()


df.info()


df.describe()


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='y', data=df)
plt.title('Target Variable Distribution (Subscription: 1=Yes, 0=No)')
plt.show()


import gc
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# optional models
from xgboost import XGBClassifier
df.isna().sum()




# Numeric distributions
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
df[num_cols].hist(figsize=(12,8), bins=20)
plt.tight_layout()
plt.show()



# Target vs numeric (boxplots / violin)
for c in ['age','balance','duration','campaign']:
    plt.figure(figsize=(6,3))
    sns.boxplot(x='y', y=c, data=df)
    plt.title(f'{c} by target')
    plt.show()


 #Categorical counts (top categories)
cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']
for c in cat_cols:
    plt.figure(figsize=(10,3))
    order = df[c].value_counts().index
    sns.countplot(x=c, data=df, order=order)
    plt.title(c)
    plt.xticks(rotation=45)
    plt.show()


df.columns


# Keep ids
train_ids = df['id'] if 'id' in df.columns else df.index
test_ids  = df1['id'] if 'id' in df1.columns else df1.index

# Mark train/test and combine
df['is_train'] = 1
df1['is_train'] = 0
df1['y'] = np.nan  # placeholder

combined = pd.concat([df, df1], ignore_index=True)
print("Combined shape:", combined.shape)

# Map binary columns
for c in ['default','housing','loan']:
    combined[c] = combined[c].map({'yes':1, 'no':0})
    
# Fix month -> numeric ordering (optional) for some models:
month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
if 'month' in combined.columns:
    combined['month_num'] = combined['month'].map(month_map)

# If pdays uses -1 as 'never called', keep it but create flag
combined['pdays_never'] = (combined['pdays'] == -1).astype(int)

# One-hot encode categorical columns (drop_first to avoid collinearity)
cat_cols = ['job','marital','education','contact','poutcome']  # month handled above
combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

# If 'id' exists we will drop it later before training
print("After encoding:", combined.shape)



# Split
train_p = combined[combined['is_train']==1].copy()
test_p  = combined[combined['is_train']==0].copy()

# drop helper columns
drop_cols = ['is_train','id'] if 'id' in combined.columns else ['is_train']
if 'month' in train_p.columns: drop_cols += ['month']   # drop original month if month_num used
if 'y' in test_p.columns:
    test_p.drop(columns=['y'], inplace=True)

train_p['y'] = train_p['y'].astype(int)

# Features and target
X = train_p.drop(columns=['y'] + drop_cols, errors='ignore')
y = train_p['y'].copy()
X_test = test_p.drop(columns=drop_cols, errors='ignore')

print("X shape", X.shape, "X_test shape", X_test.shape)



print("Target value counts:\n", y.value_counts())




X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fill any remaining missing numeric values (if present)
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
imputer = SimpleImputer(strategy='median')
X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
X_val[num_cols]   = imputer.transform(X_val[num_cols])
X_test[num_cols]  = imputer.transform(X_test[num_cols])

# Scale numeric columns (optional for tree models you can skip scaling)
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols]   = scaler.transform(X_val[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

# Train RandomForest baseline
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# Validate
y_val_proba = rf.predict_proba(X_val)[:,1]
y_val_pred  = rf.predict(X_val)

print("Validation ROC AUC:", roc_auc_score(y_val, y_val_proba))
print("\nClassification report:")
print(classification_report(y_val, y_val_pred))



# Predict on test data
y_test_pred = rf.predict(X_test)



# Create submission DataFrame
submission = pd.DataFrame({
    'id': df1['id'],   # ensure this matches your test file column name
    'y': y_test_pred    # your predicted labels
})



# Save submission file
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved successfully!")



submission.head()

