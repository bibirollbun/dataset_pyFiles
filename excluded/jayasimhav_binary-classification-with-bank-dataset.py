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


# Cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Models & Utilities
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)



train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col='id')


print(train.info())
print("\nMissing values in train:\n", train.isnull().sum())
print(test.info())
print("\nMissing values in test:\n", test.isnull().sum())


sns.countplot(x='y', data=train)
plt.title('Distribution of Target y')
plt.show()

train['y'].value_counts(normalize=True).plot.pie(autopct='%.2f%%', label='')
plt.title('Target y Proportion')
plt.ylabel('')
plt.show()

print(train['y'].value_counts())



print(train.describe())
cat_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
for col in cat_features:
    print("\nValue counts for", col)
    print(train[col].value_counts())



num_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
corr = train[num_features + ['y']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='RdBu_r')
plt.title('Correlation Heatmap')
plt.show()



for col in cat_features:
    plt.figure()
    sns.countplot(y=col, data=train, order=train[col].value_counts().index)
    plt.title(f"{col} Distribution")
    plt.show()



# Fill -1 or 'unknown' as a new category
for col in cat_features + ['month']:
    for df in [train, test]:
        df[col] = df[col].astype(str).replace({'unknown': 'Unknown', '-1': 'Unknown'})

# Impute missing numerical values if present
for col in num_features:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(train[col].median())

# Label Encoding for categorical columns
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

# Standardize numerical features
scaler = StandardScaler()
train[num_features] = scaler.fit_transform(train[num_features])
test[num_features] = scaler.transform(test[num_features])

features = cat_features + num_features
X_train = train[features].values
y_train = train['y'].values
X_test = test[features].values



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


models = {
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42, n_jobs=-1)
,
    # Enable GPU for XGBoost
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, 
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42,
        tree_method='gpu_hist'        # <-- GPU support
    ),
    # Enable GPU for LightGBM
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200, 
        random_state=42,
        device='gpu',                 # <-- GPU support
        gpu_platform_id=0,
        gpu_device_id=0
    )
}

scores = {}

for name, model in models.items():
    fold_scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_val)[:,1]
        score = roc_auc_score(y_val, y_pred)
        fold_scores.append(score)
    mean_score = np.mean(fold_scores)
    scores[name] = mean_score
    print(f"{name} mean ROC AUC: {mean_score:.4f}")
print("All scores:", scores)


voting_clf = VotingClassifier(
    estimators=[
        ('lr', models['LogisticRegression']),
        ('rf', models['RandomForest']),
        ('xgb', models['XGBoost']),
        ('lgb', models['LightGBM'])
    ],
    voting='soft'
)

ensemble_scores = []
for train_idx, val_idx in skf.split(X_train, y_train):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    voting_clf.fit(X_tr, y_tr)
    y_pred = voting_clf.predict_proba(X_val)[:,1]
    score = roc_auc_score(y_val, y_pred)
    ensemble_scores.append(score)
ensemble_mean = np.mean(ensemble_scores)
print(f"Ensemble Voting mean ROC AUC: {ensemble_mean:.4f}")



rf = models['RandomForest']
rf.fit(X_train, y_train)
importances = rf.feature_importances_
feat_imp = pd.DataFrame({'feature': features, 'importance': importances})
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feat_imp)
plt.title('RandomForest Feature Importance')
plt.show()



# Train ensemble on full train and predict test
voting_clf.fit(X_train, y_train)
y_test_pred = voting_clf.predict_proba(X_test)[:,1]


submission = pd.DataFrame({'id': test.index, 'y': y_test_pred})
print(submission.head())
submission.to_csv('submission.csv', index=False)




y_train_pred = voting_clf.predict_proba(X_train)[:,1]
fpr, tpr, _ = roc_curve(y_train, y_train_pred)
plt.plot(fpr, tpr, label='VotingClassifier')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve - All Training Data')
plt.legend()
plt.show()

print('Training data ROC AUC:', roc_auc_score(y_train, y_train_pred))





