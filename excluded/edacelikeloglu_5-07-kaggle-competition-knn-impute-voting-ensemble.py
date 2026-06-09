import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from tqdm.notebook import tqdm
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import RepeatedStratifiedKFold


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')



train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train.shape, test.shape


train.info()


test.info()


features = [col for col in train.columns if col not in ['id', 'Personality']]

train['source'] = 'train'
test['source'] = 'test'
test['Personality'] = None  

combined = pd.concat([train, test], ignore_index=True)

valid_features = [col for col in train.columns if col not in ['id', 'Personality', 'source']]

for col in valid_features:
    plt.figure(figsize=(10, 4))
    if combined[col].dtype in ['float64', 'int64']:
        sns.kdeplot(data=combined, x=col, hue='source', common_norm=False, fill=True)
        plt.title(f"{col} - Sayısal Dağılım (train vs test)")
    else:
        sns.countplot(data=combined, x=col, hue='source')
        plt.title(f"{col} - Kategorik Dağılım (train vs test)")
        plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(data=train, x='Personality')
plt.title("Target - Personality Dağılımı (Sadece train)")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

train.drop('source', axis=1, inplace=True)
test.drop('source', axis=1, inplace=True)


for col in valid_features:
    plt.figure(figsize=(10, 4))
    if train[col].dtype in ['float64', 'int64']:
        sns.kdeplot(data=train, x=col, hue='Personality', fill=True, common_norm=False)
        plt.title(f"{col} - Sayısal Dağılım (Introvert vs Extrovert)")
    else:
        sns.countplot(data=train, x=col, hue='Personality')
        plt.title(f"{col} - Kategorik Dağılım (Introvert vs Extrovert)")
        plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


for col in ['Stage_fear', 'Drained_after_socializing', 'Personality']:
    train[col] = train[col].astype(str).str.strip().str.capitalize()
    test[col] = test[col].astype(str).str.strip().str.capitalize()


binary_map = {'Yes': 1, 'No': 0}
train['Stage_fear'] = train['Stage_fear'].map(binary_map)
train['Drained_after_socializing'] = train['Drained_after_socializing'].map(binary_map)
test['Stage_fear'] = test['Stage_fear'].map(binary_map)
test['Drained_after_socializing'] = test['Drained_after_socializing'].map(binary_map)

target_map = {'Extrovert': 1, 'Introvert': 0}
train['Personality'] = train['Personality'].map(target_map)

ordinal_features = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]

train.dtypes, test.dtypes


missing_ratios_train = train.isnull().mean().sort_values(ascending=False) * 100
missing_ratios_train = missing_ratios_train[missing_ratios_train > 0].round(2)

print("Train missing ratio: ", missing_ratios_train)

missing_ratios_test = test.isnull().mean().sort_values(ascending=False) * 100
missing_ratios_test = missing_ratios_test[missing_ratios_test > 0].round(2)

print("Test missing ratio: ", missing_ratios_test)


features = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency',
    'Stage_fear',
    'Drained_after_socializing'
]

train_ids = train['id'].reset_index(drop=True)
test_ids = test['id'].reset_index(drop=True)
train_target = train['Personality'].reset_index(drop=True)

combined_features = pd.concat([train[features], test[features]], ignore_index=True)

imputer = KNNImputer(n_neighbors=5)
imputed = pd.DataFrame(imputer.fit_transform(combined_features), columns=features)

imputed_train = imputed.iloc[:len(train)].reset_index(drop=True)
imputed_test = imputed.iloc[len(train):].reset_index(drop=True)

final_train = pd.concat([train_ids, imputed_train, train_target], axis=1)
final_test = pd.concat([test_ids, imputed_test], axis=1)

print("Train NaNs:\n", final_train.isnull().sum())
print("Test NaNs:\n", final_test.isnull().sum())


ordinal_features = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]
for col in ordinal_features:
    final_train[col] = final_train[col].round().astype(int)
    final_test[col] = final_test[col].round().astype(int)

binary_features = ['Stage_fear', 'Drained_after_socializing']
for col in binary_features:
    final_train[col] = final_train[col].round().clip(0, 1).astype(int)
    final_test[col] = final_test[col].round().clip(0, 1).astype(int)



X = final_train.drop(columns=['id', 'Personality'])
y = final_train['Personality']

# GridSearch parametreleri
param_grids = {
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
        }
    },
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
        }
    }
}

grid_results = {}

for name in tqdm(param_grids):
    clf = GridSearchCV(
        estimator=param_grids[name]['model'],
        param_grid=param_grids[name]['params'],
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=0
    )
    clf.fit(X, y)
    grid_results[name] = {
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    }

grid_results


xgb_clf = XGBClassifier(
    learning_rate=0.05, max_depth=3, n_estimators=100,
    use_label_encoder=False, eval_metric='logloss', random_state=42
)

lgbm_clf = LGBMClassifier(
    learning_rate=0.05, max_depth=3, n_estimators=100, random_state=42
)

gb_clf = GradientBoostingClassifier(
    learning_rate=0.05, max_depth=3, n_estimators=200, random_state=42
)

voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_clf),
        ('lgbm', lgbm_clf),
        ('gb', gb_clf)
    ],
    voting='hard',  
    n_jobs=-1
)

voting_clf.fit(X, y)

final_predictions = voting_clf.predict(final_test.drop(columns=['id']))

submission = pd.DataFrame({
    'id': final_test['id'],
    'Personality': final_predictions
})

submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
submission.to_csv("submission_voting_ensemble.csv", index=False)

