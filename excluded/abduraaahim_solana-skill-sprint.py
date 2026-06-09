# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/solana-skill-sprint-memcoin-graduation/train.csv', index_col=0)
train_df.head()


print(f"shape {train_df.shape}\n")
print(f'info\n{train_df.info()}\n')
print(f'describe\n{train_df.describe()}')


train_df['mint'].unique()


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
train_df['slot_graduated'] = imputer.fit_transform(train_df[['slot_graduated']])


train_df.head(10)


train_df['is_valid'].value_counts()


train_df['has_graduated'] = train_df['has_graduated'].map({True:0,False:1})
train_df['is_valid'] = train_df['is_valid'].map({True:0})


train_df.head()


train_df.info()


train_df.corrwith(train_df['has_graduated'], numeric_only=True).abs().sort_values(ascending=False)


train_df.isna().sum()


train_df['is_valid'] = train_df['is_valid'].astype('float')


X = train_df.drop(['has_graduated', 'mint', 'slot_graduated'], axis=1)
y = train_df['has_graduated']


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_prepared = scaler.fit_transform(X)


X.shape, X_prepared.shape


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


models = {
    'logistic_regression':LogisticRegression(max_iter=1000),
    'decision_tree':DecisionTreeClassifier(),
    'random_forest':RandomForestClassifier(),
    'xgboost':XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'lightgbm':LGBMClassifier(),
    'svm':SVC(),
    'knn':KNeighborsClassifier(),
    'mlp':MLPClassifier(max_iter=1000)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    results[name] = accuracy

    


results


max_key = max(results, key=results.get)
print(max_key) 


results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])

sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.xticks(rotation=90)
plt.title('Model Accuracy Comparison')
plt.tight_layout()
plt.grid()
plt.show()


results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])

sns.lineplot(x='Model', y='Accuracy', data=results_df)
plt.xticks(rotation=90)
plt.title('Model Accuracy Comparison')
plt.tight_layout()
plt.grid()
plt.show()


test_df = pd.read_csv("/kaggle/input/solana-skill-sprint-memcoin-graduation/test_unlabeled.csv", index_col=0)
test_df.head()


test_df.shape


test_df.head()


test_df['is_valid'].value_counts()


test_df['is_valid'] = test_df['is_valid'].map({True:0, False:1})


test_df_prepared = test_df.drop('mint', axis=1)


test_df_prepared = scaler.fit_transform(test_df_prepared)


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)


submission_pred = lr_model.predict_proba(test_df_prepared)


submission = pd.DataFrame({
    'mint':test_df['mint'],
    'has_graduated':submission_pred[:,1]
})

submission


submission.to_csv('submission.csv', index=False)




