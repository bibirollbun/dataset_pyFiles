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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s3e13/train.csv')


df_train.head()


df_train.info()


df_train.describe()


df_train.isnull().sum()


df_train.describe(include="O")


df_train.drop('id', inplace=True, axis=1)


df_train['prognosis'].value_counts()


from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
df_train['prognosis'] = le.fit_transform(df_train['prognosis'])


plt.figure(figsize=(8,6))
sns.heatmap(df_train.corr(), cmap="cool")
plt.show()


X = df_train.drop(['prognosis'], axis=1)
y = df_train['prognosis']


from sklearn.model_selection import train_test_split


X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.25 , random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=1000,
        min_samples_split=8,
        max_features=5,
        max_depth=15,
        random_state=42
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)


    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)


    model_train_accuracy = accuracy_score(y_train, y_train_pred)
    model_train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    model_train_precision = precision_score(y_train, y_train_pred, average='macro')
    model_train_recall = recall_score(y_train, y_train_pred, average='macro')
    model_train_rocauc_score = roc_auc_score(y_train, y_train_proba, multi_class='ovr')


    model_test_accuracy = accuracy_score(y_test, y_test_pred)
    model_test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    model_test_precision = precision_score(y_test, y_test_pred, average='macro')
    model_test_recall = recall_score(y_test, y_test_pred, average='macro')
    model_test_rocauc_score = roc_auc_score(y_test, y_test_proba, multi_class='ovr')

    # Print results
    print(name)
    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))
    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))

    print('----------------------------------')

    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))

    print('='*35)
    print('\n')

