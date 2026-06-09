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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')


train_df.head()


train_df = train_df.drop('id', axis = 1)


train_df.head()


train_df.nunique()


train_df.info()


train_df.isnull().sum()


train_df['job'].unique()


import seaborn as sns
import matplotlib.pyplot as plt


train_df.hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()


train_df['y'].value_counts(normalize=True) * 100


numerical_df = train_df.select_dtypes(include=['number'])
sns.heatmap(numerical_df.corr(), annot = True )


sns.violinplot(x='y', y='duration', data=train_df)


X = train_df.drop('y', axis = 1)
y = train_df['y']


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape


train_df.info()


train_df.nunique()


from sklearn.preprocessing import LabelEncoder
le_default = LabelEncoder()
le_loan = LabelEncoder()
le_housing = LabelEncoder()
train_df['default'] = le_default.fit_transform(train_df['default'])
train_df['loan'] = le_loan.fit_transform(train_df['loan'])
train_df['housing'] = le_housing.fit_transform(train_df['housing'])

categorical_cols = ['poutcome', 'month', 'contact', 'job', 'education', 'marital']
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)



train_df.info()


!pip install imbalanced-learn==0.9.1
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd

X = train_df.drop('y', axis=1).copy()
X = X.astype({col: int for col in X.select_dtypes('bool').columns})

y = train_df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

from collections import Counter
print("Before SMOTE:", Counter(y_train))
print("After SMOTE: ", Counter(y_train_resampled))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))




