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


train_df = pd.read_csv('/kaggle/input/playground-series-s3e4/train.csv')


train_df.head()


train_df.corr()


import seaborn as sns


sns.pairplot(train_df[['V1','V2']])


X = train_df.iloc[:, 2:30]


X.head()


y = train_df['Class']


y


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from xgboost import XGBClassifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


accuracy


from sklearn.linear_model import LogisticRegression
LogisticRegressionModel = LogisticRegression(max_iter=1000)
LogisticRegressionModel.fit(X_train, y_train)

y_pred_LG = LogisticRegressionModel.predict(X_test)
accuracy_LG = accuracy_score(y_test, y_pred_LG)
accuracy_LG


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


print(classification_report(y_test, y_pred_LG))


# Precision = True Positive / (True Positive + False Positive)
# interpretation: Of all the instances predicted as positive, how many were truely positive?
# Recall = True Positive / (True Positive + False Negatives)
# interpretation: Of all the actual positive cases, how many did the model identify correctly?


# Combine X and y into one DataFrame
df_combined = pd.concat([X, y], axis=1)

# Separate classes
majority_class = df_combined[df_combined['Class'] == 0]
minority_class = df_combined[df_combined['Class'] == 1]

# Downsample majority class
majority_downsampled = majority_class.sample(n=len(minority_class), random_state=42)

# Combine and shuffle
df_balanced = pd.concat([majority_downsampled, minority_class]).sample(frac=1, random_state=42)

# Separate features and target again
X_balanced = df_balanced.drop('Class', axis=1)
y_balanced = df_balanced['Class']


X_balanced.describe()


y_balanced.describe()

