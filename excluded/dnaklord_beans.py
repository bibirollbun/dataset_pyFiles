!pip install imbalanced-learn==0.10.1


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np


df = pd.read_csv('/kaggle/input/squid-beans/train.csv')
df.head()


df.corr()['Class']


X_train = df[['MajorAxisLength', 'MinorAxisLength']]
X_train.head()


y_train = df['Class']
y_train


df['Class'].value_counts()


ros = RandomOverSampler()
X_train, y_train = ros.fit_resample(X_train, y_train)
y_train.value_counts()


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)


param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(model, param_grid, scoring='f1_macro', cv=5)
grid.fit(X_train, y_train)


df = pd.read_csv('/kaggle/input/squid-beans/test.csv')
X_test = df[['MajorAxisLength', 'MinorAxisLength']]
X_test = scaler.transform(X_test)
y_pred = grid.predict(X_test)
df['Class'] = y_pred
df[['Bean ID', 'Class']].to_csv('submission.csv', index=False)

